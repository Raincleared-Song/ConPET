import os
import copy
import torch
import shutil
import random
from tqdm import tqdm
import loralib as lora
from loss_similarity import LossSimilarity
from models import mark_only_adapter_as_trainable, get_model_mean_fisher, BertLoRAWithSelector
from .training_selector import train_selector
from .testing import valid_save, test_by_best, test
from utils import load_json, save_json, load_partial_checkpoint, update_tag_loss_count
from preprocess import get_contrastive_loader_by_dataset, get_tag_set_by_dataset


def generate_grad(config, train_loader, model, tokenizer, loss_sim,
                  type_embeds=None, type_counter=None, tag_to_loss_weight=None):
    print('Calculating EWC grad and fisher ......')
    exp_path = os.path.join(config['logging']['path_base'], config['logging']['unique_string'])
    model_path = os.path.join(exp_path, 'models')
    grad_fisher_path = os.path.join(model_path, 'grad_fisher.pkl')
    grad_mean, grad_fisher = get_model_mean_fisher(
        config, train_loader, model, tokenizer, loss_sim,
        type_embeds=type_embeds, type_counter=type_counter,
        extra_module_info=None, tag_to_loss_weight=tag_to_loss_weight,
    )
    torch.save((grad_mean, grad_fisher), grad_fisher_path)


@torch.no_grad()
def generate_logit(config, train_loader, model, tokenizer, loss_sim):
    print('Calculating LWF logits ......')
    exp_path = os.path.join(config['logging']['path_base'], config['logging']['unique_string'])
    model_path = os.path.join(exp_path, 'models')
    lwf_logit_path, generated_logits = os.path.join(model_path, 'lwf_logit.pkl'), {}
    epoch_iterator = tqdm(train_loader, desc="data iteration") if len(train_loader) > 20 else train_loader
    model.eval()
    for batch in epoch_iterator:
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(config['device'], non_blocking=True)
        with torch.no_grad():
            logits = loss_sim.generate_continual_logits('test', model, tokenizer, batch, lwf_logit=True)
        assert len(batch['sample_keys']) == len(logits)
        for sample_key, logit in zip(batch['sample_keys'], logits):
            generated_logits[sample_key] = logit.cpu()
    torch.save(generated_logits, lwf_logit_path)


def train(config, data_loaders, model, tokenizer, loss_sim: LossSimilarity,
          optimizer, scheduler, trained_epoch, global_step, best_results=None,
          type_embeds=None, type_counter=None, extra_module_info=None):
    # train selector from initialization
    if config['train']['train_expert_selector'] == 2 and config['train']['use_expert_selector']:
        exp_global_step, exp_average_loss, exp_best_step, exp_best_step_acc, \
            exp_best_epoch, exp_best_epoch_acc, exp_valid_results, exp_results = \
            train_selector(config, data_loaders, model, tokenizer, loss_sim, optimizer, scheduler, 0, 0, None)
        print('selector global step:', exp_global_step)
        print('selector average loss:', exp_average_loss)
        print('selector best step:', exp_best_step, 'selector best step acc:', exp_best_step_acc)
        print('selector best epoch:', exp_best_epoch, 'selector best epoch acc:', exp_best_epoch_acc)
        print('selector test results:', exp_results)

    # mark only model itself as trainable
    if isinstance(model, BertLoRAWithSelector):
        for n, p in model.named_parameters():
            p.requires_grad = 'bert_lora' in n

    exp_path = os.path.join(config['logging']['path_base'], config['logging']['unique_string'])
    model_path = os.path.join(exp_path, 'models')
    os.makedirs(model_path, exist_ok=True)
    train_loader, valid_groups, test_groups, train_infer = data_loaders['train'], \
        data_loaders['valid_groups'], data_loaders['test_groups'], data_loaders['train_infer']
    if config['task'] == 'fewshot':
        train_loader = train_infer
        # if not config['is_test']:
        #     valid_groups, test_groups = None, None

    num_epoch = config['train']['num_epochs']
    save_step = config['train']['save_step']
    grad_accu_step = config['train']['gradient_accumulation_steps']
    loss_adverse_step = config['train']['loss_adverse_step']
    if best_results is None:
        print('WARNING: best_results is None ......')
        best_results = {
            'best_step': -1,
            'best_step_acc': 0.,
            'best_epoch': -1,
            'best_epoch_acc': 0.,
        }
    else:
        print('loaded best results:', str(best_results))
    train_loss = 0.0
    model.zero_grad()
    if config['train']['save_option'] >= 1:
        tokenizer.save_pretrained(model_path)

    train_tag_set = list(get_tag_set_by_dataset(train_loader))
    print('label number in training set:', len(train_tag_set), '......')

    tag_to_loss_weight = {tag: 1. for tag in train_tag_set}
    tag_to_loss_count = {tag: [0, 0.] for tag in train_tag_set}  # tag -> [count, total_loss]

    if config['generate_grad']:
        generate_grad(config, train_loader, model, tokenizer, loss_sim, type_embeds, type_counter, tag_to_loss_weight)
        exit()

    if config['generate_logit']:
        generate_logit(config, train_loader, model, tokenizer, loss_sim)
        exit()

    # generate lwf logits first
    if config['train']['continual_method'] == 'lwf':
        generate_logit(config, train_loader, model, tokenizer, loss_sim)

    step_per_epoch = len(train_loader) // grad_accu_step
    if global_step % step_per_epoch != 0:
        trained_epoch -= 1
    step_to_skip = global_step - trained_epoch * step_per_epoch
    assert 0 <= step_to_skip < step_per_epoch
    step_to_skip *= grad_accu_step

    if config['train']['valid_zero_shot']:
        valid_save(config, model, tokenizer, loss_sim, is_step=True,
                   best_results=best_results, cur_step=global_step, cur_epoch=trained_epoch,
                   optimizer=optimizer, scheduler=scheduler,
                   valid_loader=valid_groups, valid_infer=None, train_infer=train_infer,
                   extra_module_info=extra_module_info, train_expert_selector=False)

    for epoch in range(trained_epoch, int(num_epoch)):
        if not (config['task'] == 'fewshot' and config['is_test']):
            print(f'training epoch {epoch} skipped step {step_to_skip} ......')
        epoch_iterator = tqdm(train_loader, desc="data iteration") if len(train_loader) > 20 else train_loader
        for step, batch in enumerate(epoch_iterator):
            if step < step_to_skip:
                continue

            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(config['device'], non_blocking=True)

            model.train()
            if config['plm']['apply_lora']:
                lora.mark_only_lora_as_trainable(model)
            elif config['plm']['apply_adapter']:
                mark_only_adapter_as_trainable(model)

            try:
                loss, loss_vec = loss_sim.forward_similarity_train(model, tokenizer, batch,
                                                                   type_embeds, type_counter,
                                                                   extra_module_info, tag_to_loss_weight)
                update_tag_loss_count(batch['tags'].cpu().tolist(), loss_vec, tag_to_loss_count)
            except RuntimeError as err:
                from IPython import embed
                embed()
                raise err

            if grad_accu_step > 1:
                loss /= grad_accu_step

            loss.backward()
            train_loss += loss.item()

            if (step + 1) % grad_accu_step == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # 梯度累加
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                model.zero_grad()
                global_step += 1

                if save_step > 0 and global_step % save_step == 0:
                    valid_save(config, model, tokenizer, loss_sim, is_step=True,
                               best_results=best_results, cur_step=global_step, cur_epoch=epoch,
                               optimizer=optimizer, scheduler=scheduler,
                               valid_loader=valid_groups, valid_infer=None, train_infer=train_infer,
                               extra_module_info=extra_module_info, train_expert_selector=False)

            if loss_adverse_step > 0 and (step + 1) % loss_adverse_step == 0:
                cur_tag_loss_vec = [tag_to_loss_count[tag][1] / tag_to_loss_count[tag][0] for tag in train_tag_set]
                cur_tag_loss_vec = torch.softmax(torch.tensor(cur_tag_loss_vec), dim=0) * len(train_tag_set)
                tag_to_loss_weight = {tag: cur_tag_loss_vec[tid].item() for tid, tag in enumerate(train_tag_set)}
                tag_to_loss_count = {tag: [0, 0.] for tag in train_tag_set}

            if 0 < config['train']['num_steps'] < global_step:
                epoch_iterator.close()
                break
        if 0 < config['train']['num_steps'] < global_step:
            break
        valid_save(config, model, tokenizer, loss_sim, is_step=False,
                   best_results=best_results, cur_step=global_step, cur_epoch=epoch,
                   optimizer=optimizer, scheduler=scheduler,
                   valid_loader=valid_groups, valid_infer=None, train_infer=train_infer,
                   extra_module_info=extra_module_info, train_expert_selector=False)
    average_loss = train_loss / global_step if global_step > 0 else 0
    best_step, best_step_acc, best_epoch, best_epoch_acc = best_results['best_step'], best_results['best_step_acc'], \
        best_results['best_epoch'], best_results['best_epoch_acc']
    if test_groups is None:
        return global_step, average_loss, \
            best_step, best_step_acc, best_epoch, best_epoch_acc, best_results, None
    print('got best epoch:', best_epoch, 'accuracy:', best_epoch_acc)
    print('got best step:', best_step, 'accuracy:', best_step_acc)
    if best_epoch_acc > best_step_acc:
        best_key = 'epoch-model'
        shutil.copy(os.path.join(model_path, 'epoch-best.pkl'), os.path.join(model_path, 'tot-best.pkl'))
    else:
        best_key = 'step-model'
        shutil.copy(os.path.join(model_path, 'step-best.pkl'), os.path.join(model_path, 'tot-best.pkl'))
    test_results = test_by_best(config, model, tokenizer, loss_sim,
                                best_epoch, best_epoch_acc, best_step, best_step_acc,
                                test_groups, test_infer=None, train_infer=train_infer,
                                best_model=best_results[best_key],
                                extra_module_info=extra_module_info)
    if config['train']['continual_method'] == 'ewc':
        generate_grad(config, train_loader, model, tokenizer, loss_sim, type_embeds, type_counter, tag_to_loss_weight)
    with open(os.path.join(exp_path, 'flag'), 'w') as fout:
        fout.write('complete\n')
    return global_step, average_loss, \
        best_step, best_step_acc, best_epoch, best_epoch_acc, best_results, test_results


def train_fewshot_model(config, mode, data_groups, model, tokenizer, loss_sim: LossSimilarity, optimizer, scheduler,
                        ori_state_dict, ori_opt_dict, ori_sche_dict):
    correct_num, instance_num = 0, 0
    for item in tqdm(data_groups, desc=mode):
        supports, queries = item['support'], item['query']
        few_valid_ratio = config[mode]['fewshot_valid_ratio']
        support_num = len(supports)
        assert support_num > 1
        few_idx = random.sample(range(support_num), round(support_num * few_valid_ratio))
        few_infer, few_remain = [], []
        for idx in range(support_num):
            if idx in few_idx:
                few_infer.append(supports[idx])
            else:
                few_remain.append(supports[idx])
        few_infer = get_contrastive_loader_by_dataset(config, 'valid', few_infer, tokenizer)
        few_loader = get_contrastive_loader_by_dataset(config, 'test', queries, tokenizer)
        few_remain = get_contrastive_loader_by_dataset(config, 'train', few_remain, tokenizer)
        few_loaders = {
            "train": few_remain, "valid_groups": few_infer, "test_groups": few_loader, "train_infer": few_remain,
        }
        if config['train']['cold_startup']:
            for _ in range(config['train']['fewshot_epoch']):
                _, _, _, _, _, _, valid_results, test_results = train(
                    config, few_loaders, model, tokenizer, loss_sim, optimizer, scheduler, 0, 0)
                if valid_results['correct_num'] > 0:
                    correct_num += test_results["correct_num"]
                    instance_num += test_results["instance_num"]
                    break
        else:
            test_results = test(config, few_loader, None, model, tokenizer, loss_sim, 'test',
                                output_path=None, tag_set=None)[0]
            correct_num += test_results["correct_num"]
            instance_num += test_results["instance_num"]
        assert correct_num > 0 and instance_num > 0
        print('correct:', correct_num, 'instance:', instance_num, 'accuracy:', round(100*correct_num/instance_num, 2))

        model.load_state_dict(ori_state_dict)
        optimizer.load_state_dict(ori_opt_dict)
        if scheduler is not None:
            scheduler.load_state_dict(ori_sche_dict)
    results = {
        "correct_num": correct_num, "instance_num": instance_num,
        "accuracy": round(correct_num * 100 / instance_num, 2),
    }
    return results


def train_fewshot_valid(config, data_loaders, model, tokenizer, loss_sim: LossSimilarity,
                        optimizer, scheduler=None, test_output_path=None):
    exp_path = os.path.join(config['logging']['path_base'], config['logging']['unique_string'])
    model_path = os.path.join(exp_path, 'models')
    valid_groups, test_groups = data_loaders['valid_groups'], data_loaders['test_groups']
    assert config['train']['save_option'] <= 1

    step_model_ls = [pkl for pkl in os.listdir(model_path) if pkl.startswith('step-') and pkl.endswith('.pkl')]
    epoch_model_ls = [pkl for pkl in os.listdir(model_path) if pkl.startswith('epoch-') and pkl.endswith('.pkl')]
    step_model_ls.sort(key=lambda x: int(x[5:-4]))
    epoch_model_ls.sort(key=lambda x: int(x[6:-4]))
    has_zero_shot = 'step-0.pkl' in step_model_ls
    if has_zero_shot:
        assert step_model_ls[0] == 'step-0.pkl'
        step_model_ls = step_model_ls[1:]
    all_model_ls = sorted(step_model_ls + epoch_model_ls, key=lambda x: os.path.getmtime(os.path.join(model_path, x)))
    if has_zero_shot:
        all_model_ls = ['step-0.pkl'] + all_model_ls
    results_path, valid_accuracies, passed_pkl = os.path.join(exp_path, 'fewshot_test_result.json'), [], []
    if os.path.exists(results_path):
        valid_accuracies = load_json(results_path)
        passed_pkl = [item[1] for item in valid_accuracies]
    zero_state_dict, zero_opt_dict, zero_sche_dict = \
        copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()), None
    if scheduler is not None:
        zero_sche_dict = copy.deepcopy(scheduler.state_dict())
    if config['train']['valid_zero_shot']:
        if 'zero_shot' in passed_pkl:
            print(f'fewshot skip zero_shot ......')
        else:
            print(f'fewshot validating zero_shot ......')
            valid_results = train_fewshot_model(config, 'valid', valid_groups, model, tokenizer, loss_sim, optimizer,
                                                scheduler, zero_state_dict, zero_opt_dict, zero_sche_dict)
            valid_accuracies.append((valid_results, 'zero_shot'))
            print(valid_accuracies[-1])
            save_json(valid_accuracies, results_path)
    for pkl in all_model_ls:
        pkl_path = os.path.join(model_path, pkl)
        if pkl in passed_pkl:
            print(f'fewshot skip model {pkl_path} ......')
            continue
        print(f'fewshot validating model {pkl_path} ......')
        ori_state_dict, ori_opt_dict, ori_sche_dict = \
            load_partial_checkpoint(config, pkl_path, model, optimizer, scheduler)

        valid_results = train_fewshot_model(config, 'valid', valid_groups, model, tokenizer, loss_sim, optimizer,
                                            scheduler, ori_state_dict, ori_opt_dict, ori_sche_dict)
        valid_accuracies.append((valid_results, pkl))
        print(valid_accuracies[-1])
        save_json(valid_accuracies, results_path)

    best_valid_res, best_valid_pkl = max(valid_accuracies, key=lambda x: x[0]['accuracy'])
    best_pkl_path = os.path.join(model_path, best_valid_pkl)
    print(f'fewshot testing model {best_pkl_path} ......')
    if best_valid_pkl != 'zero_shot':
        ori_state_dict, ori_opt_dict, ori_sche_dict = \
            load_partial_checkpoint(config, best_pkl_path, model, optimizer, scheduler)
    else:
        ori_state_dict, ori_opt_dict, ori_sche_dict = zero_state_dict, zero_opt_dict, zero_sche_dict

    test_results = train_fewshot_model(config, 'test', test_groups, model, tokenizer, loss_sim, optimizer,
                                       scheduler, ori_state_dict, ori_opt_dict, ori_sche_dict)
    ret = {
        "best_valid_result": best_valid_res,
        "best_valid_pkl": best_valid_pkl,
        "test_fewshot_results": test_results,
    }
    print('fewshot_test_metrics:', ret)
    if test_output_path is not None:
        os.makedirs(test_output_path, exist_ok=True)
        save_json(ret, os.path.join(test_output_path, 'fewshot_test_metrics.json'))
    return ret
