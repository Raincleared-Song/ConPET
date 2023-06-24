import os
import copy
import torch
import shutil
from tqdm import tqdm
import loralib as lora
from loss_similarity import LossSimilarity
from models import mark_only_adapter_lora_as_trainable, BertLoRAWithSelector
from .testing import valid_save, test_by_best
from preprocess import get_tag_set_by_dataset
from utils import update_tag_loss_count


def train_selector(config, data_loaders, model, tokenizer, loss_sim: LossSimilarity,
                   optimizer, scheduler, trained_epoch, global_step, best_results=None):
    # mark only model selector as trainable
    if isinstance(model, BertLoRAWithSelector) and config['plm']['apply_lora']:
        for n, p in model.named_parameters():
            p.requires_grad = 'bert_selector' in n or 'lora_linear_selector' in n or 'lora_alignment' in n

    exp_path = os.path.join(config['logging']['path_base'], config['logging']['unique_string'])
    model_path = os.path.join(exp_path, 'models')
    os.makedirs(model_path, exist_ok=True)
    assert config['task'] == 'fewshot'
    train_loader, valid_groups, test_groups, train_infer = data_loaders['train_infer'], \
        data_loaders['valid_groups'], data_loaders['test_groups'], data_loaders['train_infer']

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
    if config['train']['save_option'] >= 1 and config['plm']['model_name'] != 'cpm':
        tokenizer.save_pretrained(model_path)

    train_tag_set = list(get_tag_set_by_dataset(train_loader))
    print('label number in training set:', len(train_tag_set), '......')

    tag_to_loss_weight = {tag: 1. for tag in train_tag_set}
    tag_to_loss_count = {tag: [0, 0.] for tag in train_tag_set}  # tag -> [count, total_loss]

    step_per_epoch = len(train_loader) // grad_accu_step
    if global_step % step_per_epoch != 0:
        trained_epoch -= 1
    step_to_skip = global_step - trained_epoch * step_per_epoch
    assert 0 <= step_to_skip < step_per_epoch
    step_to_skip *= grad_accu_step

    # restore scheduler after training
    ori_sche_dict = {}
    if scheduler is not None:
        ori_sche_dict = copy.deepcopy(scheduler.state_dict())

    if config['train']['valid_zero_shot']:
        valid_save(config, model, tokenizer, loss_sim, is_step=True,
                   best_results=best_results, cur_step=global_step, cur_epoch=trained_epoch,
                   optimizer=optimizer, scheduler=scheduler,
                   valid_loader=valid_groups, valid_infer=None, train_infer=train_infer,
                   extra_module_info=None, train_expert_selector=True)

    for epoch in range(trained_epoch, int(num_epoch)):
        if not (config['task'] == 'fewshot' and config['is_test']):
            print(f'training selector epoch {epoch} skipped step {step_to_skip} ......')
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
                mark_only_adapter_lora_as_trainable(model)

            try:
                _, loss, loss_vec = loss_sim.generate_expert_selector_logits('train', model, tokenizer, batch,
                                                                             tag_to_loss_weight)
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
                               extra_module_info=None, train_expert_selector=True)

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
                   extra_module_info=None, train_expert_selector=True)
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
        shutil.copy(os.path.join(model_path, 'exp-epoch-best.pkl'), os.path.join(model_path, 'exp-tot-best.pkl'))
    else:
        best_key = 'step-model'
        shutil.copy(os.path.join(model_path, 'exp-step-best.pkl'), os.path.join(model_path, 'exp-tot-best.pkl'))
    test_results = test_by_best(config, model, tokenizer, loss_sim,
                                best_epoch, best_epoch_acc, best_step, best_step_acc,
                                test_groups, test_infer=None, train_infer=train_infer,
                                best_model=best_results[best_key],
                                extra_module_info=None, train_expert_selector=True)

    if scheduler is not None:
        scheduler.load_state_dict(ori_sche_dict)

    return global_step, average_loss, \
        best_step, best_step_acc, best_epoch, best_epoch_acc, best_results, test_results
