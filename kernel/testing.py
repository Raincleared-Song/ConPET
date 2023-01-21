import os
import torch
from tqdm import tqdm
from global_var import GLOBAL
from torch.utils.data import DataLoader
from loss_similarity import LossSimilarity
from utils import save_json, gather_t5_result, save_model
from models import BertForMaskedLMLoRA
from transformers import T5ForConditionalGeneration, AutoModelForMaskedLM
from preprocess import get_contrastive_loader_by_dataset, get_tag_set_by_dataset


def save_test_results(mode: str, output_path, is_step=False, all_preds=None, all_tags=None,
                      results=None, epoch=-1, step=-1, train_expert_selector=False, is_proto=False):
    assert output_path is not None
    assert mode == 'test' or not is_step and epoch >= 0 or is_step and step >= 0
    assert all_preds is None and all_tags is None or \
           all_preds is not None and all_tags is not None and len(all_preds) == len(all_tags)
    if mode == 'test':
        if is_proto:
            suffix = 'test_' + (('s' + str(step)) if is_step else ('e' + str(epoch)))
        else:
            suffix = 'test'
    elif not is_step:
        suffix = 'e' + str(epoch)
    else:
        suffix = 's' + str(step)
    if train_expert_selector:
        suffix += '_exp'
    os.makedirs(output_path, exist_ok=True)
    if all_preds is not None:
        output_file_name = os.path.join(output_path, f'results_{suffix}.txt')
        fout = open(output_file_name, 'w', encoding='utf-8')
        for pred, label in zip(all_preds, all_tags):
            fout.write(f'{pred}\t{label}\n')
        fout.close()
    if results is not None:
        output_file_name = os.path.join(output_path, f'metrics_{suffix}.json')
        save_json(results, output_file_name)


@torch.no_grad()
def test(config, data_loader, data_infer, model, tokenizer, loss_sim: LossSimilarity, mode: str,
         output_path: str = None, is_step=False, epoch=-1, step=-1, type_embeds=None, type_counter=None,
         tag_set: set = None, extra_module_info: list = None, accu_results=None,
         train_expert_selector=False, prototypes=None):
    model.eval()
    assert mode in ['valid', 'test']

    embed_size, label_num, vocab_size, hidden_size = \
        config['embed_size'], config['label_num'], config['vocab_size'], config['hidden_size']
    if config['task'] in ['contrastive', 'continual']:
        assert not ((type_embeds is None) ^ (type_counter is None))
        if type_embeds is None:
            assert config['task'] != 'continual'
            type_embeds = [torch.zeros(embed_size).to(config['device']) for _ in range(label_num)]
            type_counter = [0 for _ in range(label_num)]
        if data_infer is not None:
            type_embeds, type_counter = get_type_embeddings(config, data_infer, model, tokenizer, loss_sim,
                                                            exist_embeds=type_embeds, exist_counter=type_counter)
        type_embeds = torch.stack(type_embeds)
    target_ones = torch.ones(label_num).to(type_embeds)
    tag_to_split, split_to_tags = GLOBAL['continual_tag_to_split'], GLOBAL['continual_split_to_tags']

    all_preds, all_tags = [], []
    correct_num, instance_num = 0, 0
    correct_num_exp, instance_num_exp = 0, 0
    preds, tags = [], []
    all_preds_exp, all_tags_exp = [], []

    if tag_set is None:
        tag_set = get_tag_set_by_dataset(data_loader)

    data_iterator = tqdm(data_loader, desc=mode) if len(data_loader) > 20 else data_loader
    for batch in data_iterator:
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(config['device'], non_blocking=True)
        if config['task'] in ['contrastive', 'continual']:
            preds, tags = [], batch['tags'].cpu().tolist()
            logits = loss_sim.generate_inference_logits(model, tokenizer, batch)
            for logit in logits:
                logit = logit.unsqueeze(0).expand(label_num, -1)
                assert logit.shape == type_embeds.shape
                losses = loss_sim.forward_similarity_test(logit, type_embeds, target_ones)
                cur_probs = []
                for tag in tag_set:
                    cur_probs.append((tag, losses[tag].item()))
                pred = min(cur_probs, key=lambda x: x[1])[0]
                preds.append(pred)
        elif config['task'] == 'fewshot':
            if config['plm']['model_name'].startswith('t5'):
                preds = tokenizer.batch_decode(model.generate(batch['input_ids']))
                tags = tokenizer.batch_decode(batch['labels'])
                preds = [gather_t5_result(pred) for pred in preds]
                tags = [gather_t5_result(tag) for tag in tags]
            elif not train_expert_selector:
                logits = loss_sim.generate_continual_logits(mode, model, tokenizer, batch,
                                                            extra_module_info, prototypes=prototypes)
                if config['train']['continual_method'] in ['ewc', 'lwf'] and config['is_test']:
                    total_splits = [f'p{idx}' for idx in range(1, int(config['dataset']['total_parts'][1:]) + 1)]
                else:
                    total_splits, _ = loss_sim.get_current_splits(extra_module_info, train_expert_selector=False)
                if config['train']['continual_method'] == 'lwf':
                    total_splits = [f'p{sid+1}' for sid in range(loss_sim.curr_bound)]
                label_mask = batch['input_ids'] == tokenizer.mask_token_id
                target_tokens, token_to_tid = loss_sim.generate_continual_token_map(total_splits)
                if config['train']['verbalizer_strategy'] == 'soft':
                    preds = torch.max(logits, dim=1)[1].cpu().tolist()
                    preds = [target_tokens[pred] for pred in preds]
                    tags = batch['labels'][label_mask].cpu().tolist()
                elif config['train']['verbalizer_strategy'] == 'mean':
                    preds, tags = [], batch['tags'].cpu().tolist()
                    for logit in logits:
                        cur_probs = []
                        for tag in tag_set:
                            cur_probs.append((tag, logit[token_to_tid[tag]]))
                        preds.append(max(cur_probs, key=lambda x: x[1])[0])
                else:
                    raise NotImplementedError(
                        'invalid verbalizer strategy: ' + config['train']['verbalizer_strategy'])
            # for expert selector
            if config['train']['train_expert_selector']:
                expert_logits = loss_sim.generate_expert_selector_logits(mode, model, tokenizer, batch)
                expert_splits, _ = loss_sim.get_current_splits([], train_expert_selector=True)
                _, token_to_tid = loss_sim.generate_continual_token_map(expert_splits)
                assert config['train']['verbalizer_strategy'] == 'mean'
                raw_tags = batch['tags'].cpu().tolist()
                expert_preds, expert_tags = [], [int(tag_to_split[tag][1:]) - 1 for tag in raw_tags]
                for logit in expert_logits:
                    cur_probs = []
                    for split in expert_splits:
                        split_label = int(split[1:]) - 1
                        cur_probs.append((split_label, logit[split_label]))
                    expert_preds.append(max(cur_probs, key=lambda x: x[1])[0])

                assert len(expert_preds) == len(expert_tags)
                for pred, label in zip(expert_preds, expert_tags):
                    correct_num_exp += int(pred == label)
                    instance_num_exp += 1
                all_preds_exp += expert_preds
                all_tags_exp += expert_tags
        else:
            raise NotImplementedError('invalid task name: ' + config['task'])

        assert len(preds) == len(tags)
        for pred, label in zip(preds, tags):
            correct_num += int(pred == label)
            instance_num += 1

        all_preds += preds
        all_tags += tags

    results, to_use_accu = {}, accu_results is not None
    if not train_expert_selector:
        if to_use_accu and 'correct_num' in accu_results:
            correct_num += accu_results['correct_num']
            instance_num += accu_results['instance_num']
        results.update({
            "correct_num": correct_num,
            "instance_num": instance_num,
            "accuracy": round(correct_num * 100 / instance_num, 2),
        })

        total_splits, _ = loss_sim.get_current_splits(extra_module_info, train_expert_selector=False)
        if config['train']['continual_method'] != 'our' and mode == 'test':
            total_splits = [f'p{sid+1}' for sid in range(loss_sim.curr_bound)]
        if config['train']['continual_method'] == 'our_sim_pro' and mode == 'valid':
            total_splits = [f'p{loss_sim.curr_bound}']
        task_accuracies, tag_to_split = {split: [0, 0] for split in total_splits}, GLOBAL['continual_tag_to_split']
        if to_use_accu and 'split_accuracies' in accu_results:
            task_accuracies = accu_results['split_accuracies']
        for pred, label in zip(all_preds, all_tags):
            loc_split = tag_to_split[label]
            task_accuracies[loc_split][0] += int(pred == label)
            task_accuracies[loc_split][1] += 1
        ave_accuracies = [(task_accuracies[split][0] / task_accuracies[split][1]) for split in total_splits]
        results['total_accuracy'] = results['accuracy']
        results['split_accuracies'] = task_accuracies
        results['accuracy'] = round(sum(ave_accuracies) * 100 / len(ave_accuracies), 2)

    if config['train']['train_expert_selector']:
        if to_use_accu and 'correct_num_expert' in accu_results:
            correct_num_exp += accu_results['correct_num_expert']
            instance_num_exp += accu_results['instance_num_expert']
        results.update({
            "correct_num_expert": correct_num_exp,
            "instance_num_expert": instance_num_exp,
            "accuracy_expert": round(correct_num_exp * 100 / instance_num_exp, 2),
        })

        expert_splits, _ = loss_sim.get_current_splits(extra_module_info, train_expert_selector=True)
        task_accuracies = {split: [0, 0] for split in expert_splits}
        if to_use_accu and 'split_accuracies_expert' in accu_results:
            task_accuracies = accu_results['split_accuracies_expert']
        for pred, label in zip(all_preds_exp, all_tags_exp):
            loc_split = f'p{label + 1}'
            task_accuracies[loc_split][0] += int(pred == label)
            task_accuracies[loc_split][1] += 1
        ave_accuracies = [(task_accuracies[split][0] / task_accuracies[split][1]) for split in expert_splits]
        results['total_accuracy_expert'] = results['accuracy_expert']
        results['split_accuracies_expert'] = task_accuracies
        results['accuracy_expert'] = round(sum(ave_accuracies) * 100 / len(ave_accuracies), 2)

    if train_expert_selector:
        assert len(all_preds) == len(all_tags) == 0
        if 'accuracy' in results:
            results['accuracy_bak'] = results['accuracy']
        results['accuracy'] = results['accuracy_expert']
        all_preds, all_tags = all_preds_exp, all_tags_exp

    if output_path is not None:
        save_test_results(mode, output_path, is_step, all_preds, all_tags, results, epoch=epoch, step=step,
                          train_expert_selector=train_expert_selector, is_proto=prototypes is not None)

    return results, all_preds, all_tags


def model_list_test(config, test_loader, model, extra_module_states, tokenizer, loss_sim: LossSimilarity,
                    output_path, type_embeds, type_counter):
    assert config['task'] == 'continual'
    assert len(extra_module_states) == 3
    model.eval()

    embed_size, label_num, vocab_size, hidden_size = \
        config['embed_size'], config['label_num'], config['vocab_size'], config['hidden_size']
    split_to_tags = GLOBAL['continual_split_to_tags']

    # check tag set
    cur_tag_set = set()
    for idx in range(0, 3):
        cur_tags = set(split_to_tags[f'p{idx+1}'])
        assert len(cur_tags & cur_tag_set) == 0
        cur_tag_set |= cur_tags
    assert len(cur_tag_set) == label_num == len(type_counter)
    assert all(cnt > 0 for cnt in type_counter)

    all_preds, all_tags = [], []
    correct_num, instance_num = 0, 0
    data_iterator = tqdm(test_loader, desc='test') if len(test_loader) > 20 else test_loader

    for bid, batch in enumerate(data_iterator):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(config['device'], non_blocking=True)

        preds, tags = [], batch['tags'].cpu().tolist()
        batch_sz = batch['input_ids'].shape[0]
        final_losses = torch.zeros(batch_sz, label_num).to(type_embeds)
        for idx in range(0, 3):
            cur_tags = split_to_tags[f'p{idx+1}']
            cur_embeds = torch.stack([type_embeds[tag] for tag in cur_tags])
            model.load_state_dict(extra_module_states[idx], strict=False)
            logits = loss_sim.generate_inference_logits(model, tokenizer, batch)
            for sid, logit in enumerate(logits):
                logit = logit.unsqueeze(0).expand(len(cur_tags), -1)
                assert logit.shape == cur_embeds.shape
                target_ones = torch.ones(len(cur_tags)).to(type_embeds)
                cur_losses = loss_sim.forward_similarity_test(logit, cur_embeds, target_ones)
                assert torch.sum(final_losses[sid, cur_tags] == 0.) == len(cur_tags)
                final_losses[sid, cur_tags] = cur_losses
        for losses in final_losses:
            cur_probs = []
            for tag in range(label_num):
                cur_probs.append((tag, losses[tag].item()))
            pred = min(cur_probs, key=lambda x: x[1])[0]
            preds.append(pred)

        assert len(preds) == len(tags)
        for pred, label in zip(preds, tags):
            correct_num += int(pred == label)
            instance_num += 1

        all_preds += preds
        all_tags += tags

    results = {
        "correct_num": correct_num,
        "instance_num": instance_num,
        "accuracy": round(correct_num * 100 / instance_num, 2),
    }

    if output_path is not None:
        save_test_results('test', output_path, False, all_preds, all_tags, results)

    return results, all_preds, all_tags


def valid_save(config, model, tokenizer, loss_sim: LossSimilarity, is_step: bool,
               best_results: dict, cur_step, cur_epoch, optimizer, scheduler,
               valid_loader, valid_infer=None, train_infer=None, extra_module_info=None,
               train_expert_selector=False, prototypes=None):
    exp_path = os.path.join(config['logging']['path_base'], config['logging']['unique_string'])
    valid_output_path = os.path.join(exp_path, 'valid')
    os.makedirs(valid_output_path, exist_ok=True)
    # if config['task'] == 'fewshot':
    #     valid_output_path = None
    model_path = os.path.join(exp_path, 'models')
    if config['task'] in ['contrastive', 'continual']:
        type_embeds, type_counter = get_type_embeddings(config, train_infer, model, tokenizer, loss_sim)
    else:
        type_embeds, type_counter = None, None
    cur_num = cur_step if is_step else cur_epoch
    best_step, best_step_acc, best_epoch, best_epoch_acc = best_results['best_step'], best_results['best_step_acc'], \
        best_results['best_epoch'], best_results['best_epoch_acc']
    best_num, best_acc = (best_step, best_step_acc) if is_step else (best_epoch, best_epoch_acc)
    if valid_loader is not None:
        if isinstance(valid_loader, list):
            results, all_preds, all_tags = contrastive_group_test(
                config, 'valid', valid_loader, model, tokenizer, loss_sim,
                output_path=valid_output_path, is_step=is_step, step=cur_step, epoch=cur_epoch,
                type_embeds=type_embeds, type_counter=type_counter,
            )
        else:
            # accu_results=best_results for accumulation test
            results = test(config, valid_loader, valid_infer, model, tokenizer, loss_sim, 'valid',
                           valid_output_path, is_step=is_step, step=cur_step, epoch=cur_epoch,
                           type_embeds=type_embeds, type_counter=type_counter, tag_set=None,
                           extra_module_info=extra_module_info, accu_results=None,
                           train_expert_selector=train_expert_selector, prototypes=prototypes)[0]
        if results['accuracy'] > best_acc or best_num == -1:
            best_acc = results['accuracy']
            best_num = cur_num
            best_results.update(results)
            best_key = f'{"step" if is_step else "epoch"}-model'
            best_results[best_key] = {key: val.cpu() for key, val in model.state_dict().items()}
            if is_step:
                best_results['best_step'], best_results['best_step_acc'] = best_num, best_acc
            else:
                best_results['best_epoch'], best_results['best_epoch_acc'] = best_num, best_acc

    if config['train']['save_option'] == 2 or config['train']['save_option'] == 1 and best_num == cur_num:
        prefix = f'{"step" if is_step else "epoch"}-{cur_num if config["train"]["save_option"] == 2 else "best"}'
        if train_expert_selector:
            prefix = 'exp-' + prefix
        save_model_base(model_path, prefix, config['train']['save_pretrained'],
                        model, loss_sim, optimizer, scheduler, cur_epoch, cur_step,
                        best_results=best_results, type_embeds=type_embeds, type_counter=type_counter)
    if config['train']['save_option'] == 1:
        prefix = 'exp-last' if train_expert_selector else 'last'
        save_model_base(model_path, prefix, config['train']['save_pretrained'],
                        model, loss_sim, optimizer, scheduler, cur_epoch, cur_step,
                        best_results=best_results, type_embeds=type_embeds, type_counter=type_counter)


def save_model_base(path_base: str, prefix: str, save_pretrained: bool,
                    model, loss_sim: LossSimilarity, optimizer, scheduler, epoch, global_step,
                    best_results=None, type_embeds=None, type_counter=None):
    # Save model checkpoint
    model_to_save = model.module if hasattr(model, "module") else model
    assert not ((type_embeds is not None) ^ (type_counter is not None))
    extra_dict = {}
    if best_results is not None:
        result_dict = {}
        for key, val in best_results.items():
            if not key.endswith('model'):
                result_dict[key] = val
        extra_dict['best_results'] = result_dict
    if type_embeds is not None:
        extra_dict["type_embeds"] = type_embeds
        extra_dict["type_counter"] = type_counter
    if loss_sim.extra_module is not None:
        extra_dict["loss_sim"] = loss_sim.extra_module.state_dict()
    if save_pretrained:
        cp_output_dir = os.path.join(path_base, prefix)
        model_to_save.save_pretrained(cp_output_dir)
        meta_output_dir = os.path.join(cp_output_dir, 'meta.pkl')
        save_model(meta_output_dir, None, 'AdamW', optimizer, scheduler, epoch, global_step, extra=extra_dict)
    else:
        cp_output_dir = os.path.join(path_base, prefix + '.pkl')
        save_model(cp_output_dir, model_to_save, 'AdamW', optimizer, scheduler, epoch, global_step, extra=extra_dict)


@torch.no_grad()
def get_type_embeddings(config, data_loader, model, tokenizer, loss_sim: LossSimilarity,
                        exist_embeds=None, exist_counter=None):
    model.eval()

    embed_size, label_num, vocab_size, hidden_size = \
        config['embed_size'], config['label_num'], config['vocab_size'], config['hidden_size']
    reduction_batch_size = config['train']['reduction_batch_size']
    if exist_embeds is None:
        assert exist_counter is None
        res_embeds = [torch.zeros(embed_size).to(config['device']) for _ in range(label_num)]
        res_counter = [0 for _ in range(label_num)]
    else:
        assert exist_counter is not None
        res_embeds = [embed.clone() for embed in exist_embeds]
        res_counter = exist_counter.copy()
    exist_sample_num, accu_sample_size = sum(res_counter), 0
    accu_embeds = [torch.zeros(embed_size).to(config['device']) for _ in range(label_num)]
    accu_counter = [0 for _ in range(label_num)]
    embed_iterator = tqdm(data_loader, desc="embedding iteration") if len(data_loader) > 20 else data_loader

    def batch_update_type_embedding():
        for typ in range(label_num):
            old_num, new_num = res_counter[typ], accu_counter[typ]
            if new_num == 0:
                continue
            res_embeds[typ] = res_embeds[typ] * (old_num / (old_num + new_num)) + \
                accu_embeds[typ] / (old_num + new_num)
            res_counter[typ] = old_num + new_num
            accu_embeds[typ].zero_()
            accu_counter[typ] = 0

    for step, batch in enumerate(embed_iterator):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(config['device'], non_blocking=True)

        tags = batch['tags']
        logits = loss_sim.generate_inference_logits(model, tokenizer, batch)
        batch_sz, _ = logits.shape
        try:
            assert len(logits) == len(tags) == batch_sz
        except AssertionError as err:
            from IPython import embed
            embed()
            raise err
        for logit, tag in zip(logits, tags):
            accu_embeds[tag] += logit
            accu_counter[tag] += 1
        accu_sample_size += batch_sz
        if accu_sample_size % reduction_batch_size == 0:
            # update res_embeds
            batch_update_type_embedding()
    batch_update_type_embedding()
    assert sum(res_counter) - exist_sample_num == accu_sample_size
    return res_embeds, res_counter


def contrastive_group_test(config, mode, groups, model, tokenizer, loss_sim: LossSimilarity, output_path: str = None,
                           is_step=False, epoch=-1, step=-1, type_embeds=None, type_counter=None):
    correct_num, instance_num, all_preds, all_tags = 0, 0, [], []
    for item in tqdm(groups, desc=mode):
        supports, queries = item['support'], item['query']
        few_loader = get_contrastive_loader_by_dataset(config, mode, queries, tokenizer)
        few_infer = get_contrastive_loader_by_dataset(config, mode, supports, tokenizer)
        cur_results, cur_preds, cur_tags = test(
            config, few_loader, few_infer, model, tokenizer, loss_sim, mode,
            output_path=None, is_step=is_step, step=step, epoch=epoch,
            type_embeds=type_embeds, type_counter=type_counter, tag_set=None)
        correct_num += cur_results["correct_num"]
        instance_num += cur_results["instance_num"]
        all_preds += cur_preds
        all_tags += cur_tags
    results = {
        "correct_num": correct_num, "instance_num": instance_num,
        "accuracy": round(correct_num * 100 / instance_num, 2),
    }
    if output_path is not None:
        save_test_results(mode, output_path, is_step, all_preds, all_tags, results, epoch=epoch, step=step)
    return results, all_preds, all_tags


def test_by_best(config, model, tokenizer, loss_sim: LossSimilarity,
                 best_epoch, best_epoch_acc, best_step, best_step_acc, test_loader,
                 test_infer=None, train_infer=None, best_model=None, extra_module_info=None,
                 train_expert_selector=False, prototypes=None):
    exp_path = os.path.join(config['logging']['path_base'], config['logging']['unique_string'])
    model_path = os.path.join(exp_path, 'models')
    test_output_path = os.path.join(exp_path, 'test')
    os.makedirs(test_output_path, exist_ok=True)
    # if config['task'] == 'fewshot':
    #     test_output_path = None
    if config['train']['save_option'] <= 1:
        prefix = f'{"epoch" if best_epoch_acc > best_step_acc else "step"}-best'
        p_epoch, p_step = best_epoch, best_step
    elif best_epoch_acc > best_step_acc:
        prefix = f'epoch-{best_epoch}'
        p_epoch, p_step = best_epoch, -1
    else:
        prefix = f'step-{best_step}'
        p_epoch, p_step = -1, best_step
    if best_model is not None:
        strict = not (config['plm']['apply_lora'] or config['plm']['apply_adapter'])
        model.load_state_dict(best_model, strict=strict)
    else:
        if config['train']['save_pretrained']:
            cp_output_dir = os.path.join(model_path, prefix)
            use_new_model = config['plm']['apply_lora'] or config['plm']['apply_adapter']
            target_class = T5ForConditionalGeneration if config['plm']['model_name'] == 't5' else \
                (BertForMaskedLMLoRA if use_new_model else AutoModelForMaskedLM)
            model.load_state_dict(target_class.from_pretrained(cp_output_dir).state_dict())
        else:
            cp_output_dir = os.path.join(model_path, prefix + '.pkl')
            strict = not (config['plm']['apply_lora'] or config['plm']['apply_adapter'])
            model.load_state_dict(torch.load(cp_output_dir, map_location='cpu')['model'], strict=strict)
    if config['task'] in ['contrastive', 'continual']:
        assert train_infer is not None
        type_embeds, type_counter = get_type_embeddings(config, train_infer, model, tokenizer, loss_sim)
    else:
        type_embeds, type_counter = None, None
    if isinstance(test_loader, list):
        results, all_preds, all_tags = contrastive_group_test(
            config, 'test', test_loader, model, tokenizer, loss_sim,
            output_path=test_output_path, is_step='step' in prefix, epoch=p_epoch, step=p_step,
            type_embeds=type_embeds, type_counter=type_counter,
        )
    else:
        assert isinstance(test_loader, DataLoader)
        results = test(config, test_loader, test_infer, model, tokenizer, loss_sim, 'test', test_output_path,
                       is_step='step' in prefix, epoch=p_epoch, step=p_step,
                       type_embeds=type_embeds, type_counter=type_counter,
                       tag_set=None, extra_module_info=extra_module_info,
                       train_expert_selector=train_expert_selector, prototypes=prototypes)[0]
    return results


@torch.no_grad()
def select_continual_samples(config, data_loaders, model, tokenizer, loss_sim: LossSimilarity):
    assert config['task'] == 'fewshot' and config['train']['continual_method'] in ['emr', 'our_abl']
    model.eval()
    train_loader = data_loaders['train_infer']
    epoch_iterator = tqdm(train_loader, desc="data iteration") if len(train_loader) > 20 else train_loader
    for step, batch in enumerate(epoch_iterator):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(config['device'], non_blocking=True)
        loss_sim.forward_select_continual_sample(model, tokenizer, batch)
    selected_sample_keys = loss_sim.obtain_select_continual_sample()
    selected_samples = {'train_infer': [], 'valid_groups': []}
    cycle_suffix = config['logging']['cycle_suffix']
    is_emar = 'emar' in cycle_suffix or 'eaemr' in cycle_suffix
    for key, sample in train_loader.dataset:
        if key in selected_sample_keys['train_infer']:
            selected_samples['train_infer'].append((key, sample) if is_emar else sample)
        elif key in selected_sample_keys['valid_groups']:
            selected_samples['valid_groups'].append((key, sample) if is_emar else sample)
    print(f'selected {len(selected_samples["train_infer"])} train_infer samples')
    print(f'selected {len(selected_samples["valid_groups"])} valid_groups samples')
    assert len(selected_samples['train_infer']) == len(selected_sample_keys['train_infer'])
    assert len(selected_samples['valid_groups']) == len(selected_sample_keys['valid_groups'])
    data_conf = config["dataset"]
    if cycle_suffix != '':
        cycle_suffix = '_' + cycle_suffix
    method_prefix_map = {'prompt': 'pt', 'marker': 'mk', 'linear': 'li'}
    exp_prefix = method_prefix_map[data_conf['method_type']]
    save_json(selected_samples,
              f'cache/{data_conf["dataset_name"]}_continual_{data_conf["total_parts"]}_selected_{exp_prefix}'
              f'_{data_conf["special_part"]}{cycle_suffix}.json')
    return selected_samples
