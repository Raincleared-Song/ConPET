import math
import copy
import torch
import warnings
from global_var import GLOBAL
from torch.utils.data import DataLoader
from utils import load_json
from .custom_loader import CustomLoader


def init_type_descriptions(config):
    dataset_name = config['dataset']['dataset_name']
    continual_split_to_tags = load_json(f'scripts/{dataset_name}_class_split_'
                                        f'{config["dataset"]["total_parts"]}_tags.json')
    label_num = sum(len(tags) for split, tags in continual_split_to_tags.items() if split != 'all')
    continual_tag_to_split = ['' for _ in range(label_num)]
    for split, tags in continual_split_to_tags.items():
        if split == 'all':
            continue
        for tag in tags:
            continual_tag_to_split[tag] = split
    assert all(split != '' for split in continual_tag_to_split)
    GLOBAL["continual_split_to_tags"] = continual_split_to_tags
    GLOBAL["continual_tag_to_split"] = continual_tag_to_split


def init_contrastive_dataset(config, type_splits=None):
    assert config['dataset']['type'] == 'continual'
    special_part = config['dataset']['special_part']
    use_mask = config['dataset']['use_mask']
    dataset_name = config['dataset']['dataset_name']
    total_parts = config['dataset']['total_parts']
    if config['generate_logit']:
        current_split = int(config['dataset']['special_part'][1:])
        assert 1 <= current_split < 10
        special_part = f'p{current_split + 1}'
    ori_dataset = {}
    for part in ['train', 'valid', 'test']:
        ori_dataset[part] = load_json(f'data/{dataset_name}/{dataset_name}_split_{part}_dataset_{total_parts}'
                                      f'{"_mask" if use_mask else ""}.json')
    cur_sid = int(special_part[1:]) - 1
    dataset = {
        'train': copy.deepcopy(ori_dataset['train'][cur_sid]),
        'train_infer': copy.deepcopy(ori_dataset['train'][cur_sid]),
        'valid_groups': copy.deepcopy(ori_dataset['valid'][cur_sid]),
        'test_groups': copy.deepcopy(ori_dataset['test'][cur_sid]),
    }
    label_num = len(GLOBAL['continual_tag_to_split'])
    use_selected = config['dataset']['use_selected'] if 'use_selected' in config['dataset'] else False
    use_selected &= not config['is_test']
    assert config['task'] == 'fewshot' and not use_selected
    if config['task'] == 'fewshot':
        assert type_splits is not None
        extra_special_part = [s for s in config['dataset']['extra_special_part'].split(',') if s]
        total_splits = [split for split in type_splits]
        total_splits = sorted(total_splits)
        for split in extra_special_part:
            if split not in total_splits:
                total_splits.append(split)
        for split in total_splits:
            sid = int(split[1:]) - 1
            extra_dataset = {
                'train': ori_dataset['train'][sid],
                'train_infer': ori_dataset['train'][sid],
                'valid_groups': ori_dataset['valid'][sid],
                'test_groups': ori_dataset['test'][sid],
            }
            for key in dataset.keys():
                if use_selected and key in ['train_infer', 'valid_groups']:
                    continue
                dataset[key] += extra_dataset[key]
        if config['is_test'] or config['train']['continual_method'] != 'our':
            new_test_set = []
            for sid in range(int(config['dataset']['special_part'][1:])):
                new_test_set += ori_dataset['test'][sid]
            dataset['test_groups'] = new_test_set
        all_in_splits = total_splits + [special_part]
        print(f'got test set for splits {all_in_splits} size {len(dataset["test_groups"])}')
    for key, val in dataset.items():
        dataset[key] = [(f'{key}_{idx}', sample) for idx, sample in enumerate(val)]
    return dataset, label_num


def get_sim_mat_by_config(config, tags, loss_sim):
    sim_mat, batch_sz = [], len(tags)
    negative_label = loss_sim.negative_label
    if config['dataset']['in_batch']:
        for i in range(batch_sz):
            for j in range(batch_sz):
                sim_mat.append(1 if tags[i] == tags[j] else negative_label)
    else:
        for i in range(0, batch_sz, 2):
            assert tags[i] == tags[i+1]
            sim_mat.append(1 if tags[i] >= 0 else negative_label)
    return sim_mat


def get_tag_set_by_dataset(dataset) -> set:
    if isinstance(dataset, DataLoader) or isinstance(dataset, CustomLoader):
        dataset = dataset.dataset
    tag_set = set()
    for sample in dataset:
        if isinstance(sample[0], str):
            sample = sample[1]
        tag_set.add(sample[0])
    return tag_set


def data_collate_fn(config, tokenizer):
    method_type = config['dataset']['method_type']
    max_seq_len = config['dataloader']['max_seq_length']
    entity_blank_ratio = config['dataset']['entity_blank_ratio']
    use_mask = config['dataset']['use_mask']

    def truncate_sentence_with_entity(sent_limit: int, head_pos: int, tail_pos: int, sent_ids):
        assert 0 < tail_pos - head_pos < sent_limit
        remain_len = sent_limit - (tail_pos - head_pos + 1)
        if head_pos < remain_len / 2:
            ret_sent_ids = sent_ids[:sent_limit]
        elif len(sent_ids) - tail_pos - 1 < remain_len / 2:
            ret_sent_ids = sent_ids[-sent_limit:]
        else:
            head_pos -= math.ceil(remain_len / 2)
            tail_pos += math.floor(remain_len / 2) + 1
            ret_sent_ids = sent_ids[head_pos:tail_pos]
        assert len(ret_sent_ids) == sent_limit
        return ret_sent_ids

    def loc_collate_fn_linear_re(batch):
        warnings.filterwarnings(action='ignore')
        texts, tags = [], []
        sample_keys = [item[0] for item in batch]
        batch = [item[1] for item in batch]
        for sample in batch:
            tag, text = sample
            texts.append(text)
            tags.append(tag)

        head_start, head_end, tail_start, tail_end, blank_token = [f'[unused{idx}]' for idx in range(5)]
        head_before_tails = []
        for text in texts:
            pos = text.find(head_start)
            assert pos != -1, text
            pos = text.find(tail_end, pos)
            if pos == -1:
                pos = text.find(tail_start)
                assert pos != -1, text
                pos = text.find(head_end, pos)
                assert pos != -1, text
                head_before_tails.append(False)
            else:
                head_before_tails.append(True)

        # mask entities by some ratio
        if entity_blank_ratio > 0:
            raise NotImplementedError('entity_blank_ratio must be 0')

        text_inputs = tokenizer(texts, return_tensors='pt',
                                padding='longest', truncation='longest_first', max_length=max_seq_len)
        input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask
        assert len(texts) == len(input_ids) == len(attention_mask) == len(tags) == len(head_before_tails)
        mask_id = tokenizer.mask_token_id

        for text, input_id, head_before_tail in zip(texts, input_ids, head_before_tails):
            head_id = tokenizer.convert_tokens_to_ids(head_start if head_before_tail else tail_start)
            tail_id = tokenizer.convert_tokens_to_ids(tail_end if head_before_tail else head_end)
            if head_id not in input_id or tail_id not in input_id or use_mask and mask_id not in input_id:
                if use_mask:
                    pos = text.find(' In this sentence,')
                    assert pos != -1, text
                    sent_ids = tokenizer(text[:pos], return_tensors='pt').input_ids.squeeze(0)[1:-1]
                    extra_ids = tokenizer(text[pos+1:], return_tensors='pt').input_ids.squeeze(0)[1:-1]
                else:
                    sent_ids = tokenizer(text, return_tensors='pt').input_ids.squeeze(0)[1:-1]
                    extra_ids = torch.LongTensor([])
                extra_len, sent_limit = len(extra_ids), max_seq_len - 2 - len(extra_ids)

                assert len(sent_ids) + len(extra_ids) + 2 > max_seq_len and len(input_id) == max_seq_len, text
                head_pos = torch.nonzero(torch.eq(sent_ids, head_id), as_tuple=True)[0].item()
                tail_pos = torch.nonzero(torch.eq(sent_ids, tail_id), as_tuple=True)[0].item()
                assert 0 < tail_pos - head_pos, text
                if tail_pos - head_pos >= sent_limit:
                    head_limit, tail_limit = sent_limit // 2, sent_limit - sent_limit // 2
                    head_aux_id = tokenizer.convert_tokens_to_ids(head_end if head_before_tail else tail_end)
                    head_aux_pos = torch.nonzero(torch.eq(sent_ids, head_aux_id), as_tuple=True)[0].item()
                    tail_aux_id = tokenizer.convert_tokens_to_ids(tail_start if head_before_tail else head_start)
                    tail_aux_pos = torch.nonzero(torch.eq(sent_ids, tail_aux_id), as_tuple=True)[0].item()
                    head_ids = truncate_sentence_with_entity(head_limit, head_pos, head_aux_pos, sent_ids)
                    tail_ids = truncate_sentence_with_entity(tail_limit, tail_aux_pos, tail_pos, sent_ids)
                    truncated_sent_ids = torch.cat((head_ids, tail_ids), dim=0)
                else:
                    truncated_sent_ids = truncate_sentence_with_entity(sent_limit, head_pos, tail_pos, sent_ids)
                input_id[1:1 + sent_limit] = truncated_sent_ids
                input_id[-1 - extra_len:-1] = extra_ids
            assert head_id in input_id and tail_id in input_id, text
            assert not use_mask or mask_id in input_id, text
            assert input_id[0] == tokenizer.cls_token_id and tokenizer.sep_token_id in input_id, text
            assert torch.sum(torch.eq(input_id, tokenizer.cls_token_id)) == 1 and \
                   torch.sum(torch.eq(input_id, tokenizer.sep_token_id)) == 1
            assert not use_mask or torch.sum(torch.eq(input_id, tokenizer.mask_token_id)) == 1

        ret = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "tags": torch.LongTensor(tags),
            "sample_keys": sample_keys,
        }
        return ret

    def loc_collate_fn_linear_et(batch):
        warnings.filterwarnings(action='ignore')
        texts, tags = [], []
        sample_keys = [item[0] for item in batch]
        batch = [item[1] for item in batch]
        for sample in batch:
            tag, text = sample
            texts.append(text)
            tags.append(tag)

        head_marker, tail_marker, blank_token = '[unused0]', '[unused1]', '[unused2]'

        # mask entities by some ratio
        if entity_blank_ratio > 0:
            raise NotImplementedError('entity_blank_ratio must be 0')

        text_inputs = tokenizer(texts, return_tensors='pt',
                                padding='longest', truncation='longest_first', max_length=max_seq_len)
        input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask
        assert len(texts) == len(input_ids) == len(attention_mask) == len(tags)
        mask_id = tokenizer.mask_token_id

        head_id = tokenizer.convert_tokens_to_ids(head_marker)
        tail_id = tokenizer.convert_tokens_to_ids(tail_marker)

        for text, input_id in zip(texts, input_ids):
            if head_id not in input_id or tail_id not in input_id or use_mask and mask_id not in input_id:
                if use_mask:
                    pos = text.find(' In this sentence,')
                    assert pos != -1, text
                    sent_ids = tokenizer(text[:pos], return_tensors='pt').input_ids.squeeze(0)[1:-1]
                    extra_ids = tokenizer(text[pos+1:], return_tensors='pt').input_ids.squeeze(0)[1:-1]
                else:
                    sent_ids = tokenizer(text, return_tensors='pt').input_ids.squeeze(0)[1:-1]
                    extra_ids = torch.LongTensor([])
                extra_len, sent_limit = len(extra_ids), max_seq_len - 2 - len(extra_ids)

                assert len(sent_ids) + len(extra_ids) + 2 > max_seq_len and len(input_id) == max_seq_len, text
                head_pos = torch.nonzero(torch.eq(sent_ids, head_id), as_tuple=True)[0].item()
                tail_pos = torch.nonzero(torch.eq(sent_ids, tail_id), as_tuple=True)[0].item()
                assert 0 < tail_pos - head_pos, text
                input_id[1:1 + sent_limit] = truncate_sentence_with_entity(sent_limit, head_pos, tail_pos, sent_ids)
                input_id[-1 - extra_len:-1] = extra_ids
            assert head_id in input_id and tail_id in input_id, text
            assert not use_mask or mask_id in input_id, text
            assert input_id[0] == tokenizer.cls_token_id and tokenizer.sep_token_id in input_id, text
            assert torch.sum(torch.eq(input_id, tokenizer.cls_token_id)) == 1 and \
                   torch.sum(torch.eq(input_id, tokenizer.sep_token_id)) == 1
            assert not use_mask or torch.sum(torch.eq(input_id, tokenizer.mask_token_id)) == 1

        ret = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "tags": torch.LongTensor(tags),
            "sample_keys": sample_keys,
        }
        return ret

    assert method_type == 'linear'
    if method_type in ['prompt', 'linear']:
        dataset_name = config['dataset']['dataset_name']
        if dataset_name in ['fewrel', 'tacred', 'ace']:
            return loc_collate_fn_linear_re
        if dataset_name in ['fewnerd', 'ontonotes', 'bbn']:
            return loc_collate_fn_linear_et
        raise NotImplementedError('invalid dataset_name: ' + dataset_name)
    raise NotImplementedError('invalid method_type: ' + method_type)


def get_contrastive_loader_by_dataset(config, part, dataset, tokenizer):
    mode = part.split('_')[0]
    if part == 'train_infer' and config['task'] != 'fewshot':
        mode = 'valid'
    cur_config = config[mode]
    if config['dataset']['batch_limit_policy'] > 0 and mode != 'test' and config['dataset']['special_part'] != 'p1':
        loader = CustomLoader(
            config=config,
            mode=mode,
            dataset=dataset,
            batch_size=cur_config['batch_size'],
            shuffle=cur_config['shuffle_data'],
            num_workers=4,
            collate_fn=data_collate_fn(config, tokenizer),
            drop_last=False,
        )
    else:
        loader = DataLoader(
            dataset=dataset,
            batch_size=cur_config['batch_size'],
            shuffle=cur_config['shuffle_data'],
            num_workers=4,
            collate_fn=data_collate_fn(config, tokenizer),
            drop_last=False,
        )
    return loader


def init_contrastive_dataloader(config, datasets, tokenizer):
    data_loaders = {}
    for part, dataset in datasets.items():
        if isinstance(dataset, list) and (len(dataset) == 0 or isinstance(dataset[0], dict)):
            cur_loader = dataset
        else:
            cur_loader = get_contrastive_loader_by_dataset(config, part, dataset, tokenizer)
        data_loaders[part] = cur_loader
    train_sz = len(data_loaders['train']) if 'train' in data_loaders else len(data_loaders['train_infer'])
    return data_loaders, train_sz
