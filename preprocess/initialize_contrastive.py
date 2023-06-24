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
    assert config['dataset']['use_mask']
    dataset_name = config['dataset']['dataset_name']
    total_parts = config['dataset']['total_parts']
    method_type = config['dataset']['method_type']
    method_prefix_map = {'prompt': 'pt', 'marker': 'mk', 'linear': 'li'}
    exp_prefix = method_prefix_map[method_type]
    if config['generate_logit']:
        current_split = int(config['dataset']['special_part'][1:])
        assert 1 <= current_split < 10
        special_part = f'p{current_split + 1}'
    ori_dataset = {}
    for part in ['train', 'valid', 'test']:
        ori_dataset[part] = load_json(f'data/{dataset_name}/{dataset_name}_split_{part}_dataset_{total_parts}_key.json')
    # filter entities too long
    for ds_id in range(len(ori_dataset['train'])):
        new_dataset = []
        for sample_key, tag, text in ori_dataset['train'][ds_id]:
            pos = text.find(' In this sentence,')
            if len(text[pos + 1:].split(' ')) > 50:
                continue
            new_dataset.append((sample_key, tag, text))
        ori_dataset['train'][ds_id] = new_dataset

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
    assert config['task'] == 'fewshot'
    extra_special_part = [s for s in config['dataset']['extra_special_part'].split(',') if s]
    total_splits = [split for split in type_splits]
    total_splits = sorted(total_splits)
    for split in extra_special_part:
        if split not in total_splits:
            total_splits.append(split)
    # ALERT: if select_sample, do not add more samples
    if config['select_sample'] or config['train']['continual_method'] == 'our_sim_pro':
        total_splits = []
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
        if use_selected:
            assert config['train']['continual_method'] in ['emr', 'our_abl']
            cycle_suffix = config['logging']['cycle_suffix']
            if cycle_suffix != '':
                cycle_suffix = '_' + cycle_suffix
            selected_cache_path = f'cache/{dataset_name}_continual_{total_parts}_selected_{exp_prefix}' \
                                  f'_{split}{cycle_suffix}.json'
            selected_samples = load_json(selected_cache_path)
            for _ in range(config['dataset']['replay_frequency']):
                dataset['train_infer'] += selected_samples['train_infer']
            dataset['valid_groups'] += selected_samples['valid_groups']
    if config['is_test'] or config['train']['continual_method'] not in ['our', 'our_abl']:
        new_test_set = []
        for sid in range(int(config['dataset']['special_part'][1:])):
            new_test_set += ori_dataset['test'][sid]
        dataset['test_groups'] = new_test_set
    all_in_splits = total_splits + [special_part]
    print(f'got test set for splits {all_in_splits} size {len(dataset["test_groups"])}')
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
        tag = sample[1] if len(sample) == 3 else sample[0]
        assert isinstance(tag, int)
        tag_set.add(tag)
    return tag_set


def data_collate_fn(config, tokenizer):
    method_type = config['dataset']['method_type']
    max_seq_len = config['dataloader']['max_seq_length']
    entity_blank_ratio = config['dataset']['entity_blank_ratio']
    use_mask = config['dataset']['use_mask']
    model_path = config['plm']['model_path']
    model_name = config['plm']['model_name']

    def convert_text_batch_to_ids(texts: list, p_tokenizer):
        if model_name == 'cpm':
            encode_results, lengths = [], []
            for text in texts:
                if not text.startswith(p_tokenizer.bos_token):
                    text = p_tokenizer.bos_token + text
                if not text.endswith(p_tokenizer.eos_token):
                    text = text + p_tokenizer.eos_token
                encode_res = p_tokenizer.encode(text)[0][:max_seq_len]
                encode_res[0] = p_tokenizer.bos_id
                encode_res[-1] = p_tokenizer.eos_id
                encode_results.append(encode_res)
                lengths.append(len(encode_res))
            pad_len = max(len(entry) for entry in encode_results)
            for entry in encode_results:
                entry.extend([p_tokenizer.pad_id] * (pad_len - len(entry)))
            return torch.LongTensor(encode_results), torch.LongTensor(lengths)
        elif model_name == 'llama':
            encode_results, attention_mask = [], []
            for text in texts:
                encode_res = p_tokenizer.encode(text)[:max_seq_len]
                assert encode_res[0] == p_tokenizer.bos_token_id
                assert encode_res[-1] != p_tokenizer.eos_token_id
                if len(encode_res) == max_seq_len:
                    encode_res[-1] = p_tokenizer.eos_token_id
                else:
                    encode_res.append(p_tokenizer.eos_token_id)
                encode_results.append(encode_res)
            pad_max = max(len(res) for res in encode_results)
            for encode_res in encode_results:
                pad_len = pad_max - len(encode_res)
                attention_mask.append([1] * len(encode_res) + [0] * pad_len)
                encode_res.extend([p_tokenizer.pad_token_id] * pad_len)
            return torch.LongTensor(encode_results), torch.LongTensor(attention_mask)
        else:
            encode_res = p_tokenizer(texts, return_tensors='pt',
                                     padding='longest', truncation='longest_first', max_length=max_seq_len)
            return encode_res.input_ids, encode_res.attention_mask

    def convert_text_single_to_ids(text: str, p_tokenizer):
        if model_name == 'cpm':
            if not text.startswith(p_tokenizer.bos_token):
                text = p_tokenizer.bos_token + text
            if not text.endswith(p_tokenizer.eos_token):
                text = text + p_tokenizer.eos_token
            return torch.LongTensor(p_tokenizer.encode(text)[0])
        elif model_name == 'llama':
            encode_res = p_tokenizer.encode(text)
            assert encode_res[0] == p_tokenizer.bos_token_id
            assert encode_res[-1] != p_tokenizer.eos_token_id
            encode_res.append(p_tokenizer.eos_token_id)
            return torch.LongTensor(encode_res)
        else:
            return p_tokenizer(text, return_tensors='pt').input_ids.squeeze(0)

    def convert_token_to_id(token: str, p_tokenizer):
        if model_name != 'cpm':
            return p_tokenizer.convert_tokens_to_ids(token)
        else:
            return p_tokenizer.encoder[token]

    def batch_adaption(batch: dict):
        if model_name == 'llama':
            return {
                "input_ids": batch["input_ids"].to(dtype=torch.int32),
                "attention_mask": batch["attention_mask"].to(dtype=torch.int32),
                "tags": batch["tags"].to(dtype=torch.int32),
                "sample_keys": batch["sample_keys"],
            }
        elif model_name == 'cpm':
            input_ids, lengths = batch['input_ids'].tolist(), batch['attention_mask'].tolist()
            bmt_prefix = '<s><root></s><s>input</s>'
            bmt_prefix = tokenizer.encode(bmt_prefix)[0]
            assert len(bmt_prefix) == 6
            new_input_ids, new_lengths, contexts = [], [], []
            num_segments, segment_ids, segment_rels = [], [], []
            seq_len = len(input_ids[0])
            for idx in range(len(lengths)):
                new_input_ids.append(bmt_prefix + input_ids[idx])
                new_lengths.append(6 + lengths[idx])
                contexts.append([True] * (6 + lengths[idx]) + [False] * (seq_len - lengths[idx]))
                num_segments.append([3] * (6 + lengths[idx]) + [0] * (seq_len - lengths[idx]))
                segment_ids.append([0] * 3 + [1] * 3 + [2] * lengths[idx] + [0] * (seq_len - lengths[idx]))
                segment_rels.append([0, 2, 3, 9, 0, 2, 17, 9, 0])
            new_input_ids = torch.IntTensor(new_input_ids)
            new_batch = {
                'input_ids': new_input_ids,
                'input_sub': torch.zeros(new_input_ids.shape, dtype=torch.int32),
                'length': torch.IntTensor(new_lengths),
                'context': torch.BoolTensor(contexts),
                'sample_ids': torch.zeros(new_input_ids.shape, dtype=torch.int32),
                'num_segments': torch.IntTensor(num_segments),
                'segment': torch.IntTensor(segment_ids),
                'segment_rel_offset': torch.zeros(new_input_ids.shape, dtype=torch.int32),
                'segment_rel': torch.IntTensor(segment_rels),
                'span': torch.zeros(new_input_ids.shape, dtype=torch.int32),
                'tags': batch['tags'].int(),
                'sample_keys': batch['sample_keys'],
            }
            return new_batch
        else:
            return batch

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
        texts, tags, sample_keys = [], [], []
        for sample_key, tag, text in batch:
            texts.append(text)
            tags.append(tag)
            sample_keys.append(sample_key)

        unused_range = range(1, 6) if model_path.startswith('hfl/') else range(5)
        if model_name == 'cpm':
            new_texts = []
            for text in texts:
                text = text.replace('<', '<<')
                text = text.replace('[MASK]', '<mask>')
                for idx in unused_range:
                    text = text.replace(f'[unused{idx}] ', f'[unused{idx}]')
                    text = text.replace(f' [unused{idx}]', f'[unused{idx}]')
                    text = text.replace(f'[unused{idx}]', f'<unused{idx}>')
                new_texts.append(tokenizer.bos_token + text + tokenizer.eos_token)
            texts = new_texts
            head_start, head_end, tail_start, tail_end, blank_token = [f'<unused{idx}>' for idx in unused_range]
            mask_id, cls_id, sep_id = tokenizer.mask_id, tokenizer.bos_id, tokenizer.eos_id
        elif model_name == 'llama':
            mask_token = tokenizer.mask_token
            new_texts = []
            for text in texts:
                text = text.replace('[MASK]', mask_token)
                for idx in unused_range:
                    text = text.replace(f'[unused{idx}]', f'<0x{idx:02}>')
                new_texts.append(text)
            texts = new_texts
            head_start, head_end, tail_start, tail_end, blank_token = [f'<0x{idx:02}>' for idx in unused_range]
            mask_id, cls_id, sep_id = tokenizer.mask_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id
        else:
            head_start, head_end, tail_start, tail_end, blank_token = [f'[unused{idx}]' for idx in unused_range]
            mask_id, cls_id, sep_id = tokenizer.mask_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id

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

        input_ids, attention_mask = convert_text_batch_to_ids(texts, tokenizer)
        assert len(texts) == len(input_ids) == len(attention_mask) == len(tags) == len(head_before_tails)

        for text, input_id, head_before_tail in zip(texts, input_ids, head_before_tails):
            head_id = convert_token_to_id(head_start if head_before_tail else tail_start, tokenizer)
            tail_id = convert_token_to_id(tail_end if head_before_tail else head_end, tokenizer)
            if head_id not in input_id or tail_id not in input_id or use_mask and mask_id not in input_id:
                if use_mask:
                    pos = text.find(' In this sentence,')
                    assert pos != -1, text
                    sent_ids = convert_text_single_to_ids(text[:pos], tokenizer)[1:-1]
                    extra_ids = convert_text_single_to_ids(text[pos+1:], tokenizer)[1:-1]
                else:
                    sent_ids = convert_text_single_to_ids(text, tokenizer)[1:-1]
                    extra_ids = torch.LongTensor([])
                extra_len, sent_limit = len(extra_ids), max_seq_len - 2 - len(extra_ids)

                assert len(sent_ids) + len(extra_ids) + 2 > max_seq_len and len(input_id) == max_seq_len, text
                head_pos = torch.nonzero(torch.eq(sent_ids, head_id), as_tuple=True)[0][0].item()
                tail_pos = torch.nonzero(torch.eq(sent_ids, tail_id), as_tuple=True)[0][0].item()
                assert 0 < tail_pos - head_pos, text
                if tail_pos - head_pos >= sent_limit:
                    head_limit, tail_limit = sent_limit // 2, sent_limit - sent_limit // 2
                    head_aux_id = convert_token_to_id(head_end if head_before_tail else tail_end, tokenizer)
                    head_aux_pos = torch.nonzero(torch.eq(sent_ids, head_aux_id), as_tuple=True)[0][0].item()
                    tail_aux_id = convert_token_to_id(tail_start if head_before_tail else head_start, tokenizer)
                    tail_aux_pos = torch.nonzero(torch.eq(sent_ids, tail_aux_id), as_tuple=True)[0][0].item()
                    head_ids = truncate_sentence_with_entity(head_limit, head_pos, head_aux_pos, sent_ids)
                    tail_ids = truncate_sentence_with_entity(tail_limit, tail_aux_pos, tail_pos, sent_ids)
                    truncated_sent_ids = torch.cat((head_ids, tail_ids), dim=0)
                else:
                    truncated_sent_ids = truncate_sentence_with_entity(sent_limit, head_pos, tail_pos, sent_ids)
                input_id[1:1 + sent_limit] = truncated_sent_ids
                input_id[-1 - extra_len:-1] = extra_ids
            assert head_id in input_id and tail_id in input_id, text
            assert not use_mask or mask_id in input_id, text
            assert input_id[0] == cls_id and sep_id in input_id, text
            assert torch.sum(torch.eq(input_id, cls_id)) == 1 and torch.sum(torch.eq(input_id, sep_id)) == 1
            assert not use_mask or torch.sum(torch.eq(input_id, mask_id)) == 1

        ret = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "tags": torch.LongTensor(tags),
            "sample_keys": sample_keys,
        }
        return batch_adaption(ret)

    def loc_collate_fn_linear_et(batch):
        warnings.filterwarnings(action='ignore')
        texts, tags, sample_keys = [], [], []
        for sample_key, tag, text in batch:
            texts.append(text)
            tags.append(tag)
            sample_keys.append(sample_key)

        unused_range = range(1, 4) if model_path.startswith('hfl/') else range(3)
        if model_name == 'cpm':
            new_texts = []
            for text in texts:
                text = text.replace('<', '<<')
                text = text.replace('[MASK]', '<mask>')
                for idx in unused_range:
                    text = text.replace(f'[unused{idx}] ', f'[unused{idx}]')
                    text = text.replace(f' [unused{idx}]', f'[unused{idx}]')
                    text = text.replace(f'[unused{idx}]', f'<unused{idx}>')
                new_texts.append(tokenizer.bos_token + text + tokenizer.eos_token)
            texts = new_texts
            head_marker, tail_marker, blank_token = [f'<unused{idx}>' for idx in unused_range]
            mask_id, cls_id, sep_id = tokenizer.mask_id, tokenizer.bos_id, tokenizer.eos_id
        elif model_name == 'llama':
            mask_token = tokenizer.mask_token
            new_texts = []
            for text in texts:
                text = text.replace('[MASK]', mask_token)
                for idx in unused_range:
                    text = text.replace(f'[unused{idx}]', f'<0x{idx:02}>')
                new_texts.append(text)
            texts = new_texts
            head_marker, tail_marker, blank_token = [f'<0x{idx:02}>' for idx in unused_range]
            mask_id, cls_id, sep_id = tokenizer.mask_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id
        else:
            head_marker, tail_marker, blank_token = [f'[unused{idx}]' for idx in unused_range]
            mask_id, cls_id, sep_id = tokenizer.mask_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id

        # mask entities by some ratio
        if entity_blank_ratio > 0:
            raise NotImplementedError('entity_blank_ratio must be 0')

        input_ids, attention_mask = convert_text_batch_to_ids(texts, tokenizer)
        assert len(texts) == len(input_ids) == len(attention_mask) == len(tags)

        head_id = convert_token_to_id(head_marker, tokenizer)
        tail_id = convert_token_to_id(tail_marker, tokenizer)

        for text, input_id in zip(texts, input_ids):
            if head_id not in input_id or tail_id not in input_id or use_mask and mask_id not in input_id:
                if use_mask:
                    pos = text.find(' In this sentence,')
                    assert pos != -1, text
                    sent_ids = convert_text_single_to_ids(text[:pos], tokenizer)[1:-1]
                    extra_ids = convert_text_single_to_ids(text[pos + 1:], tokenizer)[1:-1]
                else:
                    sent_ids = convert_text_single_to_ids(text, tokenizer)[1:-1]
                    extra_ids = torch.LongTensor([])
                extra_len, sent_limit = len(extra_ids), max_seq_len - 2 - len(extra_ids)

                assert len(sent_ids) + len(extra_ids) + 2 > max_seq_len and len(input_id) == max_seq_len, text
                head_pos = torch.nonzero(torch.eq(sent_ids, head_id), as_tuple=True)[0][0].item()
                tail_pos = torch.nonzero(torch.eq(sent_ids, tail_id), as_tuple=True)[0][0].item()
                assert 0 < tail_pos - head_pos < sent_limit, text
                input_id[1:1 + sent_limit] = truncate_sentence_with_entity(sent_limit, head_pos, tail_pos, sent_ids)
                input_id[-1 - extra_len:-1] = extra_ids
            assert head_id in input_id and tail_id in input_id, text
            assert not use_mask or mask_id in input_id, text
            assert input_id[0] == cls_id and sep_id in input_id, text
            assert torch.sum(torch.eq(input_id, cls_id)) == 1 and torch.sum(torch.eq(input_id, sep_id)) == 1
            assert not use_mask or torch.sum(torch.eq(input_id, mask_id)) == 1

        ret = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "tags": torch.LongTensor(tags),
            "sample_keys": sample_keys,
        }
        return batch_adaption(ret)

    assert method_type == 'linear'
    if method_type in ['prompt', 'linear']:
        dataset_name = config['dataset']['dataset_name']
        if dataset_name in ['fewrel', 'tacred', 'ace']:
            return loc_collate_fn_linear_re
        if dataset_name in ['fewnerd', 'ontonotes', 'bbn', 'chent']:
            return loc_collate_fn_linear_et
        raise NotImplementedError('invalid dataset_name: ' + dataset_name)
    raise NotImplementedError('invalid method_type: ' + method_type)


def get_contrastive_loader_by_dataset(config, part, dataset, tokenizer):
    mode = part.split('_')[0]
    if part == 'train_infer' and config['task'] != 'fewshot':
        mode = 'valid'
    cur_config = config[mode]
    if config['dataset']['batch_limit_policy'] > 0 and mode != 'test' and config['dataset']['special_part'] != 'p1':
        use_cache = config['train']['continual_method'] == 'emr'
        max_epoch_num = 1 if not use_cache else config['train']['num_epochs']
        loader = CustomLoader(
            config=config,
            mode=mode,
            dataset=dataset,
            batch_size=cur_config['batch_size'],
            shuffle=cur_config['shuffle_data'],
            num_workers=4,
            collate_fn=data_collate_fn(config, tokenizer),
            drop_last=False,
            max_epoch_num=max_epoch_num,
            use_cache=use_cache,
        )
        if not use_cache:
            assert loader.total_len == loader.overall_len
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
