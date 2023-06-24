import math
import torch
import warnings
from .data_utils import get_tokenizer


def custom_collate_fn(args):
    max_seq_len = args.max_length
    if 'llama' in args.bert_path:
        model_name = 'llama'
    elif 'bert' in args.bert_path:
        model_name = 'bert'
    else:
        raise NotImplementedError
    tokenizer = get_tokenizer(args)

    def convert_text_batch_to_ids(texts: list, p_tokenizer):
        if model_name == 'llama':
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
            assert model_name == 'bert'
            encode_res = p_tokenizer(texts, return_tensors='pt',
                                     padding='longest', truncation='longest_first', max_length=max_seq_len)
            return encode_res.input_ids, encode_res.attention_mask

    def convert_text_single_to_ids(text: str, p_tokenizer):
        if model_name == 'llama':
            encode_res = p_tokenizer.encode(text)
            assert encode_res[0] == p_tokenizer.bos_token_id
            assert encode_res[-1] != p_tokenizer.eos_token_id
            encode_res.append(p_tokenizer.eos_token_id)
            return torch.LongTensor(encode_res)
        else:
            return p_tokenizer(text, return_tensors='pt').input_ids.squeeze(0)

    def convert_token_to_id(token: str, p_tokenizer):
        return p_tokenizer.convert_tokens_to_ids(token)

    def batch_adaption(batch: dict):
        if model_name == 'llama':
            batch = {
                "input_ids": batch["input_ids"].to(dtype=torch.int32),
                "attention_mask": batch["attention_mask"].to(dtype=torch.int32),
                "tags": batch["tags"].to(dtype=torch.int32),
                "sample_keys": batch["sample_keys"].to(dtype=torch.int32),
                "mask_ids": batch["mask_ids"].to(dtype=torch.int32),
            }
        # labels, tokens, attention_mask, mask_ids, ind
        return batch['tags'], batch['input_ids'], batch['attention_mask'], batch['mask_ids'], batch['sample_keys']

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

        if model_name == 'llama':
            mask_token = tokenizer.mask_token
            new_texts = []
            for text in texts:
                text = text.replace('[MASK]', mask_token)
                for idx in range(5):
                    text = text.replace(f'[unused{idx}]', f'<0x{idx:02}>')
                new_texts.append(text)
            texts = new_texts
            head_start, head_end, tail_start, tail_end, blank_token = [f'<0x{idx:02}>' for idx in range(5)]
            mask_id, cls_id, sep_id = tokenizer.mask_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id
        else:
            head_start, head_end, tail_start, tail_end, blank_token = [f'[unused{idx}]' for idx in range(5)]
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

        input_ids, attention_mask = convert_text_batch_to_ids(texts, tokenizer)
        assert len(texts) == len(input_ids) == len(attention_mask) == len(tags) == len(head_before_tails)

        for text, input_id, head_before_tail in zip(texts, input_ids, head_before_tails):
            head_id = convert_token_to_id(head_start if head_before_tail else tail_start, tokenizer)
            tail_id = convert_token_to_id(tail_end if head_before_tail else head_end, tokenizer)
            if head_id not in input_id or tail_id not in input_id or mask_id not in input_id:
                pos = text.find(' In this sentence,')
                assert pos != -1, text
                sent_ids = convert_text_single_to_ids(text[:pos], tokenizer)[1:-1]
                extra_ids = convert_text_single_to_ids(text[pos+1:], tokenizer)[1:-1]
                extra_len, sent_limit = len(extra_ids), max_seq_len - 2 - len(extra_ids)

                assert len(sent_ids) + len(extra_ids) + 2 > max_seq_len and len(input_id) == max_seq_len, text
                head_pos = torch.nonzero(torch.eq(sent_ids, head_id), as_tuple=True)[0].item()
                tail_pos = torch.nonzero(torch.eq(sent_ids, tail_id), as_tuple=True)[0].item()
                assert 0 < tail_pos - head_pos, text
                if tail_pos - head_pos >= sent_limit:
                    head_limit, tail_limit = sent_limit // 2, sent_limit - sent_limit // 2
                    head_aux_id = convert_token_to_id(head_end if head_before_tail else tail_end, tokenizer)
                    head_aux_pos = torch.nonzero(torch.eq(sent_ids, head_aux_id), as_tuple=True)[0].item()
                    tail_aux_id = convert_token_to_id(tail_start if head_before_tail else head_start, tokenizer)
                    tail_aux_pos = torch.nonzero(torch.eq(sent_ids, tail_aux_id), as_tuple=True)[0].item()
                    head_ids = truncate_sentence_with_entity(head_limit, head_pos, head_aux_pos, sent_ids)
                    tail_ids = truncate_sentence_with_entity(tail_limit, tail_aux_pos, tail_pos, sent_ids)
                    truncated_sent_ids = torch.cat((head_ids, tail_ids), dim=0)
                else:
                    truncated_sent_ids = truncate_sentence_with_entity(sent_limit, head_pos, tail_pos, sent_ids)
                input_id[1:1 + sent_limit] = truncated_sent_ids
                input_id[-1 - extra_len:-1] = extra_ids
            assert head_id in input_id and tail_id in input_id, text
            assert mask_id in input_id, text
            assert input_id[0] == cls_id and sep_id in input_id, text
            assert torch.sum(torch.eq(input_id, cls_id)) == 1 and torch.sum(torch.eq(input_id, sep_id)) == 1
            assert torch.sum(torch.eq(input_id, mask_id)) == 1

        mask_ids = [torch.eq(input_id, tokenizer.mask_token_id) for input_id in input_ids]
        mask_ids = torch.stack(mask_ids, dim=0)
        ret = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "tags": torch.LongTensor(tags),
            "sample_keys": torch.LongTensor(sample_keys),
            "mask_ids": torch.LongTensor(mask_ids),
        }
        return batch_adaption(ret)

    def loc_collate_fn_linear_et(batch):
        warnings.filterwarnings(action='ignore')
        texts, tags, sample_keys = [], [], []
        for sample_key, tag, text in batch:
            texts.append(text)
            tags.append(tag)
            sample_keys.append(sample_key)

        if model_name == 'llama':
            mask_token = tokenizer.mask_token
            new_texts = []
            for text in texts:
                text = text.replace('[MASK]', mask_token)
                for idx in range(3):
                    text = text.replace(f'[unused{idx}]', f'<0x{idx:02}>')
                new_texts.append(text)
            texts = new_texts
            head_marker, tail_marker, blank_token = [f'<0x{idx:02}>' for idx in range(3)]
            mask_id, cls_id, sep_id = tokenizer.mask_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id
        else:
            head_marker, tail_marker, blank_token = [f'[unused{idx}]' for idx in range(3)]
            mask_id, cls_id, sep_id = tokenizer.mask_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id

        input_ids, attention_mask = convert_text_batch_to_ids(texts, tokenizer)
        assert len(texts) == len(input_ids) == len(attention_mask) == len(tags)

        head_id = convert_token_to_id(head_marker, tokenizer)
        tail_id = convert_token_to_id(tail_marker, tokenizer)

        for text, input_id in zip(texts, input_ids):
            if head_id not in input_id or tail_id not in input_id or mask_id not in input_id:
                pos = text.find(' In this sentence,')
                assert pos != -1, text
                sent_ids = convert_text_single_to_ids(text[:pos], tokenizer)[1:-1]
                extra_ids = convert_text_single_to_ids(text[pos + 1:], tokenizer)[1:-1]
                extra_len, sent_limit = len(extra_ids), max_seq_len - 2 - len(extra_ids)

                assert len(sent_ids) + len(extra_ids) + 2 > max_seq_len and len(input_id) == max_seq_len, text
                head_pos = torch.nonzero(torch.eq(sent_ids, head_id), as_tuple=True)[0].item()
                tail_pos = torch.nonzero(torch.eq(sent_ids, tail_id), as_tuple=True)[0].item()
                assert 0 < tail_pos - head_pos < sent_limit, text
                input_id[1:1 + sent_limit] = truncate_sentence_with_entity(sent_limit, head_pos, tail_pos, sent_ids)
                input_id[-1 - extra_len:-1] = extra_ids
            assert head_id in input_id and tail_id in input_id, text
            assert mask_id in input_id, text
            assert input_id[0] == cls_id and sep_id in input_id, text
            assert torch.sum(torch.eq(input_id, cls_id)) == 1 and torch.sum(torch.eq(input_id, sep_id)) == 1
            assert torch.sum(torch.eq(input_id, mask_id)) == 1

        mask_ids = [torch.eq(input_id, tokenizer.mask_token_id) for input_id in input_ids]
        mask_ids = torch.stack(mask_ids, dim=0)
        ret = {
            "input_ids": input_ids.to(dtype=torch.int),
            "attention_mask": attention_mask.to(dtype=torch.int),
            "tags": torch.LongTensor(tags),
            "sample_keys": torch.LongTensor(sample_keys),
            "mask_ids": torch.LongTensor(mask_ids),
        }
        return batch_adaption(ret)

    dataset_name = args.dataset
    if dataset_name in ['fewrel', 'tacred', 'ace']:
        return loc_collate_fn_linear_re
    elif dataset_name in ['fewnerd', 'ontonotes', 'bbn', 'chent']:
        return loc_collate_fn_linear_et
    else:
        raise NotImplementedError('invalid dataset_name: ' + dataset_name)
