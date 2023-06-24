import os
import json
import math
import torch
import pickle
import random
from tqdm import trange
from .data_utils import get_tokenizer


class DataSampler:

    def __init__(self, args, seed=None):
        self.set_path(args)
        self.args = args
        temp_name = [args.dataset, args.seed]
        if 'bert' in args.bert_path:
            temp_name.append('bert')
        # if args.dynamic_sampling:
        #     temp_name.append('dynamic')
        file_name = "{}.pkl".format(
            "-".join([str(x) for x in temp_name])
        )
        mid_dir = "processed_data/"
        os.makedirs(mid_dir, exist_ok=True)
        self.save_data_path = os.path.join(mid_dir, file_name)

        self.tokenizer = get_tokenizer(args)
        args.mask_token_id = self.tokenizer.mask_token_id
        args.pad_token_id = self.tokenizer.pad_token_id

        # read relation data
        self.id2rel, self.rel2id = self._read_relations(args)

        # random sampling
        self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)

        # regenerate data
        self.training_dataset, self.valid_dataset, self.test_dataset = self._read_data()

        # record relations
        self.seen_relations = []
        self.history_valid_data = {}
        self.history_test_data = {}

        self.batch, self.task_length = 0, args.task_length
        with open(args.split_file, encoding='utf-8') as fin:
            self.class_split = json.load(fin)
        self.relation_num = len(self.id2rel)
        self.rel2split = [-1 for _ in range(self.relation_num)]
        for split, tags in self.class_split.items():
            if split[0] != 'p':
                continue
            sid = int(split[1:]) - 1
            for tag in tags:
                self.rel2split[tag] = sid
        assert all(sid >= 0 for sid in self.rel2split)

    @staticmethod
    def set_path(args):
        args.rel2id_file = f"scripts/{args.dataset}_label_to_tag.json"
        args.split_file = f"scripts/{args.dataset}_class_split_p{args.task_length}_tags.json"
        args.train_data_path = f"data/{args.dataset}/{args.dataset}_split_train_dataset_p{args.task_length}_key.json"
        args.valid_data_path = f"data/{args.dataset}/{args.dataset}_split_valid_dataset_p{args.task_length}_key.json"
        args.test_data_path = f"data/{args.dataset}/{args.dataset}_split_test_dataset_p{args.task_length}_key.json"

    def set_seed(self, seed):
        self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)

    def skip_step(self, step: int):
        for _ in range(step):
            self.__next__()

    def __iter__(self):
        return self

    def __next__(self):

        if self.batch == self.task_length:
            raise StopIteration()

        cur_split = f'p{self.batch + 1}'
        indexes = self.class_split[cur_split]
        self.batch += 1

        current_relations = []
        cur_training_data = {}
        cur_valid_data = {}
        cur_test_data = {}

        history_rel_num = len(self.seen_relations)

        for index in indexes:
            current_relations.append(self.id2rel[index])
            self.seen_relations.append(self.id2rel[index])
            cur_training_data[self.id2rel[index]] = self.training_dataset[index]
            cur_valid_data[self.id2rel[index]] = self.valid_dataset[index]
            cur_test_data[self.id2rel[index]] = self.test_dataset[index]
            self.history_valid_data[self.id2rel[index]] = self.valid_dataset[index]
            self.history_test_data[self.id2rel[index]] = self.test_dataset[index]

        return cur_training_data, cur_valid_data, cur_test_data, \
            current_relations, self.history_valid_data, self.history_test_data, self.seen_relations, history_rel_num

    def convert_token_to_id(self, token: str):
        if 'cpm' not in self.args.bert_path:
            return self.tokenizer.convert_tokens_to_ids(token)
        else:
            return self.tokenizer.encoder[token]

    def convert_text_single_to_ids(self, text: str):
        if 'cpm' in self.args.bert_path:
            if not text.startswith(self.tokenizer.bos_token):
                text = self.tokenizer.bos_token + text
            if not text.endswith(self.tokenizer.eos_token):
                text = text + self.tokenizer.eos_token
            return torch.LongTensor(self.tokenizer.encode(text)[0])
        elif 'llama' in self.args.bert_path:
            encode_res = self.tokenizer.encode(text)
            assert encode_res[0] == self.tokenizer.bos_token_id
            assert encode_res[-1] != self.tokenizer.eos_token_id
            encode_res.append(self.tokenizer.eos_token_id)
            return torch.LongTensor(encode_res)
        elif 'bert' in self.args.bert_path:
            return self.tokenizer(text, return_tensors='pt').input_ids.squeeze(0)
        else:
            raise NotImplementedError(f'invalid bert_path: {self.args.bert_path}')

    def _read_data(self):
        if os.path.isfile(self.save_data_path):
            with open(self.save_data_path, 'rb') as f:
                datas = pickle.load(f)
            train_dataset, valid_dataset, test_dataset = datas
            return train_dataset, valid_dataset, test_dataset
        else:
            num_rel = len(self.id2rel)

            data_path = {
                'train': self.args.train_data_path,
                'valid': self.args.valid_data_path,
                'test': self.args.test_data_path,
            }
            datasets = {
                'train': [[] for _ in range(num_rel)],
                'valid': [[] for _ in range(num_rel)],
                'test': [[] for _ in range(num_rel)],
            }
            for part, path in data_path.items():
                with open(path, encoding='utf-8') as fin:
                    base_data = json.load(fin)
                if part == 'train':
                    for ds_id in range(len(base_data)):
                        new_dataset = []
                        for sample_key, tag, text in base_data[ds_id]:
                            pos = text.find(' In this sentence,')
                            if len(text[pos + 1:].split(' ')) > 50:
                                continue
                            new_dataset.append((sample_key, tag, text))
                        base_data[ds_id] = new_dataset
                total_cnt = sum(len(split) for split in base_data)
                pbar = trange(total_cnt, desc=part)
                for split in base_data:
                    for sample_key, tag, text in split:
                        if 'bert' in self.args.bert_path:
                            tokenized_res = self.tokenizer.encode(
                                text, return_tensors='pt', truncation=True, max_length=self.args.max_length
                            ).squeeze(0)
                        elif 'llama' in self.args.bert_path:
                            text = text.replace('[MASK]', self.tokenizer.mask_token)
                            for idx in range(5):
                                text = text.replace(f'[unused{idx}]', f'<0x{idx:02}>')
                            encode_res = self.tokenizer.encode(text)[:self.args.max_length]
                            assert encode_res[0] == self.tokenizer.bos_token_id
                            assert encode_res[-1] != self.tokenizer.eos_token_id
                            if len(encode_res) == self.args.max_length:
                                encode_res[-1] = self.tokenizer.eos_token_id
                            else:
                                encode_res.append(self.tokenizer.eos_token_id)
                            tokenized_res = torch.LongTensor(encode_res)
                        else:
                            raise NotImplementedError(f'invalid bert_path: {self.args.bert_path}')
                        if self.args.dataset in ['fewrel', 'tacred', 'ace']:
                            tokenized_res = self.fix_marker_re(text, tokenized_res)
                        else:
                            tokenized_res = self.fix_marker_et(text, tokenized_res)
                        tokenized_sample = {'relation': tag, 'tokens': tokenized_res}
                        datasets[part][tag].append(tokenized_sample)
                        pbar.update()
                pbar.close()

            train_dataset, valid_dataset, test_dataset = datasets['train'], datasets['valid'], datasets['test']
            with open(self.save_data_path, 'wb') as f:
                pickle.dump((train_dataset, valid_dataset, test_dataset), f)
            return train_dataset, valid_dataset, test_dataset

    def fix_marker_re(self, text, input_id):
        assert self.args.dataset in ['fewrel', 'tacred', 'ace']
        tokenizer, max_seq_len = self.tokenizer, self.args.max_length
        if 'bert' in self.args.bert_path:
            head_start, head_end, tail_start, tail_end, blank_token = [f'[unused{idx}]' for idx in range(5)]
            mask_id, cls_id, sep_id = tokenizer.mask_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id
        elif 'llama' in self.args.bert_path:
            head_start, head_end, tail_start, tail_end, blank_token = [f'<0x{idx:02}>' for idx in range(5)]
            mask_id, cls_id, sep_id = tokenizer.mask_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id
        else:
            raise NotImplementedError(f'invalid bert_path: {self.args.bert_path}')

        # find head_before_tail
        pos = text.find(head_start)
        assert pos != -1, text
        pos = text.find(tail_end, pos)
        if pos == -1:
            pos = text.find(tail_start)
            assert pos != -1, text
            pos = text.find(head_end, pos)
            assert pos != -1, text
            head_before_tail = False
        else:
            head_before_tail = True

        head_id = self.convert_token_to_id(head_start if head_before_tail else tail_start)
        tail_id = self.convert_token_to_id(tail_end if head_before_tail else head_end)
        if head_id not in input_id or tail_id not in input_id or mask_id not in input_id:
            pos = text.find(' In this sentence,')
            assert pos != -1, text
            sent_ids = self.convert_text_single_to_ids(text[:pos])[1:-1]
            extra_ids = self.convert_text_single_to_ids(text[pos + 1:])[1:-1]
            extra_len, sent_limit = len(extra_ids), max_seq_len - 2 - len(extra_ids)

            assert len(sent_ids) + len(extra_ids) + 2 > max_seq_len and len(input_id) == max_seq_len, text
            head_pos = torch.nonzero(torch.eq(sent_ids, head_id), as_tuple=True)[0].item()
            tail_pos = torch.nonzero(torch.eq(sent_ids, tail_id), as_tuple=True)[0].item()
            assert 0 < tail_pos - head_pos, text
            if tail_pos - head_pos >= sent_limit:
                head_limit, tail_limit = sent_limit // 2, sent_limit - sent_limit // 2
                head_aux_id = self.convert_token_to_id(head_end if head_before_tail else tail_end)
                head_aux_pos = torch.nonzero(torch.eq(sent_ids, head_aux_id), as_tuple=True)[0].item()
                tail_aux_id = self.convert_token_to_id(tail_start if head_before_tail else head_start)
                tail_aux_pos = torch.nonzero(torch.eq(sent_ids, tail_aux_id), as_tuple=True)[0].item()
                head_ids = self.truncate_sentence_with_entity(head_limit, head_pos, head_aux_pos, sent_ids)
                tail_ids = self.truncate_sentence_with_entity(tail_limit, tail_aux_pos, tail_pos, sent_ids)
                truncated_sent_ids = torch.cat((head_ids, tail_ids), dim=0)
            else:
                truncated_sent_ids = self.truncate_sentence_with_entity(sent_limit, head_pos, tail_pos, sent_ids)
            input_id[1:1 + sent_limit] = truncated_sent_ids
            input_id[-1 - extra_len:-1] = extra_ids
        assert head_id in input_id and tail_id in input_id, text
        assert mask_id in input_id, text
        assert input_id[0] == cls_id and sep_id in input_id, text
        assert torch.sum(torch.eq(input_id, cls_id)) == 1 and torch.sum(torch.eq(input_id, sep_id)) == 1
        assert torch.sum(torch.eq(input_id, mask_id)) == 1
        return input_id

    def fix_marker_et(self, text, input_id):
        assert self.args.dataset in ['fewnerd', 'ontonotes', 'bbn', 'chent']
        tokenizer, max_seq_len = self.tokenizer, self.args.max_length
        if 'bert' in self.args.bert_path:
            head_marker, tail_marker, blank_token = [f'[unused{idx}]' for idx in range(3)]
            mask_id, cls_id, sep_id = tokenizer.mask_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id
        elif 'llama' in self.args.bert_path:
            head_marker, tail_marker, blank_token = [f'<0x{idx:02}>' for idx in range(3)]
            mask_id, cls_id, sep_id = tokenizer.mask_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id
        else:
            raise NotImplementedError(f'invalid bert_path: {self.args.bert_path}')
        head_id = self.convert_token_to_id(head_marker)
        tail_id = self.convert_token_to_id(tail_marker)

        if head_id not in input_id or tail_id not in input_id or mask_id not in input_id:
            pos = text.find(' In this sentence,')
            assert pos != -1, text
            sent_ids = self.convert_text_single_to_ids(text[:pos])[1:-1]
            extra_ids = self.convert_text_single_to_ids(text[pos + 1:])[1:-1]
            extra_len, sent_limit = len(extra_ids), max_seq_len - 2 - len(extra_ids)

            assert len(sent_ids) + len(extra_ids) + 2 > max_seq_len and len(input_id) == max_seq_len, text
            head_pos = torch.nonzero(torch.eq(sent_ids, head_id), as_tuple=True)[0].item()
            tail_pos = torch.nonzero(torch.eq(sent_ids, tail_id), as_tuple=True)[0].item()
            assert 0 < tail_pos - head_pos < sent_limit, text
            input_id[1:1 + sent_limit] = self.truncate_sentence_with_entity(sent_limit, head_pos, tail_pos, sent_ids)
            input_id[-1 - extra_len:-1] = extra_ids
        assert head_id in input_id and tail_id in input_id, text
        assert mask_id in input_id, text
        assert input_id[0] == cls_id and sep_id in input_id, text
        assert torch.sum(torch.eq(input_id, cls_id)) == 1 and torch.sum(torch.eq(input_id, sep_id)) == 1
        assert torch.sum(torch.eq(input_id, mask_id)) == 1
        return input_id

    @staticmethod
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

    @staticmethod
    def _read_relations(args):
        with open(args.rel2id_file, encoding='utf-8') as fin:
            rel2id = json.load(fin)
        id2rel = ['' for _ in range(len(rel2id))]
        for rel, idx in rel2id.items():
            id2rel[idx] = rel
        assert [rel != '' for rel in id2rel]
        return id2rel, rel2id
