import json
import torch
import random
from .data_utils import complex_sample, get_tokenizer


class CustomLoader:
    def __init__(self, args, cur_split_id, dataset, epoch_num, batch_size, shuffle,
                 num_workers, drop_last, seen_relations):
        self.args = args
        self.cur_split_id = cur_split_id
        self.dataset = dataset
        self.data_len = len(dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.min_sample_num = args.min_sample_num
        self.num_workers = num_workers  # not used
        self.drop_last = drop_last  # not used
        self.seen_relations = seen_relations
        self.int_type = torch.int if 'llama' in args.bert_path else torch.long
        self.raw_labels = torch.zeros(self.data_len, dtype=self.int_type)
        self.continual_split_to_tags, self.continual_tag_to_split = self.load_tag_to_split()
        self.old_dataset, self.new_dataset, self.old_tag_to_indexes, self.new_tag_to_indexes = self.init_dataset_split()
        self.old_split_ids = sorted(list(self.old_dataset.keys()))
        if shuffle:
            for split_id, samples in self.old_dataset.items():
                random.shuffle(samples)
            random.shuffle(self.new_dataset)
        self.batch_new_old_ratio = 1
        if self.batch_new_old_ratio > 0:
            self.new_sample_num = int(batch_size * self.batch_new_old_ratio / (self.batch_new_old_ratio + 1))
            self.old_sample_num = batch_size - self.new_sample_num
        else:
            new_old_ratio = - self.batch_new_old_ratio
            self.old_sample_num = int(batch_size * new_old_ratio / (new_old_ratio + 1))
            self.new_sample_num = batch_size - self.old_sample_num
        self.batch_limit_policy = 2
        self.total_len = args.batch_limit * epoch_num
        self.tokenizer = get_tokenizer(args)
        self.cached_batches = []
        self.iter_pointer = 0
        self.get_sample_batches()

    def get_original_tag_to_data(self):
        tag_to_dataset = {}
        for entry in self.dataset:
            tag = entry['relation']
            if tag not in tag_to_dataset:
                tag_to_dataset[tag] = []
            tag_to_dataset[tag].append(entry)
        return tag_to_dataset

    def load_tag_to_split(self):
        args = self.args
        with open(f'scripts/{args.dataset}_class_split_p{args.task_length}_tags.json') as fin:
            continual_split_to_tags = json.load(fin)
        label_num = sum(len(tags) for split, tags in continual_split_to_tags.items() if split != 'all')
        continual_tag_to_split = ['' for _ in range(label_num)]
        for split, tags in continual_split_to_tags.items():
            if split == 'all':
                continue
            for tag in tags:
                continual_tag_to_split[tag] = split
        assert all(split != '' for split in continual_tag_to_split)
        return continual_split_to_tags, continual_tag_to_split

    def init_dataset_split(self):
        """
        split the total dataset to old samples and new samples
        dataset entry: {"relation": tag, "tokens": tokenized_res}
        :return: old_dataset, new_dataset
        """
        tag_to_split = self.continual_tag_to_split
        assert self.cur_split_id > 1
        old_part_ids = list(range(1, self.cur_split_id))
        old_dataset, new_dataset = {}, []
        old_tag_to_indexes, new_tag_to_indexes = {}, {}
        for eid, entry in enumerate(self.dataset):
            entry['id'] = eid
            tag = entry['relation']
            split_id = int(tag_to_split[tag][1:])
            if split_id != self.cur_split_id:
                assert split_id in old_part_ids
                if split_id not in old_dataset:
                    old_dataset[split_id] = []
                old_dataset[split_id].append(entry)
                if tag not in old_tag_to_indexes:
                    old_tag_to_indexes[tag] = []
                old_tag_to_indexes[tag].append(eid)
            else:
                new_dataset.append(entry)
                if tag not in new_tag_to_indexes:
                    new_tag_to_indexes[tag] = []
                new_tag_to_indexes[tag].append(eid)
            self.raw_labels[eid] = tag
        assert len(old_tag_to_indexes) + len(new_tag_to_indexes) == len(self.seen_relations)
        print(f'old dataset pool: {len(old_dataset)}; new dataset pool: {len(new_dataset)}')
        return old_dataset, new_dataset, old_tag_to_indexes, new_tag_to_indexes

    def collate_fn(self, batch):
        pad_max = max(len(entry['tokens']) for entry in batch)
        input_ids, attention_mask, mask_ids, tags, sample_keys = [], [], [], [], []
        for entry in batch:
            sample_keys.append(entry['id'])
            tags.append(entry['relation'])
            input_id = entry['tokens']
            pad_len = pad_max - len(input_id)
            attention_mask.append([1] * len(input_id) + [0] * pad_len)
            input_id = torch.cat([input_id, torch.LongTensor([0] * pad_len)], dim=0)
            input_ids.append(input_id)
            mask_ids.append(torch.eq(input_id, self.tokenizer.mask_token_id))

        ret_batch = {
            "input_ids": torch.stack(input_ids).to(dtype=self.int_type),
            "attention_mask": torch.tensor(attention_mask).to(dtype=self.int_type),
            "tags": torch.tensor(tags).to(dtype=self.int_type),
            "sample_keys": torch.tensor(sample_keys).to(dtype=torch.long),
            "mask_ids": torch.stack(mask_ids).to(dtype=torch.bool),
        }
        # cpu_labels, tokens, attention_mask, mask_ids, ind
        return ret_batch['tags'], ret_batch['input_ids'], ret_batch['attention_mask'], \
            ret_batch['mask_ids'], ret_batch['sample_keys']

    def get_sample_batches(self):
        assert len(self.seen_relations) == sum(
            len(self.continual_split_to_tags[f"p{idx+1}"]) for idx in range(self.cur_split_id))

        all_old_data, all_new_data = [], []
        for tag, indexes in self.old_tag_to_indexes.items():
            basic_num = min(len(indexes), self.min_sample_num)
            assert basic_num > 0
            all_old_data_indexes = complex_sample(indexes, basic_num, replace=False)
            all_old_data += [self.dataset[idx] for idx in all_old_data_indexes]
        for _ in range(self.total_len):
            for _ in range(self.old_sample_num):
                cur_split_id = complex_sample(self.old_split_ids, 1, replace=False)[0]
                all_old_data.append(complex_sample(self.old_dataset[cur_split_id], 1, replace=False)[0])

        for tag, indexes in self.new_tag_to_indexes.items():
            basic_num = min(len(indexes), self.min_sample_num)
            assert basic_num > 0
            all_new_data_indexes = complex_sample(indexes, basic_num, replace=False)
            all_new_data += [self.dataset[idx] for idx in all_new_data_indexes]
        for _ in range(self.total_len):
            all_new_data += complex_sample(self.new_dataset, self.new_sample_num, replace=False)

        total_old_sample_num = self.old_sample_num * self.total_len
        total_new_sample_num = self.new_sample_num * self.total_len
        all_old_data, all_new_data = all_old_data[:total_old_sample_num], all_new_data[:total_new_sample_num]
        assert len(all_old_data) == total_old_sample_num and len(all_new_data) == total_new_sample_num

        random.shuffle(all_old_data)
        random.shuffle(all_new_data)
        for bid in range(self.total_len):
            cur_batch = all_old_data[bid*self.old_sample_num:(bid+1)*self.old_sample_num]
            cur_batch += all_new_data[bid*self.new_sample_num:(bid+1)*self.new_sample_num]
            if self.shuffle:
                random.shuffle(cur_batch)
            self.cached_batches.append(cur_batch)

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter_pointer == self.total_len:
            self.iter_pointer = 0
            raise StopIteration()
        assert len(self.cached_batches) == self.total_len
        cur_batch = self.cached_batches[self.iter_pointer]
        ret = self.collate_fn(cur_batch)
        self.iter_pointer += 1
        return ret

    def __len__(self):
        return self.total_len
