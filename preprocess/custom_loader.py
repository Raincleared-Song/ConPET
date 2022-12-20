import random
from global_var import GLOBAL
from utils import complex_sample


class CustomLoader:
    def __init__(self, config, mode, dataset, batch_size, shuffle, num_workers, collate_fn, drop_last):
        assert mode != 'test'
        self.config = config
        self.mode = mode
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers  # not used
        self.collate_fn = collate_fn
        self.drop_last = drop_last  # not used
        self.old_dataset, self.new_dataset = self.init_dataset_split(dataset)
        self.old_split_ids = sorted(list(self.old_dataset.keys()))
        if shuffle:
            for split_id, samples in self.old_dataset.items():
                random.shuffle(samples)
            random.shuffle(self.new_dataset)
        self.batch_new_old_ratio = config['dataset']['batch_new_old_ratio']
        self.new_sample_num = int(batch_size * self.batch_new_old_ratio / (self.batch_new_old_ratio + 1))
        self.old_sample_num = batch_size - self.new_sample_num
        self.batch_limit = config['dataset']['batch_limit']
        self.batch_limit_policy = config['dataset']['batch_limit_policy']
        assert 0 < self.batch_limit_policy <= 2
        if self.batch_limit_policy == 1:
            self.total_len = int(config['dataset']['special_part'][1:]) * self.batch_limit
        else:
            self.total_len = self.batch_limit
        factor = 0.8 if self.mode == 'train' else 0.2
        self.total_len = int(self.total_len * factor)
        self.iter_pointer = 0

    def init_dataset_split(self, dataset):
        """
        split the total dataset to old samples and new samples
        :return: old_dataset, new_dataset
        """
        tag_to_split = GLOBAL['continual_tag_to_split']
        special_part_id = int(self.config['dataset']['special_part'][1:])
        assert special_part_id > 1
        old_part_ids = list(range(1, special_part_id))
        old_dataset, new_dataset = {}, []
        for sample_key, sample in dataset:
            tag = sample[0]
            split_id = int(tag_to_split[tag][1:])
            if split_id != special_part_id:
                assert split_id in old_part_ids
                if split_id not in old_dataset:
                    old_dataset[split_id] = []
                old_dataset[split_id].append((sample_key, sample))
            else:
                new_dataset.append((sample_key, sample))
        print(f'old dataset pool: {len(old_dataset)}; new dataset pool: {len(new_dataset)}')
        return old_dataset, new_dataset

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter_pointer == self.total_len:
            self.iter_pointer = 0
            raise StopIteration()
        cur_batch = []
        for sample_id in range(self.old_sample_num):
            cur_split_id = complex_sample(self.old_split_ids, 1, replace=False)[0]
            cur_batch.append(complex_sample(self.old_dataset[cur_split_id], 1, replace=False)[0])
        cur_batch += complex_sample(self.new_dataset, self.new_sample_num, replace=False)
        if self.shuffle:
            random.shuffle(cur_batch)
        ret = self.collate_fn(cur_batch)
        self.iter_pointer += 1
        return ret

    def __len__(self):
        return self.total_len
