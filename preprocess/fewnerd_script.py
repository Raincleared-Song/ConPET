import json
import random


total_bound = 20


def gen_data_splits():
    random.seed(100)
    dataset = {}
    for part in ['train', 'valid', 'test']:
        with open(f'data/fewnerd/fewnerd_split_{part}_dataset_p10_mask.json') as fin:
            ori_dataset = json.load(fin)
        cur_split_dataset = {}
        for sub_dataset in ori_dataset:
            for tag, example in sub_dataset:
                if tag not in cur_split_dataset:
                    cur_split_dataset[tag] = []
                cur_split_dataset[tag].append((tag, example))
        dataset[part] = cur_split_dataset
    assert len(dataset) == 3
    assert all(len(val) == 66 for val in dataset.values())

    tag_set_shuffle = list(range(66))
    random.shuffle(tag_set_shuffle)
    split_number, accu = [4] * 6 + [3] * 14, 0
    split_to_tags = {}
    for idx in range(total_bound):
        split_to_tags[f'p{idx+1}'] = sorted(tag_set_shuffle[accu:accu+split_number[idx]])
        accu += split_number[idx]
    split_to_tags['all'] = list(range(66))
    with open(f'scripts/fewnerd_class_split_p{total_bound}_tags.json', 'w') as fout:
        json.dump(split_to_tags, fout)

    split_dataset = {'train': [], 'valid': [], 'test': []}
    for part in ['train', 'valid', 'test']:
        cur_data = dataset[part]
        for sid in range(total_bound):
            cur_tags = split_to_tags[f'p{sid+1}']
            cur_train = []
            for tag in cur_tags:
                cur_train += cur_data[tag]
            split_dataset[part].append(cur_train)
    for part, split_datasets in split_dataset.items():
        print(part, [len(sub_set) for sub_set in split_datasets])
        with open(f'data/fewnerd/fewnerd_split_{part}_dataset_p{total_bound}_mask.json', 'w', encoding='utf-8') as fout:
            json.dump(split_datasets, fout)


def find_max_length():
    for part in ['train', 'valid', 'test']:
        max_sent_len, max_sent = 0, ''
        with open(f'data/fewnerd/fewnerd_split_{part}_dataset_p{total_bound}_mask.json', encoding='utf-8') as fin:
            dataset = json.load(fin)
        for split_set in dataset:
            print(part, len(split_set))
            for tag, sent in split_set:
                for token in [f'[unused{idx}]' for idx in range(2)]:
                    assert sent.find(token) != -1
                sent_len = len(sent.split(' '))
                if sent_len > max_sent_len:
                    max_sent_len = sent_len
                    max_sent = sent
        print(part, max_sent_len, max_sent)


def generate_fewnerd_p8():
    """
    train [14872, 17601, 14062, 95414, 66974, 33624, 75997, 21839]
    valid [2072, 2484, 2038, 13667, 9604, 4959, 10981, 2954]
    test [4064, 5007, 4104, 27259, 19095, 9562, 21579, 6232]
    """
    dataset, total_bound = {}, 8
    for part in ['train', 'valid', 'test']:
        with open(f'data/fewnerd/fewnerd_split_{part}_dataset_p10_mask.json') as fin:
            ori_dataset = json.load(fin)
        cur_split_dataset = {}
        for sub_dataset in ori_dataset:
            for tag, example in sub_dataset:
                if tag not in cur_split_dataset:
                    cur_split_dataset[tag] = []
                cur_split_dataset[tag].append((tag, example))
        dataset[part] = cur_split_dataset
    assert all(len(val) == 66 for val in dataset.values())

    with open('scripts/fewnerd_label_to_tag.json') as fin:
        label_to_tag = json.load(fin)
    prefix_to_tags = {}
    for label, tag in label_to_tag.items():
        prefix = label.split('-')[0]
        if prefix not in prefix_to_tags:
            prefix_to_tags[prefix] = []
        prefix_to_tags[prefix].append(tag)
    assert len(prefix_to_tags) == total_bound
    prefix_to_tags = sorted(list(prefix_to_tags.items()), key=lambda x: x[0])

    split_to_tags = {}
    for sid, (prefix, tags) in enumerate(prefix_to_tags):
        split = f'p{sid+1}'
        split_to_tags[split] = tags
    with open('scripts/fewnerd_class_split_p8_tags.json', 'w') as fout:
        json.dump(split_to_tags, fout)

    split_dataset = {'train': [], 'valid': [], 'test': []}
    for part in ['train', 'valid', 'test']:
        cur_data = dataset[part]
        for sid in range(total_bound):
            cur_tags = split_to_tags[f'p{sid+1}']
            cur_train = []
            for tag in cur_tags:
                cur_train += cur_data[tag]
            split_dataset[part].append(cur_train)
    for part, split_datasets in split_dataset.items():
        print(part, [len(sub_set) for sub_set in split_datasets])
        with open(f'data/fewnerd/fewnerd_split_{part}_dataset_p{total_bound}_mask.json', 'w', encoding='utf-8') as fout:
            json.dump(split_datasets, fout)


if __name__ == '__main__':
    gen_data_splits()
    find_max_length()
    generate_fewnerd_p8()
