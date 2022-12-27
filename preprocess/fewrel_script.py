import json
import copy
import random


def gen_data_splits():
    random.seed(100)
    dataset = {}
    with open('data/fewrel/train_wiki.json', encoding='utf-8') as fin:
        train_dict = json.load(fin)
        dataset.update(train_dict)
        print(len(train_dict))
    with open('data/fewrel/val_wiki.json', encoding='utf-8') as fin:
        val_dict = json.load(fin)
        dataset.update(val_dict)
        print(len(val_dict))
    print(len(dataset))
    ori_label_set = sorted(list(dataset.keys()))
    label_to_tag = {lab: idx for idx, lab in enumerate(ori_label_set)}
    with open('scripts/fewrel_label_to_tag.json', 'w') as fout:
        json.dump(label_to_tag, fout)

    label_set = copy.deepcopy(ori_label_set)
    random.shuffle(label_set)
    print({lab: len(dataset[lab]) for lab in label_set})
    assert len(label_set) % 10 == 0
    split_num = len(label_set) // 10
    split_to_tags = {}
    for idx in range(10):
        cur_tags = [label_to_tag[tag] for tag in label_set[split_num*idx:split_num*(idx+1)]]
        split_to_tags[f'p{idx+1}'] = sorted(cur_tags)
    split_to_tags['all'] = list(range(len(label_set)))
    with open('scripts/fewrel_class_split_p10_tags.json', 'w') as fout:
        json.dump(split_to_tags, fout)

    head_start, head_end, tail_start, tail_end = [f"[unused{idx}]" for idx in range(4)]
    split_dataset = {'train': [], 'valid': [], 'test': []}
    for sid in range(10):
        cur_tags = split_to_tags[f'p{sid+1}']
        cur_train, cur_valid, cur_test = [], [], []
        for tag in cur_tags:
            tag_label = ori_label_set[tag]
            ori_tag_dataset = dataset[tag_label]

            tag_dataset = []
            for item in ori_tag_dataset:
                tokens = [[token] for token in item['tokens']]
                head_poses, tail_poses = item['h'][2], item['t'][2]
                for head_pos in head_poses:
                    assert head_pos[-1] - head_pos[0] == len(head_pos) - 1
                    tokens[head_pos[0]] = [head_start] + tokens[head_pos[0]]
                    tokens[head_pos[-1]] += [head_end]
                for tail_pos in tail_poses:
                    assert tail_pos[-1] - tail_pos[0] == len(tail_pos) - 1
                    tokens[tail_pos[0]] = [tail_start] + tokens[tail_pos[0]]
                    tokens[tail_pos[-1]] += [tail_end]
                tokens = [' '.join(sub_token) for sub_token in tokens]
                sentence = ' '.join([token for token in tokens if token != ' '])

                head_name, tail_name = item['h'][0], item['t'][0]
                sentence += f' In this sentence, {tail_name} is the [MASK] of {head_name}.'
                tag_dataset.append((tag, sentence))

            random.shuffle(tag_dataset)
            total_len = len(tag_dataset)
            train_len, valid_len = int(total_len * 0.8), int(total_len * 0.1)
            cur_train += tag_dataset[:train_len]
            cur_valid += tag_dataset[train_len:train_len+valid_len]
            cur_test += tag_dataset[train_len+valid_len:]
        split_dataset['train'].append(cur_train)
        split_dataset['valid'].append(cur_valid)
        split_dataset['test'].append(cur_test)
    for part, split_datasets in split_dataset.items():
        print(part, [len(sub_set) for sub_set in split_datasets])
        with open(f'data/fewrel/fewrel_split_{part}_dataset_p10_mask.json', 'w', encoding='utf-8') as fout:
            json.dump(split_datasets, fout)
    print(split_dataset['train'][0][9])


def find_max_length():
    for part in ['train', 'valid', 'test']:
        max_sent_len, max_sent = 0, ''
        with open(f'data/fewrel/fewrel_split_{part}_dataset_p10_mask.json', encoding='utf-8') as fin:
            dataset = json.load(fin)
        for split_set in dataset:
            print(part, len(split_set))
            for tag, sent in split_set:
                for token in [f'[unused{idx}]' for idx in range(4)]:
                    assert sent.find(token) != -1
                sent_len = len(sent.split(' '))
                if sent_len > max_sent_len:
                    max_sent_len = sent_len
                    max_sent = sent
        print(part, max_sent_len, max_sent)


if __name__ == '__main__':
    # gen_data_splits()
    find_max_length()
