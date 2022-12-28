import json
import random


def get_split_data():
    random.seed(200)
    file_map = {
        'train': 'data/bbn/train.txt',
        'valid': 'data/bbn/dev.txt',
        'test': 'data/bbn/test.txt',
    }
    label_to_data = {}
    head_marker, tail_marker, blank_token = '[unused0]', '[unused1]', '[unused2]'
    for part, fp in file_map.items():
        with open(fp, encoding='utf-8') as fin:
            lines = [line.strip() for line in fin.readlines()]
        for line in lines:
            if not line:
                continue
            start_pos, end_pos, tokens, label = line.split('\t')
            start_pos, end_pos = int(start_pos), int(end_pos)
            tokens = tokens.split(' ')
            entity_name = ' '.join(tokens[start_pos:end_pos])
            tokens = tokens[:start_pos] + [head_marker] + [entity_name] + [tail_marker] + tokens[end_pos:]
            sentence = ' '.join(tokens) + f' In this sentence, {entity_name} is a [MASK].'
            if label not in label_to_data:
                label_to_data[label] = []
            label_to_data[label].append(sentence)
    label_set = sorted(list(label_to_data.keys()))
    label_to_tag = {lab: idx for idx, lab in enumerate(label_set)}
    with open('scripts/bbn_label_to_tag.json', 'w', encoding='utf-8') as fout:
        json.dump(label_to_tag, fout)

    assert len(label_set) == 46
    sf_label_set = list(range(len(label_set)))
    random.shuffle(sf_label_set)
    split_to_tags, split_sz, accu_cnt = {}, [5] * 6 + [4] * 4, 0
    for sid in range(10):
        split_to_tags[f'p{sid + 1}'] = sorted(sf_label_set[accu_cnt:accu_cnt + split_sz[sid]])
        accu_cnt += split_sz[sid]
    with open('scripts/bbn_class_split_p10_tags.json', 'w', encoding='utf-8') as fout:
        json.dump(split_to_tags, fout)

    datasets = {'train': [], 'valid': [], 'test': []}
    for sid in range(1, 11):
        split_train, split_valid, split_test = [], [], []
        cur_tags = split_to_tags[f'p{sid}']
        for tag in cur_tags:
            label = label_set[tag]
            cur_data = [(tag, sent) for sent in label_to_data[label]]
            total_len = len(cur_data)
            train_len, valid_len = int(total_len * 0.8), int(total_len * 0.1)
            random.shuffle(cur_data)
            split_train += cur_data[:train_len]
            split_valid += cur_data[train_len:train_len + valid_len]
            split_test += cur_data[train_len + valid_len:]
        datasets['train'].append(split_train)
        datasets['valid'].append(split_valid)
        datasets['test'].append(split_test)
    for part, dataset in datasets.items():
        print(part, [len(item) for item in dataset])
        with open(f'data/bbn/bbn_split_{part}_dataset_p10_mask.json', 'w', encoding='utf-8') as fout:
            json.dump(dataset, fout)


def find_max_length():
    for part in ['train', 'valid', 'test']:
        max_sent_len, max_sent = 0, ''
        with open(f'data/bbn/bbn_split_{part}_dataset_p10_mask.json', encoding='utf-8') as fin:
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


if __name__ == '__main__':
    # get_split_data()
    find_max_length()
