import json
import random


def gen_data_splits():
    random.seed(100)
    file_map = {
        'train': 'data/ontonotes/train.txt',
        'valid': 'data/ontonotes/dev.txt',
        'test': 'data/ontonotes/test.txt',
    }
    label_set, whole_dataset = set(), []
    for part in ['train', 'valid', 'test']:
        with open(file_map[part], encoding='utf-8') as fin:
            lines = [line.strip() for line in fin.readlines()]
        for line in lines:
            if not line:
                continue
            start_pos, end_pos, tokens, label = line.split('\t')
            label_set.add(label)
            whole_dataset.append((int(start_pos), int(end_pos), tokens, label))
    random.shuffle(whole_dataset)
    total_len = len(whole_dataset)
    train_len, valid_len = int(total_len * 0.8), int(total_len * 0.1)
    split_dataset = {
        'train': whole_dataset[:train_len],
        'valid': whole_dataset[train_len:train_len+valid_len],
        'test': whole_dataset[train_len+valid_len:]
    }
    label_set = sorted(list(label_set))
    print(label_set)
    assert len(label_set) == 86

    label_to_tag = {label: idx for idx, label in enumerate(label_set)}
    with open('scripts/ontonotes_label_to_tag.json', 'w', encoding='utf-8') as fout:
        json.dump(label_to_tag, fout)
    split_to_tags, tag_to_sid = {}, [-1 for _ in range(86)]
    split_numbers, all_tags, accu_cnt = [9] * 6 + [8] * 4, list(range(86)), 0
    random.shuffle(all_tags)
    for sid in range(10):
        split_to_tags[f'p{sid+1}'] = sorted(all_tags[accu_cnt:accu_cnt+split_numbers[sid]])
        for tag in split_to_tags[f'p{sid+1}']:
            tag_to_sid[tag] = sid
        accu_cnt += split_numbers[sid]
    split_to_tags['all'] = list(range(86))
    with open('scripts/ontonotes_class_split_p10_tags.json', 'w', encoding='utf-8') as fout:
        json.dump(split_to_tags, fout)

    head_marker, tail_marker, blank_token = '[unused0]', '[unused1]', '[unused2]'
    for part in ['train', 'valid', 'test']:
        cur_dataset = [[] for _ in range(10)]
        for start_pos, end_pos, tokens, label in split_dataset[part]:
            tag = label_to_tag[label]
            sid = tag_to_sid[tag]
            tokens = tokens.split(' ')
            entity_name = ' '.join(tokens[start_pos:end_pos])
            tokens = tokens[:start_pos] + [head_marker] + [entity_name] + [tail_marker] + tokens[end_pos:]
            sentence = ' '.join(tokens) + f' In this sentence, {entity_name} is a [MASK].'
            cur_dataset[sid].append((tag, sentence))
        print(part, [len(sub) for sub in cur_dataset])
        with open(f'data/ontonotes/ontonotes_split_{part}_dataset_p10_mask.json', 'w', encoding='utf-8') as fout:
            json.dump(cur_dataset, fout)


def find_max_length():
    for part in ['train', 'valid', 'test']:
        max_sent_len, max_sent = 0, ''
        with open(f'data/ontonotes/ontonotes_split_{part}_dataset_p10_mask.json', encoding='utf-8') as fin:
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
    # gen_data_splits()
    find_max_length()
