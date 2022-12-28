import json
import random


def gen_split_to_tags():
    """
    41 class, 0-40, relation categories
    """
    random.seed(100)
    part_to_file = {
        'train': 'data/tacred/train.txt',
        'valid': 'data/tacred/val.txt',
        'test':  'data/tacred/test.txt',
    }
    total_tag_set = set()
    for part in ['train', 'valid', 'test']:
        file = part_to_file[part]
        with open(file, encoding='utf-8') as fin:
            lines = fin.readlines()
        samples, tag_set = [], set()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            tag, _, sentence = line.split('\t')
            tag = int(tag) - 1
            samples.append((tag, sentence))
            tag_set.add(tag)
            total_tag_set.add(tag)
        assert total_tag_set == tag_set
        print(part, len(tag_set), sorted(list(tag_set)))
    total_tag_set = sorted(list(total_tag_set))
    split_to_tags = {}
    random.shuffle(total_tag_set)
    assert len(total_tag_set) == 41
    split_count, accu_cnt = [5] + [4] * 9, 0
    for idx in range(1, 11):
        split_to_tags[f'p{idx}'] = sorted(total_tag_set[accu_cnt:accu_cnt+split_count[idx-1]])
        accu_cnt += split_count[idx-1]
    split_to_tags['all'] = list(range(len(total_tag_set)))
    with open('scripts/tacred_class_split_p10_tags.json', 'w') as fout:
        json.dump(split_to_tags, fout)


def gen_tacred_dataset():
    part_to_file, ret = {
        'train': 'data/tacred/train.txt',
        'valid': 'data/tacred/val.txt',
        'test': 'data/tacred/test.txt',
    }, {}
    with open('scripts/tacred_class_split_p10_tags.json') as fin:
        split_to_tags = json.load(fin)
    num_splits = len(split_to_tags) - int('all' in split_to_tags)
    assert num_splits == 10
    tag_to_split_id = [-1 for _ in range(41)]
    for split, tags in split_to_tags.items():
        if split == 'all':
            continue
        for tag in tags:
            tag_to_split_id[tag] = int(split[1:]) - 1
    assert all(sid >= 0 for sid in tag_to_split_id)

    head_start, head_end, tail_start, tail_end = [f"[unused{idx}]" for idx in range(4)]

    def token_abandon(token):
        return token.startswith('http') or len(token.strip('=-*')) == 0 and len(token) > 2

    for part in ['train', 'valid', 'test']:
        file = part_to_file[part]
        with open(file, encoding='utf-8') as fin:
            lines = fin.readlines()
        split_samples = [[] for _ in range(num_splits)]
        split_mask_samples = [[] for _ in range(num_splits)]
        for line in lines:
            line = line.strip()
            if not line:
                continue
            tag, _, sentence = line.split('\t')
            tag = int(tag) - 1
            sentence = sentence.replace('<H>', head_start)
            sentence = sentence.replace('</H>', head_end)
            sentence = sentence.replace('<T>', tail_start)
            sentence = sentence.replace('</T>', tail_end)
            sentence = ' '.join([token if not token_abandon(token) else '[UNK]'
                                 for token in sentence.split(' ') if token != ''])
            cur_sid = tag_to_split_id[tag]
            split_samples[cur_sid].append((tag, sentence))

            head_start_id, head_end_id, tail_start_id, tail_end_id = \
                [sentence.find(f"[unused{idx}]") for idx in range(4)]
            assert 0 <= head_start_id < head_end_id
            head_entity = sentence[head_start_id+len(head_start)+1:head_end_id-1]
            assert 0 <= tail_start_id < tail_end_id
            tail_entity = sentence[tail_start_id+len(tail_start)+1:tail_end_id-1]
            sentence += f' In this sentence, {tail_entity} is the [MASK] of {head_entity}.'
            split_mask_samples[cur_sid].append((tag, sentence))
        with open(f'data/tacred/tacred_split_{part}_dataset_p10.json', 'w', encoding='utf-8') as fout:
            json.dump(split_samples, fout)
        with open(f'data/tacred/tacred_split_{part}_dataset_p10_mask.json', 'w', encoding='utf-8') as fout:
            json.dump(split_mask_samples, fout)
        print(part, [len(split_set) for split_set in split_samples])


def find_max_length():
    for part in ['train', 'valid', 'test']:
        max_sent_len, max_sent = 0, ''
        with open(f'data/tacred/tacred_split_{part}_dataset_p10_mask.json', encoding='utf-8') as fin:
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
    # gen_split_to_tags()
    # gen_tacred_dataset()
    find_max_length()
