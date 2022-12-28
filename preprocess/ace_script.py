import json
import random
import jsonlines


def gen_data_splits():
    random.seed(100)
    head_start, head_end, tail_start, tail_end = [f"[unused{idx}]" for idx in range(4)]
    reader = jsonlines.open('data/ace/ace2005/ace2005.jsonl')
    label_to_data = {}
    for doc in reader:
        accu_token_num = 0
        assert len(doc['sentences']) == len(doc['relations'])
        for sentence, relations in zip(doc['sentences'], doc['relations']):
            for head_s, head_e, tail_s, tail_e, rel in relations:
                head_s -= accu_token_num
                head_e -= accu_token_num
                tail_s -= accu_token_num
                tail_e -= accu_token_num
                head_ent = ' '.join(sentence[head_s:head_e+1])
                tail_ent = ' '.join(sentence[tail_s:tail_e+1])
                sub_sentence = [[token] for token in sentence]
                sub_sentence[head_s] = [head_start] + sub_sentence[head_s]
                sub_sentence[head_e] += [head_end]
                sub_sentence[tail_s] = [tail_start] + sub_sentence[tail_s]
                sub_sentence[tail_e] += [tail_end]
                sub_sentence = [' '.join(tokens) for tokens in sub_sentence]
                sub_sentence = ' '.join(sub_sentence) + f' In this sentence, {tail_ent} is the [MASK] of {head_ent}.'
                if rel not in label_to_data:
                    label_to_data[rel] = []
                label_to_data[rel].append(sub_sentence)
            accu_token_num += len(sentence)
    print(sum(len(val) for val in label_to_data.values()), set(label_to_data.keys()))
    label_set = sorted(list(label_to_data.keys()))
    label_to_tag = {lab: idx for idx, lab in enumerate(label_set)}
    with open('scripts/ace_label_to_tag.json', 'w', encoding='utf-8') as fout:
        json.dump(label_to_tag, fout)

    assert len(label_set) == 18
    sf_label_set = list(range(len(label_set)))
    random.shuffle(sf_label_set)
    split_to_tags, split_sz, accu_cnt = {}, [4] * 3 + [3] * 2, 0
    for sid in range(5):
        split_to_tags[f'p{sid+1}'] = sorted(sf_label_set[accu_cnt:accu_cnt+split_sz[sid]])
        accu_cnt += split_sz[sid]
    with open('scripts/ace_class_split_p5_tags.json', 'w', encoding='utf-8') as fout:
        json.dump(split_to_tags, fout)

    datasets = {'train': [], 'valid': [], 'test': []}
    for sid in range(1, 6):
        split_train, split_valid, split_test = [], [], []
        cur_tags = split_to_tags[f'p{sid}']
        for tag in cur_tags:
            label = label_set[tag]
            cur_data = [(tag, sent) for sent in label_to_data[label]]
            total_len = len(cur_data)
            train_len, valid_len = int(total_len * 0.8), int(total_len * 0.1)
            random.shuffle(cur_data)
            split_train += cur_data[:train_len]
            split_valid += cur_data[train_len:train_len+valid_len]
            split_test += cur_data[train_len+valid_len:]
        datasets['train'].append(split_train)
        datasets['valid'].append(split_valid)
        datasets['test'].append(split_test)
    for part, dataset in datasets.items():
        print(part, [len(item) for item in dataset])
        with open(f'data/ace/ace_split_{part}_dataset_p5_mask.json', 'w', encoding='utf-8') as fout:
            json.dump(dataset, fout)


def find_max_length():
    for part in ['train', 'valid', 'test']:
        max_sent_len, max_sent = 0, ''
        with open(f'data/ace/ace_split_{part}_dataset_p5_mask.json', encoding='utf-8') as fin:
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
