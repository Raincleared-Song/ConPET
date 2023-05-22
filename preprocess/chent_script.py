import json
import random
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data():
    """
    length: 279953 legal length: 279871
    train_size: 195866
    valid_size: 42574
    test_size: 41431
    """
    data_path = '/data/private/hechaoqun/typing/DataSets/98_data/98_data.json'
    with open(data_path, encoding='utf-8') as f:
        data = json.load(f)

    label_set = set()
    legal_data = []
    for entry, label in data:
        label_set.add(label)
        text, entity = entry['text'], entry['entity']
        if text.count(entity) == 1:
            pos = text.find(entity)
            assert pos != -1 and text.find(entity, pos + len(entity)) == -1
            legal_data.append((text, entity, pos, label))
    print('label set length:', len(label_set))  # 98
    print('length:', len(data), 'legal length:', len(legal_data))  # 279953 279871
    print(legal_data[0])

    df = pd.DataFrame(legal_data, columns=["text", "entity", "pos", "label"])
    grouped_data = df.groupby("label")

    train_data_list = []
    val_data_list = []
    test_data_list = []

    for label, group in grouped_data:

        entity_grouped_data = group.groupby("entity")

        unique_entities = group["entity"].unique()
        train_entities, temp_entities = train_test_split(unique_entities, test_size=0.3, random_state=42)
        val_entities, test_entities = train_test_split(temp_entities, test_size=0.5, random_state=42)

        train_data = pd.concat([entity_grouped_data.get_group(entity) for entity in train_entities])
        val_data = pd.concat([entity_grouped_data.get_group(entity) for entity in val_entities])
        test_data = pd.concat([entity_grouped_data.get_group(entity) for entity in test_entities])

        train_data_list.append(train_data)
        val_data_list.append(val_data)
        test_data_list.append(test_data)

    train_data = pd.concat(train_data_list)
    val_data = pd.concat(val_data_list)
    test_data = pd.concat(test_data_list)

    x_train, y_train = train_data[["text", "entity", "pos"]].values.tolist(), train_data["label"].values.tolist()
    x_valid, y_valid = val_data[["text", "entity", "pos"]].values.tolist(), val_data["label"].values.tolist()
    x_test, y_test = test_data[["text", "entity", "pos"]].values.tolist(), test_data["label"].values.tolist()

    assert len(x_train) == len(y_train)
    print('train_size:', len(x_train))
    train = [({'text': item[0], 'entity': item[1], 'pos': item[2]}, label) for item, label in zip(x_train, y_train)]

    assert len(x_valid) == len(y_valid)
    print('valid_size:', len(x_valid))
    valid = [({'text': item[0], 'entity': item[1], 'pos': item[2]}, label) for item, label in zip(x_valid, y_valid)]

    assert len(x_test) == len(y_test)
    print('test_size:', len(x_test))
    test = [({'text': item[0], 'entity': item[1], 'pos': item[2]}, label) for item, label in zip(x_test, y_test)]

    with open('data/chent/train.json', 'w', encoding='utf-8') as f:
        json.dump(train, f, indent=4, ensure_ascii=False)

    with open('data/chent/valid.json', 'w', encoding='utf-8') as f:
        json.dump(valid, f, indent=4, ensure_ascii=False)

    with open('data/chent/test.json', 'w', encoding='utf-8') as f:
        json.dump(test, f, indent=4, ensure_ascii=False)


def gen_data_splits():
    """
    train [10516, 4451, 54931, 18106, 11730, 22629, 20031, 28184, 10116, 15190]
    valid [2253, 847, 12535, 3827, 2472, 4771, 4384, 6154, 2116, 3214]
    test [2236, 817, 10787, 4010, 2642, 4825, 4359, 6001, 2503, 3260]
    """
    random.seed(100)
    label_set = set()
    split_dataset = {}
    for part in ['train', 'valid', 'test']:
        local_label_set = set()
        with open(f'data/chent/{part}.json', encoding='utf-8') as fin:
            data = json.load(fin)
        for entry, label in data:
            label_set.add(label)
            local_label_set.add(label)
        assert label_set == local_label_set
        split_dataset[part] = data
    label_set = sorted(list(label_set))
    print('length of label_set:', len(label_set))
    assert len(label_set) == 98

    label_to_tag = {label: idx for idx, label in enumerate(label_set)}
    with open('scripts/chent_label_to_tag.json', 'w', encoding='utf-8') as fout:
        json.dump(label_to_tag, fout)
    split_to_tags, tag_to_sid = {}, [-1 for _ in range(98)]
    split_numbers, all_tags, accu_cnt = [10] * 8 + [9] * 2, list(range(98)), 0
    random.shuffle(all_tags)
    for sid in range(10):
        split_to_tags[f'p{sid + 1}'] = sorted(all_tags[accu_cnt:accu_cnt + split_numbers[sid]])
        for tag in split_to_tags[f'p{sid + 1}']:
            tag_to_sid[tag] = sid
        accu_cnt += split_numbers[sid]
    split_to_tags['all'] = list(range(98))
    with open('scripts/chent_class_split_p10_tags.json', 'w', encoding='utf-8') as fout:
        json.dump(split_to_tags, fout)

    head_marker, tail_marker = '[unused1]', '[unused2]'
    for part in ['train', 'valid', 'test']:
        cur_dataset = [[] for _ in range(10)]
        for entry, label in split_dataset[part]:
            text, entity, pos = entry['text'], entry['entity'], entry['pos']
            tag = label_to_tag[label]
            sid = tag_to_sid[tag]
            assert text[pos:pos+len(entity)] == entity
            sentence = text[:pos] + head_marker + entity + tail_marker + text[pos+len(entity):]
            sentence += f' 在这个句子中，{entity}是一个[MASK]。'
            cur_dataset[sid].append((tag, sentence))
        print(part, [len(sub) for sub in cur_dataset])
        with open(f'data/chent/chent_split_{part}_dataset_p10_mask.json', 'w', encoding='utf-8') as fout:
            json.dump(cur_dataset, fout)


if __name__ == "__main__":
    split_data()
    gen_data_splits()
