import json


def add_sample_keys():
    for data_name in ['fewnerd', 'ontonotes', 'bbn', 'fewrel', 'tacred', 'ace', 'chent']:
        for part in ['train', 'valid', 'test']:
            total_parts = 'p5' if data_name == 'ace' else 'p10'
            dataset_path = f'data/{data_name}/{data_name}_split_{part}_dataset_{total_parts}_mask.json'
            with open(dataset_path, encoding='utf-8') as fin:
                dataset = json.load(fin)
            cur_sample_idx = 0
            new_dataset = []
            for sid, split in enumerate(dataset):
                new_split = []
                for tag, text in split:
                    sample_key = f'{part}-{sid}-{cur_sample_idx}'
                    cur_sample_idx += 1
                    new_split.append((sample_key, tag, text))
                new_dataset.append(new_split)
            dataset_path = f'data/{data_name}/{data_name}_split_{part}_dataset_{total_parts}_key.json'
            with open(dataset_path, 'w', encoding='utf-8') as fout:
                json.dump(new_dataset, fout)


if __name__ == "__main__":
    add_sample_keys()
