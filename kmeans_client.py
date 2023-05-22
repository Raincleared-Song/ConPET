import os
import json
import time
import yaml
import argparse
import subprocess
import numpy as np
from tqdm import tqdm
from utils import init_db, read_cache, complex_sample
from sklearn.cluster import KMeans


global_cp_path = 'scy_test/checkpoints'
sample_num_map = {
    'fewnerd': 100,
    'ontonotes': 100,
    'bbn': 50,
    'fewrel': 50,
    'tacred': 20,
    'ace': 20,
    'chent': 100,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, help='name of the dataset', default='fewrel',
                        choices=['fewnerd', 'ontonotes', 'bbn', 'fewrel', 'tacred', 'ace', 'chent'])
    parser.add_argument('--method_type', type=str, choices=['prompt', 'linear', 'marker'])
    parser.add_argument('--start', type=int, help='start split', default=1)
    parser.add_argument('--cycle_suffix', type=str, help='the suffix of checkpoint path')
    parser.add_argument('--server_name', type=str, default='a100node4')
    parser.add_argument('--global_cp_path', type=str, default='')
    args = parser.parse_args()

    global global_cp_path
    if args.global_cp_path != '':
        global_cp_path = args.global_cp_path

    total_parts = 'p5' if args.dataset_name == 'ace' else 'p10'
    total_bound = int(total_parts[1:])
    method_prefix_map = {'prompt': 'pt', 'marker': 'mk', 'linear': 'li'}
    exp_prefix = method_prefix_map[args.method_type]
    cycle_suffix = '_' + args.cycle_suffix if args.cycle_suffix != '' else ''
    disk_prefix = '/data2' if args.server_name in ['server109', 'server110'] else '/data'
    sample_num = sample_num_map[args.dataset_name]
    if 'emar' in cycle_suffix or 'eaemr' in cycle_suffix:
        total_bound += 1
    for idx in range(args.start, total_bound):
        print('=' * 30)
        cur_split = f'p{idx}'
        exp_path = f'{args.dataset_name}_supervised_{exp_prefix}_fine_{total_parts}_bert_large_' \
                   f'lora4_mk00_{cur_split}{cycle_suffix}'
        scp_cmd = ['scp', f'{args.server_name}:{disk_prefix}/'
                          f'private/{global_cp_path}/{exp_path}/cache_flag', './']
        print(' '.join(scp_cmd))
        print('waiting for', exp_path)
        while True:
            subp = subprocess.Popen(scp_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subp.wait()
            if subp.returncode == 0:
                print()
                break
            time.sleep(300)

        time.sleep(10)
        db_path = 'songchenyang/continual_re_typer/databases'
        scp_cmd = ['scp', '-r', f'{args.server_name}:{disk_prefix}/private/'
                                f'{db_path}/{args.dataset_name}{cycle_suffix}.db', './databases/']
        print(' '.join(scp_cmd))
        subp = subprocess.Popen(scp_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subp.wait()
        assert subp.returncode == 0
        init_db(f'databases/{args.dataset_name}{cycle_suffix}.db')

        os.makedirs(f'checkpoints/{exp_path}', exist_ok=True)
        scp_cmd = ['scp', '-r', f'{args.server_name}:{disk_prefix}/private/'
                                f'{global_cp_path}/{exp_path}/ori_config.yaml', f'checkpoints/{exp_path}/']
        print(' '.join(scp_cmd))
        subp = subprocess.Popen(scp_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subp.wait()
        assert subp.returncode == 0

        config_path = f'checkpoints/{exp_path}/ori_config.yaml'
        with open(config_path, encoding='utf-8') as fin:
            config_base = yaml.load(fin, yaml.Loader)
        assert config_base['train']['continual_method'] in ['emr', 'our_abl']
        assert config_base['dataset']['batch_limit_policy'] == 0 and \
               (idx == 1 or config_base['dataset']['use_selected'])
        config_base['logging']['path_base'] = 'checkpoints'
        model_name = config_base['plm']['model_name']
        with open(config_path, 'w', encoding='utf-8') as fout:
            yaml.dump(config_base, fout)

        # kmeans sampling
        train_data_path = f'data/{args.dataset_name}/{args.dataset_name}_split_train_dataset_{total_parts}_key.json'
        with open(train_data_path, encoding='utf-8') as fin:
            train_dataset = json.load(fin)[idx - 1]
        new_dataset = []
        for sample_key, tag, text in train_dataset:
            pos = text.find(' In this sentence,')
            if len(text[pos + 1:].split(' ')) > 50:
                continue
            new_dataset.append((sample_key, tag, text))
        train_dataset = new_dataset

        logit_map = {}
        float_type = np.float16 if model_name not in ['cpm', 'llama'] else np.float32
        for sample_key, tag, text in train_dataset:
            if tag not in logit_map:
                logit_map[tag] = []
            logit = read_cache(f'p{idx}', sample_key, dtype=float_type)
            if logit is None:
                from IPython import embed
                embed()
                exit()
            assert logit is not None, f'p{idx}-{sample_key}'
            logit_map[tag].append((sample_key, logit))
        selected_sample_keys = {'train_infer': set(), 'valid_groups': set()}
        for tag, samples in tqdm(logit_map.items()):
            cur_tag_samples_keys = []
            sample_keys = [sample[0] for sample in samples]
            sample_states = np.stack([sample[1] for sample in samples])
            num_clusters = min(sample_num, len(sample_keys))
            print(f'clustering tag {tag} sample number {len(sample_keys)}')
            distances = KMeans(n_clusters=num_clusters,
                               random_state=config_base['dataset']['seed']).fit_transform(sample_states)
            for cid in range(num_clusters):
                sel_index = np.argmin(distances[:, cid])
                cur_tag_samples_keys.append(sample_keys[sel_index])
            train_samples = complex_sample(range(len(cur_tag_samples_keys)),
                                           size=int(np.ceil(len(cur_tag_samples_keys) * 0.8)), replace=False)
            train_samples = set(train_samples)
            for sid in range(len(cur_tag_samples_keys)):
                if sid in train_samples:
                    selected_sample_keys['train_infer'].add(cur_tag_samples_keys[sid])
                else:
                    selected_sample_keys['valid_groups'].add(cur_tag_samples_keys[sid])
        selected_sample_keys['valid_groups'] -= selected_sample_keys['train_infer']
        selected_samples = {'train_infer': [], 'valid_groups': []}
        for key, tag, text in train_dataset:
            if key in selected_sample_keys['train_infer']:
                selected_samples['train_infer'].append((key, tag, text))
            elif key in selected_sample_keys['valid_groups']:
                selected_samples['valid_groups'].append((key, tag, text))
        print(f'selected {len(selected_samples["train_infer"])} train_infer samples')
        print(f'selected {len(selected_samples["valid_groups"])} valid_groups samples')
        assert len(selected_samples['train_infer']) == len(selected_sample_keys['train_infer'])
        assert len(selected_samples['valid_groups']) == len(selected_sample_keys['valid_groups'])
        save_path = f'cache/{args.dataset_name}_continual_{total_parts}_selected_{exp_prefix}' \
                    f'_{cur_split}{cycle_suffix}.json'
        with open(save_path, 'w', encoding='utf-8') as fout:
            json.dump(selected_samples, fout)

        scp_cmd = ['scp', save_path, f'{args.server_name}:{disk_prefix}/'
                                     f'private/songchenyang/continual_re_typer/cache/']
        print(' '.join(scp_cmd))
        subp = subprocess.Popen(scp_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subp.wait()
        assert subp.returncode == 0
        print('=' * 30)


if __name__ == '__main__':
    main()
