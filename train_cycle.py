import math
import os
import time
import yaml
import shutil
import argparse
import subprocess
from utils import load_json
from matplotlib import pyplot as plt


def plot_metrics(dataset_name: str, cycle_suffix: str, exp_prefix: str, plot_metric: str, start: int):
    total_parts = 'p5' if dataset_name == 'ace' else 'p10'
    total_bound = int(total_parts[1:])
    labels = [s for s in cycle_suffix.split(',') if s]
    assert len(labels) > 0
    colors = ['red', 'blue', 'orange', 'yellow', 'green', 'cyan']
    assert len(colors) >= len(labels)
    for color, label in zip(colors, labels):
        cur_acc = []
        for idx in range(start, total_bound + 1):
            cur_split = f'p{idx}'
            exp_path = f'{dataset_name}_supervised_{exp_prefix}_fine_{total_parts}_bert_large_' \
                       f'lora4_mk00_{cur_split}{label}'
            test_metric = os.path.join('checkpoints', exp_path, 'test', 'metrics_test.json')
            if not os.path.exists(test_metric):
                continue
            test_metric = load_json(test_metric)[plot_metric]
            cur_acc.append(test_metric)
        print(label, len(cur_acc), cur_acc)
        plt.plot(range(start, len(cur_acc)+start), cur_acc, label=label, color=color, marker='*')
    plt.legend()
    plt.xticks(range(start, total_bound + 1))
    plt.title(plot_metric)
    plt.tight_layout()
    plt.savefig(f'checkpoints/cycle_plot_{plot_metric}.png')


def get_emr_replay_frequency(dataset_name: str, sid: int):
    sample_num_map = {
        'fewnerd': 100,
        'ontonotes': 100,
        'bbn': 50,
        'fewrel': 50,
        'tacred': 20,
        'ace': 20,
    }
    class_num_map = {
        'fewnerd': [7, 7, 7, 7, 7, 7, 6, 6, 6, 6],
        'ontonotes': [9, 9, 9, 9, 9, 9, 8, 8, 8, 8],
        'bbn': [5, 5, 5, 5, 5, 5, 4, 4, 4, 4],
        'fewrel': [8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        'tacred': [5, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        'ace': [4, 4, 4, 3, 3],
    }
    training_sample_map = {
        'fewnerd': [30415, 24780, 51816, 101875, 34443, 29988, 12912, 24075, 15163, 14916],
        'ontonotes': [17932, 27842, 16288, 34221, 21156, 25248, 31235, 7818, 18138, 11645],
        'bbn': [4822, 6210, 2738, 5120, 46199, 7373, 5260, 6751, 2264, 2628],
        'fewrel': [4480, 4480, 4480, 4480, 4480, 4480, 4480, 4480, 4480, 4480],
        'tacred': [651, 341, 2313, 2509, 667, 976, 1083, 248, 1469, 2755],
        'ace': [2507, 1013, 724, 777, 627],
    }
    batch_size_map = {
        'fewnerd': 16,
        'ontonotes': 16,
        'bbn': 16,
        'fewrel': 16,
        'tacred': 16,
        'ace': 16,
    }
    batch_limit_map = {
        'fewnerd': 2500,
        'ontonotes': 1250,
        'bbn': 500,
        'fewrel': 400,
        'tacred': 100,
        'ace': 100,
    }
    base_old_frequency_map = {
        'fewnerd': 50,
        'ontonotes': 10,
        'bbn': 5,
        'fewrel': 5,
        'tacred': 5,
        'ace': 5,
    }
    batch_limit_train = int(math.ceil(batch_limit_map[dataset_name] * 0.8))
    room_for_old_samples = batch_limit_train * batch_size_map[dataset_name] - \
        training_sample_map[dataset_name][sid-1]
    if room_for_old_samples <= 0:
        return base_old_frequency_map[dataset_name]
    old_class_tot = sum(cnt for cnt in class_num_map[dataset_name][:sid-1])
    sample_num_train = int(math.ceil(sample_num_map[dataset_name] * 0.8))
    replay_frequency = room_for_old_samples / (sample_num_train * old_class_tot)
    replay_frequency = int(math.ceil(replay_frequency))
    assert replay_frequency > 0
    return replay_frequency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, help='name of the dataset', default='fewrel',
                        choices=['fewnerd', 'ontonotes', 'bbn', 'fewrel', 'tacred', 'ace'])
    parser.add_argument('--use_selector', type=int, choices=[0, 1, 2],
                        help='0-none, 1-co-train, 2-selector-first')
    parser.add_argument('--teacher_forcing', help='whether to user teacher forcing', action='store_true')
    parser.add_argument('--start', type=int, help='start split', default=1)
    parser.add_argument('--topk', type=int, help='expert topk', default=1)
    parser.add_argument('--batch_limit_policy', type=int, help='batch limit policy', default=0)
    parser.add_argument('--batch_limit', type=int, help='batch sample limit', default=1000)
    parser.add_argument('--batch_new_old_ratio', type=float, help='batch new/old sample ratio', default=1)
    parser.add_argument('--cycle_suffix', type=str, help='the suffix of checkpoint path')
    parser.add_argument('--method_type', type=str, choices=['prompt', 'linear', 'marker'])
    parser.add_argument('--continual_method', type=str, choices=['our', 'ewc', 'lwf', 'emr', 'emr_abl'])
    parser.add_argument('--loss_adverse_step', type=int, default=-1)
    parser.add_argument('--seed', type=int, help='the random seed', default=100)
    parser.add_argument('--plot', action='store_true', help='to plot the test accuracies')
    parser.add_argument('--plot_metric', type=str, help='the test metric to plot', default='accuracy')
    parser.add_argument('--device', type=str, help='the device to be used')
    args = parser.parse_args()

    method_prefix_map = {'prompt': 'pt', 'marker': 'mk', 'linear': 'li'}
    if args.plot:
        plot_metrics(args.dataset_name, args.cycle_suffix,
                     method_prefix_map[args.method_type], args.plot_metric, args.start)
        return

    bak_config_path = f'configs/{args.dataset_name}_continual_typing_{args.cycle_suffix}.yaml'
    if not os.path.exists(bak_config_path):
        shutil.copy(f'configs/{args.dataset_name}_continual_typing.yaml', bak_config_path)
    with open(bak_config_path, encoding='utf-8') as fin:
        config_base = yaml.load(fin, yaml.Loader)
    cycle_suffix = '_' + args.cycle_suffix if args.cycle_suffix != '' else ''
    exp_prefix = method_prefix_map[args.method_type]
    total_parts = config_base['dataset']['total_parts']
    total_bound = int(total_parts[1:])
    for idx in range(args.start, total_bound + 1):
        cur_split = f'p{idx}'
        config_base['train']['continual_method'] = args.continual_method
        config_base['dataset']['method_type'] = args.method_type
        config_base['dataset']['special_part'] = cur_split
        if args.continual_method.startswith('emr'):
            config_base['dataset']['extra_special_part'] = ','.join([f'p{sid}' for sid in range(1, idx)])
        else:
            config_base['dataset']['extra_special_part'] = ''
        exp_path = f'{args.dataset_name}_supervised_{exp_prefix}_fine_{total_parts}_bert_large_' \
                   f'lora4_mk00_{cur_split}{cycle_suffix}'
        config_base['logging']['unique_string'] = exp_path
        config_base['logging']['cycle_suffix'] = args.cycle_suffix
        exp_path = os.path.join('checkpoints', exp_path)
        config_base['train']['expert_topk'] = args.topk
        config_base['train']['loss_adverse_step'] = args.loss_adverse_step
        if idx >= 2:
            config_base['train']['train_expert_selector'] = args.use_selector
        else:
            config_base['train']['train_expert_selector'] = 0
        if args.use_selector == 0:
            config_base['train']['use_expert_selector'] = False
        elif args.use_selector == 1:
            config_base['train']['use_expert_selector'] = True if idx >= 3 else False
        else:
            config_base['train']['use_expert_selector'] = True if idx >= 2 else False
        config_base['train']['teacher_forcing'] = args.teacher_forcing
        config_base['dataset']['batch_limit_policy'] = args.batch_limit_policy
        config_base['dataset']['batch_limit'] = args.batch_limit
        config_base['dataset']['batch_new_old_ratio'] = args.batch_new_old_ratio
        config_base['dataset']['seed'] = config_base['reproduce']['seed'] = args.seed
        config_base['dataset']['use_selected'] = args.continual_method == 'emr'
        if args.continual_method == 'emr' and idx >= 2:
            config_base['dataset']['replay_frequency'] = get_emr_replay_frequency(args.dataset_name, idx)
        with open(f'configs/{args.dataset_name}_continual_typing{cycle_suffix}.yaml', 'w', encoding='utf-8') as fout:
            yaml.dump(config_base, fout)
        print(cur_split, config_base)
        print('=' * 30)

        if args.continual_method == 'emr' and idx >= 2:
            assert config_base['train']['train_expert_selector'] == 0 \
                   and not config_base['train']['use_expert_selector'] and config_base['dataset']['use_selected'] \
                   and config_base['dataset']['batch_limit_policy'] == 0
            wait_file = f'cache/{args.dataset_name}_continual_{total_parts}_selected_{exp_prefix}' \
                        f'_p{idx-1}{cycle_suffix}.json'
            print('waiting for ', wait_file, '.' * 30)
            while True:
                if os.path.exists(wait_file):
                    print()
                    break
                time.sleep(60)
            print('=' * 30)

        # start training
        cmd = ['python', 'train_continual.py', '--device', args.device,
               '--config', f'configs/{args.dataset_name}_continual_typing{cycle_suffix}.yaml']
        if args.continual_method == 'our':
            type_checkpoints = []
            for sub_idx in range(1, idx):
                sub_split = f'p{sub_idx}'
                sub_exp_path = f'{args.dataset_name}_supervised_{exp_prefix}_fine_{total_parts}_bert_large_' \
                               f'lora4_mk00_{sub_split}{cycle_suffix}'
                type_checkpoints.append(os.path.join('checkpoints', sub_exp_path, 'models', 'tot-best.pkl'))
            if len(type_checkpoints) > 0:
                cmd += ['--type_checkpoint', ','.join(type_checkpoints)]
            if args.use_selector > 0 and idx >= 3:
                assert len(type_checkpoints) > 0
                cmd += ['--select_checkpoint', type_checkpoints[-1]]
        elif args.continual_method == 'ewc':
            if idx >= 2:
                sub_exp_path = f'{args.dataset_name}_supervised_{exp_prefix}_fine_{total_parts}_bert_large_' \
                               f'lora4_mk00_p{idx-1}{cycle_suffix}'
                cmd += ['--grad_checkpoint', os.path.join('checkpoints', sub_exp_path, 'models', 'grad_fisher.pkl')]
                cmd += ['--checkpoint', os.path.join('checkpoints', sub_exp_path, 'models', 'tot-best.pkl')]
        elif args.continual_method == 'lwf':
            if idx >= 2:
                sub_exp_path = f'{args.dataset_name}_supervised_{exp_prefix}_fine_{total_parts}_bert_large_' \
                               f'lora4_mk00_p{idx-1}{cycle_suffix}'
                cmd += ['--grad_checkpoint', os.path.join(exp_path, 'models', 'lwf_logit.pkl')]
                cmd += ['--checkpoint', os.path.join('checkpoints', sub_exp_path, 'models', 'tot-best.pkl')]
        elif args.continual_method.startswith('emr'):
            if idx >= 2:
                sub_exp_path = f'{args.dataset_name}_supervised_{exp_prefix}_fine_{total_parts}_bert_large_' \
                               f'lora4_mk00_p{idx-1}{cycle_suffix}'
                cmd += ['--checkpoint', os.path.join('checkpoints', sub_exp_path, 'models', 'tot-best.pkl')]
        else:
            raise NotImplementedError('invalid continual_method')
        print(' '.join(cmd))
        print('=' * 30)

        os.makedirs(exp_path, exist_ok=True)
        std_log = open(os.path.join(exp_path, 'std.log'), 'w', encoding='utf-8')
        err_log = open(os.path.join(exp_path, 'err.log'), 'w', encoding='utf-8')
        subp = subprocess.Popen(cmd, stdout=std_log, stderr=err_log, bufsize=1)
        subp.wait()
        assert subp.returncode == 0
        assert os.path.exists(os.path.join(exp_path, 'flag'))
        std_log.close()
        err_log.close()


if __name__ == '__main__':
    # generate_dataset_cache()
    main()
