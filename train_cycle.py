import os
import yaml
import shutil
import argparse
import subprocess
from utils import load_json
from matplotlib import pyplot as plt


def plot_metrics(cycle_suffix: str, exp_prefix: str, plot_metric: str, start: int):
    labels = [s for s in cycle_suffix.split(',') if s]
    assert len(labels) > 0
    colors = ['red', 'blue', 'orange', 'yellow', 'green', 'cyan']
    assert len(colors) >= len(labels)
    for color, label in zip(colors, labels):
        cur_acc = []
        for idx in range(start, 11):
            cur_split = f'p{idx}'
            exp_path = f'tacred_supervised_{exp_prefix}_fine_p10_bert_large_' \
                       f'lora4_mk00_{cur_split}{label}'
            test_metric = os.path.join('checkpoints', exp_path, 'test', 'metrics_test.json')
            if not os.path.exists(test_metric):
                continue
            test_metric = load_json(test_metric)[plot_metric]
            cur_acc.append(test_metric)
        print(label, len(cur_acc), cur_acc)
        plt.plot(range(start, len(cur_acc)+start), cur_acc, label=label, color=color, marker='*')
    plt.legend()
    plt.xticks(range(start, 11))
    plt.title(plot_metric)
    plt.tight_layout()
    plt.savefig(f'checkpoints/cycle_plot_{plot_metric}.png')


def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--loss_adverse_step', type=int, default=-1)
    parser.add_argument('--seed', type=int, help='the random seed', default=100)
    parser.add_argument('--plot', action='store_true', help='to plot the test accuracies')
    parser.add_argument('--plot_metric', type=str, help='the test metric to plot', default='accuracy')
    parser.add_argument('--device', type=str, help='the device to be used')
    args = parser.parse_args()

    method_prefix_map = {'prompt': 'pt', 'marker': 'mk', 'linear': 'li'}
    if args.plot:
        plot_metrics(args.cycle_suffix, method_prefix_map[args.method_type], args.plot_metric, args.start)
        return

    bak_config_path = f'configs/tacred_continual_typing_{args.cycle_suffix}.yaml'
    if not os.path.exists(bak_config_path):
        shutil.copy('configs/tacred_continual_typing.yaml', bak_config_path)
    with open(bak_config_path, encoding='utf-8') as fin:
        config_base = yaml.load(fin, yaml.Loader)
    cycle_suffix = '_' + args.cycle_suffix if args.cycle_suffix != '' else ''
    exp_prefix = method_prefix_map[args.method_type]
    for idx in range(args.start, 11):
        cur_split = f'p{idx}'
        config_base['dataset']['method_type'] = args.method_type
        config_base['dataset']['special_part'] = cur_split
        exp_path = f'tacred_supervised_{exp_prefix}_fine_p10_bert_large_' \
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
        with open(f'configs/tacred_continual_typing{cycle_suffix}.yaml', 'w', encoding='utf-8') as fout:
            yaml.dump(config_base, fout)
        print(cur_split, config_base)
        print('=' * 30)

        assert not config_base['dataset']['use_selected']

        # start training
        cmd = ['python', 'train_continual.py', '--device', args.device,
               '--config', f'configs/tacred_continual_typing{cycle_suffix}.yaml']
        type_checkpoints = []
        for sub_idx in range(1, idx):
            sub_split = f'p{sub_idx}'
            sub_exp_path = f'tacred_supervised_{exp_prefix}_fine_p10_bert_large_' \
                           f'lora4_mk00_{sub_split}{cycle_suffix}'
            type_checkpoints.append(os.path.join('checkpoints', sub_exp_path, 'models', 'tot-best.pkl'))
        if len(type_checkpoints) > 0:
            cmd += ['--type_checkpoint', ','.join(type_checkpoints)]
        if args.use_selector > 0 and idx >= 3:
            assert len(type_checkpoints) > 0
            cmd += ['--select_checkpoint', type_checkpoints[-1]]
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