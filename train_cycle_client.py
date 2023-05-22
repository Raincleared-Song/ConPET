import os
import time
import yaml
import argparse
import subprocess


global_cp_path = 'scy_test/checkpoints'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, help='name of the dataset', default='fewrel',
                        choices=['fewnerd', 'ontonotes', 'bbn', 'fewrel', 'tacred', 'ace', 'chent'])
    parser.add_argument('--method_type', type=str, choices=['prompt', 'linear', 'marker'])
    parser.add_argument('--start', type=int, help='start split', default=1)
    parser.add_argument('--cycle_suffix', type=str, help='the suffix of checkpoint path')
    parser.add_argument('--server_name', type=str, default='a100node4')
    parser.add_argument('--sample_num', type=int, default=-1)
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--global_cp_path', type=str, default='')
    parser.add_argument('--device', type=str, help='the device to be used')
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
    if 'emar' in cycle_suffix or 'eaemr' in cycle_suffix:
        total_bound += 1
    for idx in range(args.start, total_bound):
        print('=' * 30)
        cur_split = f'p{idx}'
        exp_path = f'{args.dataset_name}_supervised_{exp_prefix}_fine_{total_parts}_bert_large_' \
                   f'lora4_mk00_{cur_split}{cycle_suffix}'
        if args.local:
            print('waiting for', exp_path)
            while True:
                flag_path = f'/raid/wangyx/continual_re_typer/checkpoints/{exp_path}/flag'
                if os.path.exists(flag_path):
                    print()
                    break
                time.sleep(300)
        else:
            scp_cmd = ['scp', f'{args.server_name}:{disk_prefix}/'
                              f'private/{global_cp_path}/{exp_path}/flag', './']
            print(' '.join(scp_cmd))
            print('waiting for', exp_path)
            while True:
                subp = subprocess.Popen(scp_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                subp.wait()
                if subp.returncode == 0:
                    print()
                    break
                time.sleep(300)

        if not args.local:
            scp_cmd = ['scp', '-r', f'{args.server_name}:{disk_prefix}/private/'
                                    f'{global_cp_path}/{exp_path}', './checkpoints/']
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
        with open(config_path, 'w', encoding='utf-8') as fout:
            yaml.dump(config_base, fout)

        # sampling process
        cmd = ['python', 'train_continual.py', '--select_sample', '--sample_num', str(args.sample_num),
               '--config', f'checkpoints/{exp_path}/ori_config.yaml', '--checkpoint',
               f'checkpoints/{exp_path}/models/tot-best.pkl', '--device', args.device]
        print(' '.join(cmd))
        exp_path = os.path.join('checkpoints', exp_path)
        assert os.path.exists(os.path.join(exp_path, 'flag'))
        std_log = open(os.path.join(exp_path, 'std.log'), 'w', encoding='utf-8')
        err_log = open(os.path.join(exp_path, 'err.log'), 'w', encoding='utf-8')
        subp = subprocess.Popen(cmd, stdout=std_log, stderr=err_log, bufsize=1)
        subp.wait()
        assert subp.returncode == 0
        std_log.close()
        err_log.close()

        sample_file = f'cache/{args.dataset_name}_continual_{total_parts}_selected_{exp_prefix}' \
                      f'_{cur_split}{cycle_suffix}.json'
        if not args.local:
            scp_cmd = ['scp', sample_file, f'{args.server_name}:{disk_prefix}/'
                                           f'private/songchenyang/continual_re_typer/cache/']
            print(' '.join(scp_cmd))
            subp = subprocess.Popen(scp_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subp.wait()
            assert subp.returncode == 0
        print('=' * 30)


if __name__ == '__main__':
    main()
