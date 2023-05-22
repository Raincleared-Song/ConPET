import os
import yaml
import json
import argparse


def load_json(path: str):
    file = open(path)
    res = json.load(file)
    file.close()
    return res


def read_lines(path: str):
    with open(path, encoding='utf-8') as fin:
        lines = fin.readlines()
    lines = [int(line.strip()) for line in lines]
    return lines


def cut_lines(path: str, count: int, mode='tail'):
    lines = read_lines(path)
    if mode == 'head':
        lines = lines[:count]
    else:
        lines = lines[(len(lines)-count):]
    with open(path, 'w', encoding='utf-8') as fout:
        fout.writelines([str(line) + '\n' for line in lines])


def load_config(args):
    if not hasattr(load_json, 'task_to_config'):
        load_config.task_to_config = {}
    if args.task not in load_config.task_to_config:
        ori_config_path = os.path.join(args.prefix, 'checkpoints', args.task, 'ori_config.yaml')
        assert os.path.exists(ori_config_path)
        with open(ori_config_path, encoding='utf-8') as fin:
            load_config.task_to_config[args.task] = yaml.load(fin, Loader=yaml.Loader)
    return load_config.task_to_config[args.task]


def read_answers_from_txt(path: str):
    with open(path, encoding='utf-8') as fin:
        lines = [line.strip() for line in fin.readlines()]
    answers = []
    for line in lines:
        tokens = line.split('\t')
        answers.append((int(tokens[0]), int(tokens[1])))
    return answers


def check_average_score(args, path: str):
    answers = read_answers_from_txt(path)
    class_split_tags = load_json(f'scripts/{args.dataset_name}_class_split_{args.total_parts}_tags.json')
    assert args.split != ''
    current_splits = args.split.split(',')
    consider_types = set()
    for split in current_splits:
        consider_types |= set(class_split_tags[split])
    counter = {f'p{idx}': [0, 0] for _ in range(len(current_splits)) for idx in range(1, int(args.total_parts[1:])+1)}
    for pred, label in answers:
        if label not in consider_types:
            continue
        if not args.selector:
            split = ''
            for split in current_splits:
                if label in class_split_tags[split]:
                    break
        else:
            split = f'p{label + 1}'
        assert split != ''
        counter[split][0] += int(pred == label)
        counter[split][1] += 1
    avg_accuracies, non_zero = [], 0
    for key, val in counter.items():
        loc_accuracy = round(val[0] * 100 / val[1] if val[1] > 0 else 0, 2)
        print(key, val, loc_accuracy)
        avg_accuracies.append(loc_accuracy)
        non_zero += int(val[1] > 0)
    return round(sum(avg_accuracies) / non_zero, 2)


def check_valid_results(args):
    valid_path = os.path.join(args.prefix, 'checkpoints', args.task, 'valid')
    metric_files = [os.path.join(valid_path, f) for f in os.listdir(valid_path) if f.endswith('.json')]
    if args.selector:
        metric_files = [f for f in metric_files if 'exp' in f]
    else:
        metric_files = [f for f in metric_files if 'exp' not in f]
    metric_files.sort(key=lambda f: os.path.getmtime(f))
    side_metrics = [s for s in args.side_metrics.split(',') if s]
    max_metric, max_f = 0, ''
    for f in metric_files:
        metrics = load_json(f)
        cur_metric = metrics[args.metric]
        if not args.not_print_every:
            print(cur_metric, [metrics[m] for m in side_metrics if m in metrics], f)
        if cur_metric > max_metric:
            max_metric = cur_metric
            max_f = f
    print('maximum:', max_metric, max_f)
    # check average_score
    max_f = max_f.replace('metrics_', 'results_')
    max_f = max_f.replace('.json', '.txt')
    avg_acc = check_average_score(args, max_f)
    print('average accuracy:', avg_acc)


def check_test_results(args):
    test_path = os.path.join(args.prefix, 'checkpoints', args.task, 'test')
    test_file = os.path.join(test_path, 'metrics_test.json')
    test_metrics = load_json(test_file)
    print(test_metrics)
    test_file = os.path.join(test_path, 'results_test.txt')
    answers = read_answers_from_txt(test_file)
    class_split_tags = load_json(f'scripts/{args.dataset_name}_class_split_{args.total_parts}_tags.json')
    assert args.split != ''
    consider_types = set()
    for split in args.split.split(','):
        consider_types |= set(class_split_tags[split])
    print(f'considering {len(consider_types)} types ......')
    correct_num, instance_num = 0, 0
    for pred, label in answers:
        if label not in consider_types:
            continue
        correct_num += int(pred == label)
        instance_num += 1
    print({'correct_num': correct_num, 'instance_num': instance_num,
           'accuracy': round(correct_num * 100 / instance_num, 2)})
    # check average_score
    avg_acc = check_average_score(args, test_file)
    print('average accuracy:', avg_acc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '-t', type=str, help='task name', required=True)
    parser.add_argument('--type', type=str, help='task type',
                        choices=['predefined', 'choice', 'auto'], default='auto')
    parser.add_argument('--label_path', type=str, help='path of the label file',
                        default='scripts/choices.json')
    parser.add_argument('--choice_num', type=int, help='choice number, -1 for auto', default=-1)
    parser.add_argument('--dataset_name', type=str, help='dataset name', default='',
                        choices=['fewnerd', 'ontonotes', 'bbn', 'fewrel', 'tacred', 'ace', 'chent'])
    parser.add_argument('--soft_prompt', type=int, help='soft prompt, -1 for auto', default=-1)
    parser.add_argument('--prefix', '-p', type=str, help='prefix of the "checkpoints"', default='.')
    parser.add_argument('--metric', type=str, help='the metric to be checked', default='accuracy')
    parser.add_argument('--side_metrics', '-sms', type=str,
                        help='other metrics to be printed', default='accuracy_expert')
    parser.add_argument('--mode', '-m', choices=['valid', 'test'], type=str, default='valid')
    parser.add_argument('--total_parts', '-tp', choices=['p5', 'p10', ''], type=str, default='')
    parser.add_argument('--selector', '-sel', action='store_true')
    parser.add_argument('--split', '-s', help='check which part of categories', type=str, default='')
    parser.add_argument('--not_print_every', help='do not print all results', action='store_true')
    args = parser.parse_args()

    if args.dataset_name == '':
        for name in ['fewnerd', 'ontonotes', 'bbn', 'fewrel', 'tacred', 'ace', 'chent']:
            if name in args.task:
                args.dataset_name = name
                break
        assert args.dataset_name != '', args.task
    if args.total_parts == '':
        args.total_parts = 'p5' if args.dataset_name == 'ace' else 'p10'

    if args.split == '':
        cur_tot_idx = int(args.total_parts[1:])
        args.split = ','.join([f'p{idx}' for idx in range(1, cur_tot_idx+1)])

    if args.mode == 'valid':
        check_valid_results(args)
        return
    if args.mode == 'test':
        check_test_results(args)
        return

    if args.type != 'auto':
        task_type = args.type
    else:
        task_type = load_config(args)['dataset']['type']

    test_labels = read_lines(os.path.join('prefix', 'checkpoints', args.task, 'test_labels.txt'))
    test_preds = read_lines(os.path.join('prefix', 'checkpoints', args.task, 'test_preds.txt'))
    assert len(test_labels) == len(test_preds)
    if task_type == 'predefined':
        from sklearn.metrics import accuracy_score
        from openprompt.utils.metrics import loose_micro, loose_macro
        labels = load_json(args.label_path)
        id2label = {i: k for (i, k) in enumerate(labels)}
        print('accuracy:', round(accuracy_score(test_labels, test_preds) * 100, 2))
        print('loose-micro:', round(loose_micro(test_labels, test_preds,
                                                id2label=id2label, label_path_sep='-')['f1'] * 100, 2))
        print('loose-macro:', round(loose_macro(test_labels, test_preds,
                                                id2label=id2label, label_path_sep='-')['f1'] * 100, 2))
    elif task_type == 'choice':
        import torch
        from sklearn.metrics import accuracy_score
        if args.choice_num != -1:
            choice_num = args.choice_num
        else:
            choice_num = load_config(args)['dataset']['choice_num']
        if args.soft_prompt != -1:
            soft_prompt = bool(args.soft_prompt)
        else:
            soft_prompt = load_config(args)['dataset']['soft_prompt']
        print('choice accuracy:', round(accuracy_score(test_labels, test_preds) * 100, 2))
        soft_prompt_map = load_json(os.path.join('prefix', 'checkpoints', args.task, 'soft_prompt_map.json'))
        true_label_num = len(soft_prompt_map)
        cache_path = os.path.join('cache', f'{args.dataset_name}_supervised_test_'
                                           f'choice{choice_num:02}{"_soft" if soft_prompt else ""}.pth')
        assert os.path.exists(cache_path), cache_path
        examples = torch.load(cache_path)
        per_sample = (true_label_num + choice_num - 1) // choice_num
        assert len(examples) == len(test_labels), f'{len(examples)} | {len(test_labels)}'
        assert len(examples) % per_sample == 0, f'{len(examples)} | {per_sample}'
        true_sample_num = len(examples) // per_sample
        print('true tested sample number:', true_sample_num)
        correct_num = 0
        for idx in range(true_sample_num):
            cur_preds = test_preds[idx*per_sample:(idx+1)*per_sample]
            cur_labels = test_labels[idx*per_sample:(idx+1)*per_sample]
            correct_num += int(cur_preds == cur_labels)
        print('correct number:', correct_num)
        print('accuracy:', round(correct_num / true_sample_num * 100, 2))
    else:
        raise NotImplementedError(task_type)


if __name__ == '__main__':
    main()
