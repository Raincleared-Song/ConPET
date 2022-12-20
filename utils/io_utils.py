import os
import json
import yaml
import numpy as np


def load_json(path: str):
    print(f'loading file {path} ......')
    file = open(path)
    res = json.load(file)
    file.close()
    return res


def save_json(obj: object, path: str):
    print(f'saving file {path} ......')
    file = open(path, 'w')
    json.dump(obj, file)
    file.close()


def load_fewnerd_data(path: str):
    file = open(path, 'r', encoding='utf-8')
    data = []
    xs = []
    ys = []
    spans = []

    for line in file.readlines():
        pair = line.split()
        if not pair:
            if xs:
                data.append((xs, ys, spans))
            xs = []
            ys = []
            spans = []
        else:
            xs.append(pair[0])

            tag = pair[-1]
            if tag != 'O':
                if len(ys) == 0 or tag != ys[-1][2:]:
                    tag = 'B-' + tag
                    spans.append([len(ys), len(ys)])
                else:
                    tag = 'I-' + tag
                    spans[-1][-1] = len(ys)
            ys.append(tag)
    file.close()
    return data


def print_json(obj, file=None):
    print(json.dumps(obj, sort_keys=True, indent=4, separators=(', ', ': '), ensure_ascii=False), file=file)


def print_value(epoch, mode, step, time, loss, info, path: str, end='\n'):
    s = str(epoch) + " "
    while len(s) < 7:
        s += " "
    s += str(mode) + " "
    while len(s) < 14:
        s += " "
    s += str(step) + " "
    while len(s) < 25:
        s += " "
    s += str(time)
    while len(s) < 40:
        s += " "
    s += str(loss)
    while len(s) < 48:
        s += " "
    s += str(info)
    print(s, end=end)
    file = open(path, 'a')
    print(s, file=file)
    file.close()


def time_to_str(time):
    time = int(time)
    minute = time // 60
    second = time % 60
    return '%2d:%02d' % (minute, second)


def load_save_config(args):
    config_path, is_test = args.config, args.is_test
    with open(config_path, encoding='utf-8') as fin:
        config = yaml.load(fin, yaml.Loader)
    exp_path = os.path.join(config['logging']['path_base'], config['logging']['unique_string'])
    if not config['logging']['overwrite'] and not is_test and os.path.exists(exp_path):
        raise RuntimeError(f'{exp_path} cannot be overwritten')
    os.makedirs(exp_path, exist_ok=True)
    target_config_path = f'{exp_path}/{"test" if is_test else "ori"}_config.yaml'
    if not (os.path.exists(target_config_path) and os.path.samefile(config_path, target_config_path)):
        assert os.system(f'cp {config_path} {target_config_path}') == 0
    config.update(vars(args))
    target_config_path = f'{exp_path}/args_{"test" if is_test else "ori"}_config.yaml'
    with open(target_config_path, 'w', encoding='utf-8') as fout:
        yaml.dump(config, fout)
    return config


def complex_sample(samples, size=None, replace=True, p=None):
    sampled_idx = np.random.choice(len(samples), size=size, replace=replace, p=p)
    if size is None:
        return samples[sampled_idx]
    res = []
    for idx in sampled_idx:
        res.append(samples[idx])
    return res
