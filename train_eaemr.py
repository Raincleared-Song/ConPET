import os
import yaml
import copy
import math
import time
import torch
import shutil
import argparse
from global_var import GLOBAL
from loss_similarity import LossSimilarity
from utils import init_seed, load_json, save_json
from preprocess import init_type_descriptions, init_contrastive_dataloader
from kernel import init_tokenizer_model, init_optimizer, train, get_embeddings, update_alignment_model, test


global_cp_path = '../../scy_test/checkpoints'
sample_num_map = {
    'fewnerd': 100,
    'ontonotes': 100,
    'bbn': 50,
    'fewrel': 50,
    'tacred': 20,
    'ace': 20,
}
emar_epoch_num_map = {
    'fewnerd': 10,
    'ontonotes': 20,
    'bbn': 20,
    'fewrel': 10,
    'tacred': 20,
    'ace': 20,
}


def get_emr_replay_frequency(dataset_name: str, sid: int, memory_train_size: int):
    if memory_train_size == 0:
        return 1
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
    replay_frequency = room_for_old_samples / memory_train_size
    replay_frequency = int(math.ceil(replay_frequency))
    assert replay_frequency > 0
    return replay_frequency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, help='name of the dataset',
                        choices=['fewnerd', 'ontonotes', 'bbn', 'fewrel', 'tacred', 'ace'])
    parser.add_argument('--start', type=int, help='start split', default=1)
    parser.add_argument('--cycle_suffix', type=str, help='the suffix of checkpoint path')
    parser.add_argument('--method_type', type=str, choices=['prompt', 'linear', 'marker'], default='linear')
    parser.add_argument('--global_cp_path', type=str, default='')
    parser.add_argument('--seed', type=int, help='the random seed', default=100)
    parser.add_argument('--device', type=str, help='the device to be used')
    args = parser.parse_args()

    global global_cp_path
    if args.global_cp_path != '':
        global_cp_path = args.global_cp_path

    method_prefix_map = {'prompt': 'pt', 'marker': 'mk', 'linear': 'li'}
    bak_config_path = f'configs/{args.dataset_name}_continual_typing_{args.cycle_suffix}.yaml'
    if not os.path.exists(bak_config_path):
        shutil.copy(f'configs/{args.dataset_name}_continual_typing.yaml', bak_config_path)
    with open(bak_config_path, encoding='utf-8') as fin:
        config = yaml.load(fin, yaml.Loader)
    cycle_suffix = '_' + args.cycle_suffix if args.cycle_suffix != '' else ''
    exp_prefix = method_prefix_map[args.method_type]
    total_parts = config['dataset']['total_parts']
    total_bound = int(total_parts[1:])

    # set the config entries
    config['type_checkpoint'] = config['grad_checkpoint'] = \
        config['select_checkpoint'] = config['checkpoint'] = ''
    config['is_test'] = config['generate_grad'] = \
        config['generate_logit'] = config['select_sample'] = False
    config['train']['continual_method'] = 'emr'  # to be checked
    config['dataset']['method_type'] = args.method_type
    config['logging']['cycle_suffix'] = args.cycle_suffix
    config['logging']['path_base'] = global_cp_path
    config['train']['train_expert_selector'] = 0
    config['train']['use_expert_selector'] = False
    config['train']['num_epochs'] = emar_epoch_num_map[args.dataset_name]
    config['train']['save_step'] = -1
    config['dataset']['batch_limit_policy'] = 0
    config['dataset']['seed'] = config['reproduce']['seed'] = args.seed
    config['dataset']['use_selected'] = True
    config['sample_num'] = sample_num_map[args.dataset_name]
    config['is_eaemr'] = True
    config['device'] = args.device

    tokenizer, model = init_tokenizer_model(config)
    init_type_descriptions(config)
    # init_contrastive_dataset
    ori_dataset = {}
    for part in ['train', 'valid', 'test']:
        ori_dataset[part] = load_json(f'data/{args.dataset_name}/{args.dataset_name}_split_{part}_'
                                      f'dataset_{total_parts}_mask.json')
        new_dataset = []
        for sid, data in enumerate(ori_dataset[part]):
            new_dataset.append([(f'{part}_{sid}_{idx}', sample) for idx, sample in enumerate(data)])
        ori_dataset[part] = new_dataset
    label_num = len(GLOBAL['continual_tag_to_split'])
    config['label_num'], config['vocab_size'], config['hidden_size'] = \
        label_num, len(tokenizer), model.config.hidden_size
    loss_sim = LossSimilarity(config)
    config['embed_size'] = loss_sim.embed_size
    assert 'scheduler' not in config
    optimizer, _ = init_optimizer(config, model, -1)

    sequence_results, prev_sequence_results = [], []
    mem_train_data, mem_valid_data = [], []
    accu_test_data = []
    mem_embeddings = {}

    for idx in range(1, args.start):
        cur_split = f'p{idx}'
        exp_path = f'{args.dataset_name}_supervised_{exp_prefix}_fine_{total_parts}_bert_large_' \
                   f'lora4_mk00_{cur_split}{cycle_suffix}'
        exp_path = os.path.join(global_cp_path, exp_path)
        result_path = os.path.join(exp_path, 'sequence_results.json')
        assert os.path.exists(result_path)
        sequence_results = load_json(result_path)
        prev_sequence_results = load_json(os.path.join(exp_path, 'prev_sequence_results.json'))

        wait_file = f'cache/{args.dataset_name}_continual_{total_parts}_selected_{exp_prefix}' \
                    f'_p{idx}{cycle_suffix}.json'
        selected_samples = load_json(wait_file)
        # selected_samples = select_continual_samples(config, data_loaders, model, tokenizer, loss_sim)
        mem_train_data += selected_samples['train_infer']
        mem_valid_data += selected_samples['valid_groups']
        accu_test_data += ori_dataset['test'][idx - 1]

        model_path = os.path.join(exp_path, 'models', 'tot-best.pkl')
        params = torch.load(model_path, map_location='cpu')['model']
        if any('lora_alignment' in name for name in params.keys()):
            model.init_alignment()
        model.load_state_dict(params, strict=False)

    for idx in range(args.start, total_bound + 1):
        cur_split = f'p{idx}'
        loss_sim.curr_bound = idx
        config['dataset']['special_part'] = cur_split
        config['dataset']['extra_special_part'] = ','.join([f'p{sid}' for sid in range(1, idx)])
        exp_path = f'{args.dataset_name}_supervised_{exp_prefix}_fine_{total_parts}_bert_large_' \
                   f'lora4_mk00_{cur_split}{cycle_suffix}'
        config['logging']['unique_string'] = exp_path
        exp_path = os.path.join(global_cp_path, exp_path)
        if os.path.exists(os.path.join(exp_path, 'flag')):
            os.remove(os.path.join(exp_path, 'flag'))
        with open(f'configs/{args.dataset_name}_continual_typing{cycle_suffix}.yaml', 'w', encoding='utf-8') as fout:
            yaml.dump(config, fout)
        print(cur_split, config)
        print('=' * 30)

        # load_save_config
        os.makedirs(exp_path, exist_ok=True)
        target_config_path = os.path.join(exp_path, 'ori_config.yaml')
        with open(target_config_path, 'w', encoding='utf-8') as fout:
            yaml.dump(config, fout)
        init_seed(config)

        # prepare_dataloader
        accu_test_data += ori_dataset['test'][idx-1]
        cur_train_data, cur_valid_data = ori_dataset['train'][idx-1], ori_dataset['valid'][idx-1]
        combined_train_set = copy.deepcopy(cur_train_data)
        replay_frequency = get_emr_replay_frequency(args.dataset_name, idx, len(mem_train_data))
        for _ in range(replay_frequency):
            combined_train_set += mem_train_data
        datasets = {
            'train': [],
            'train_infer': combined_train_set,
            'valid_groups': cur_valid_data + mem_valid_data,
            'test_groups': accu_test_data,
        }
        data_loaders, cur_train_sz = init_contrastive_dataloader(config, datasets, tokenizer)

        # I. fine-tuning
        trained_epochs, global_steps = 0, 0
        global_steps, average_loss, best_step, best_step_acc, best_epoch, best_epoch_acc, best_results, test_results \
            = train(config, data_loaders, model, tokenizer, loss_sim, optimizer, None, trained_epochs, global_steps)
        trained_epochs += config['train']['num_epochs']
        print('global step:', global_steps)
        print('average loss:', average_loss)
        print('best epoch:', best_epoch, 'best epoch acc:', best_epoch_acc)
        print('fine-tuning test results:', test_results)

        # II. sample selection
        wait_file = f'cache/{args.dataset_name}_continual_{total_parts}_selected_{exp_prefix}' \
                    f'_p{idx}{cycle_suffix}.json'
        print('waiting for ', wait_file, '.' * 30)
        while True:
            if os.path.exists(wait_file):
                print()
                break
            time.sleep(60)
        print('=' * 30)

        # III. get sample embeddings
        selected_samples = load_json(wait_file)
        mem_train_data += selected_samples['train_infer']
        mem_valid_data += selected_samples['valid_groups']
        loss_sim.continual_logit_cache.clear()

        tmp_embedding_map = get_embeddings(config, selected_samples['train_infer'], model, tokenizer, False)
        mem_embeddings.update(tmp_embedding_map)

        # evaluation
        test_output_path = os.path.join(exp_path, 'test')
        prefix = f'{"epoch" if best_epoch_acc > best_step_acc else "step"}-best'
        p_epoch, p_step = best_epoch, best_step
        test_results = test(config, data_loaders['test_groups'], None, model, tokenizer, loss_sim, 'test',
                            test_output_path, is_step='step' in prefix, epoch=p_epoch, step=p_step)[0]
        print('prev test results:', test_results)
        prev_sequence_results.append(test_results)

        # III. update alignment module
        cur_embeddings = get_embeddings(config, mem_train_data, model, tokenizer, True)
        assert cur_embeddings.keys() == mem_embeddings.keys()
        model.init_alignment()
        update_alignment_model(config, cur_embeddings, mem_embeddings, model.lora_alignment)

        mem_embeddings = get_embeddings(config, mem_train_data, model, tokenizer, False)

        """
        test_by_best(config, model, tokenizer, loss_sim,
                                best_epoch, best_epoch_acc, best_step, best_step_acc,
                                test_groups1, test_infer=None, train_infer=train_infer1,
                                best_model=best_results[best_key],
                                extra_module_info=None, prototypes=cur_prototypes)
        """
        # evaluation
        test_results = test(config, data_loaders['test_groups'], None, model, tokenizer, loss_sim, 'test',
                            test_output_path, is_step='step' in prefix, epoch=p_epoch, step=p_step)[0]

        print('final test results:', test_results)
        sequence_results.append(test_results)

        print(prev_sequence_results[-1])
        print(sequence_results[-1])
        save_json(prev_sequence_results, os.path.join(exp_path, 'prev_sequence_results.json'))
        save_json(sequence_results, os.path.join(exp_path, 'sequence_results.json'))


if __name__ == '__main__':
    main()
