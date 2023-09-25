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
from preprocess import init_type_descriptions, init_contrastive_dataloader, init_contrastive_dataset
from kernel import init_tokenizer_model, init_optimizer, train, get_embeddings, update_alignment_model, \
    test, select_continual_samples
from global_var import get_epoch_map, get_learning_rate, get_batch_size_map, get_batch_limit


global_cp_path = '../../scy_test/checkpoints'
sample_num_map = {
    'fewnerd': 100,
    'ontonotes': 100,
    'bbn': 50,
    'fewrel': 50,
    'tacred': 20,
    'ace': 20,
    'chent': 100,
}


def get_emr_replay_frequency(dataset_name: str, sid: int, memory_train_size: int, big_model: bool):
    if memory_train_size == 0:
        return 1
    training_sample_map = {
        'fewnerd': [30415, 24780, 51816, 101875, 34443, 29988, 12912, 24075, 15163, 14916],
        'ontonotes': [17932, 27842, 16288, 34221, 21156, 25248, 31235, 7818, 18138, 11645],
        'bbn': [4822, 6210, 2738, 5120, 46199, 7373, 5260, 6751, 2264, 2628],
        'fewrel': [4480, 4480, 4480, 4480, 4480, 4480, 4480, 4480, 4480, 4480],
        'tacred': [651, 341, 2313, 2509, 667, 976, 1083, 248, 1469, 2755],
        'ace': [2507, 1013, 724, 777, 627],
        'chent': [10516, 4451, 54931, 18106, 11730, 22629, 20031, 28184, 10116, 15190],
    }
    batch_size_map = get_batch_size_map(big_model)
    batch_limit_map = {
        'fewnerd': 2500,
        'ontonotes': 1250,
        'bbn': 500,
        'fewrel': 400,
        'tacred': 100,
        'ace': 100,
        'chent': 1250,
    }
    base_old_frequency_map = {
        'fewnerd': 10,
        'ontonotes': 10,
        'bbn': 5,
        'fewrel': 5,
        'tacred': 5,
        'ace': 5,
        'chent': 10,
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
                        choices=['fewnerd', 'ontonotes', 'bbn', 'fewrel', 'tacred', 'ace', 'chent'])
    parser.add_argument('--start', type=int, help='start split', default=1)
    parser.add_argument('--cycle_suffix', type=str, help='the suffix of checkpoint path')
    parser.add_argument('--method_type', type=str, choices=['prompt', 'linear', 'marker'], default='linear')
    parser.add_argument('--batch_limit_policy', type=int, help='batch limit policy', default=0)
    parser.add_argument('--batch_limit', type=int, help='batch sample limit', default=1000)
    parser.add_argument('--not_apply_lora', action='store_true')
    parser.add_argument('--apply_adapter', action='store_true')
    parser.add_argument('--global_cp_path', type=str, default='')
    parser.add_argument('--seed', type=int, help='the random seed', default=100)
    parser.add_argument('--clear', action='store_true', help='clear the history cache_flag and samples')
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

    big_model = config['plm']['model_name'] in ['llama', 'cpm']
    emar_epoch_num_map = get_epoch_map(big_model)

    # set the config entries
    use_selected = args.batch_limit_policy == 0
    config['type_checkpoint'] = config['grad_checkpoint'] = \
        config['select_checkpoint'] = config['checkpoint'] = ''
    config['is_test'] = config['generate_grad'] = \
        config['generate_logit'] = config['select_sample'] = False
    config['train']['continual_method'] = 'emr'  # to be checked
    config['plm']['apply_lora'] = not args.not_apply_lora and not args.apply_adapter
    config['plm']['apply_adapter'] = args.apply_adapter
    config['dataset']['method_type'] = args.method_type
    config['logging']['cycle_suffix'] = args.cycle_suffix
    config['logging']['path_base'] = global_cp_path
    config['train']['train_expert_selector'] = 0
    config['train']['use_expert_selector'] = False
    config['train']['num_epochs'] = emar_epoch_num_map[args.dataset_name]
    config['train']['save_step'] = -1
    config['dataset']['batch_limit_policy'] = args.batch_limit_policy
    # config['dataset']['batch_limit'] = args.batch_limit
    config['dataset']['batch_limit'] = get_batch_limit(args.dataset_name)
    config['dataset']['seed'] = config['reproduce']['seed'] = args.seed
    config['dataset']['use_selected'] = use_selected
    config['sample_num'] = sample_num_map[args.dataset_name]
    config['is_eaemr'] = True
    config['device'] = args.device

    if use_selected and args.clear:
        for idx in range(args.start, total_bound + 1):
            exp_path = f'{args.dataset_name}_supervised_{exp_prefix}_fine_{total_parts}_bert_large_' \
                       f'lora4_mk00_p{idx}{cycle_suffix}'
            wait_file = f'cache/{args.dataset_name}_continual_{total_parts}_selected_{exp_prefix}' \
                        f'_p{idx}{cycle_suffix}.json'
            cache_flag_path = os.path.join(global_cp_path, exp_path, 'cache_flag')
            if args.clear and os.path.exists(wait_file):
                os.remove(wait_file)
            if args.clear and os.path.exists(cache_flag_path):
                os.remove(cache_flag_path)

    tokenizer, model = init_tokenizer_model(config)
    init_type_descriptions(config)
    # init_contrastive_dataset
    ori_dataset = {}
    for part in ['train', 'valid', 'test']:
        ori_dataset[part] = load_json(f'data/{args.dataset_name}/{args.dataset_name}_split_{part}_'
                                      f'dataset_{total_parts}_key.json')
        if part == 'train':
            for ds_id in range(len(ori_dataset[part])):
                new_dataset = []
                for sample_key, tag, text in ori_dataset[part][ds_id]:
                    pos = text.find(' In this sentence,')
                    if len(text[pos + 1:].split(' ')) > 50:
                        continue
                    new_dataset.append((sample_key, tag, text))
                ori_dataset[part][ds_id] = new_dataset
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
    split_to_tags = GLOBAL['continual_split_to_tags']

    for idx in range(1, args.start):
        cur_split = f'p{idx}'
        exp_path = f'{args.dataset_name}_supervised_{exp_prefix}_fine_{total_parts}_bert_large_' \
                   f'lora4_mk00_{cur_split}{cycle_suffix}'
        exp_path = os.path.join(global_cp_path, exp_path)
        result_path = os.path.join(exp_path, 'sequence_results.json')
        assert os.path.exists(result_path)
        sequence_results = load_json(result_path)
        prev_sequence_results = load_json(os.path.join(exp_path, 'prev_sequence_results.json'))

        if use_selected:
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
    if args.start > 1:
        exp_path = f'{args.dataset_name}_supervised_{exp_prefix}_fine_{total_parts}_bert_large_' \
                   f'lora4_mk00_p{args.start-1}{cycle_suffix}'
        mem_embeddings = torch.load(os.path.join(global_cp_path, exp_path, 'save_embeds.pkl'))

    for idx in range(args.start, total_bound + 1):
        cur_split = f'p{idx}'
        loss_sim.curr_bound = idx
        config['plm']['optimize']['lr'] = get_learning_rate(big_model, args.dataset_name, idx, 'eaemr')
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
        if use_selected:
            combined_train_set = copy.deepcopy(cur_train_data)
            replay_frequency = get_emr_replay_frequency(args.dataset_name, idx, len(mem_train_data), big_model)
            for _ in range(replay_frequency):
                combined_train_set += mem_train_data
            datasets = {
                'train': [],
                'train_infer': combined_train_set,
                'valid_groups': cur_valid_data + mem_valid_data,
                'test_groups': accu_test_data,
            }
        else:
            datasets, _ = init_contrastive_dataset(config, type_splits=[])
        data_loaders, cur_train_sz = init_contrastive_dataloader(config, datasets, tokenizer)

        if not use_selected:
            # backup the embeddings generated by the original model
            mem_embeddings = {}
            for _ in range(config['train']['num_epochs']):
                mem_embeddings.update(get_embeddings(config, data_loaders, model, tokenizer, loss_sim, False))
                print('length of mem_embeddings:', len(mem_embeddings))
        # I. fine-tuning
        trained_epochs, global_steps = 0, 0
        global_steps, average_loss, best_step, best_step_acc, best_epoch, best_epoch_acc, best_results, test_results \
            = train(config, data_loaders, model, tokenizer, loss_sim, optimizer, None, trained_epochs, global_steps)
        trained_epochs += config['train']['num_epochs']
        print('global step:', global_steps)
        print('average loss:', average_loss)
        print('best epoch:', best_epoch, 'best epoch acc:', best_epoch_acc)
        print('fine-tuning test results:', test_results)

        # evaluation
        test_output_path = os.path.join(exp_path, 'test')
        prefix = f'{"epoch" if best_epoch_acc > best_step_acc else "step"}-best'
        p_epoch, p_step = best_epoch, best_step
        test_results = test(config, data_loaders['test_groups'], None, model, tokenizer, loss_sim, 'test',
                            test_output_path, is_step='step' in prefix, epoch=p_epoch, step=p_step)[0]
        print('prev test results:', test_results)
        prev_sequence_results.append(test_results)

        # II. sample selection
        if use_selected:
            wait_file = f'cache/{args.dataset_name}_continual_{total_parts}_selected_{exp_prefix}' \
                        f'_p{idx}{cycle_suffix}.json'
            cache_flag_path = os.path.join(global_cp_path, exp_path, 'cache_flag')
            if big_model and not os.path.exists(wait_file):
                # for big models, compute logits locally
                select_dataset = {'train_infer': cur_train_data}
                select_loader, _ = init_contrastive_dataloader(config, select_dataset, tokenizer)
                select_continual_samples(config, select_loader, model, tokenizer, loss_sim)
                with open(cache_flag_path, 'w') as fout:
                    fout.write('cache complete\n')

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

            tmp_embedding_map = get_embeddings(
                config, selected_samples['train_infer'], model, tokenizer, loss_sim, False)
            mem_embeddings.update(tmp_embedding_map)

            # IV. update alignment module
            cur_embeddings = get_embeddings(config, mem_train_data, model, tokenizer, loss_sim, True)
            assert cur_embeddings.keys() == mem_embeddings.keys()
            model.init_alignment()
            update_alignment_model(config, cur_embeddings, mem_embeddings, model.lora_alignment)

            mem_embeddings = get_embeddings(config, mem_train_data, model, tokenizer, loss_sim, False)
        else:
            # III. get sample embeddings
            cur_embeddings, key2tags = {}, {}
            for _ in range(config['train']['num_epochs']):
                tmp_embeddings, tmp_tags = get_embeddings(
                    config, data_loaders, model, tokenizer, loss_sim, True, return_tags=True)
                cur_embeddings.update(tmp_embeddings)
                key2tags.update(tmp_tags)
                print('length of cur_embeddings:', len(cur_embeddings))
            assert cur_embeddings.keys() == mem_embeddings.keys() == key2tags.keys()
            # change the sample embeddings for the new categories
            cur_tags = split_to_tags[cur_split]
            for key, tag in key2tags.items():
                if tag in cur_tags:
                    mem_embeddings[key] = cur_embeddings[key]
            # IV. update alignment module
            model.init_alignment()
            update_alignment_model(config, cur_embeddings, mem_embeddings, model.lora_alignment)

        # evaluation
        test_results = test(config, data_loaders['test_groups'], None, model, tokenizer, loss_sim, 'test',
                            test_output_path, is_step='step' in prefix, epoch=p_epoch, step=p_step)[0]

        print('final test results:', test_results)
        sequence_results.append(test_results)

        print(prev_sequence_results[-1])
        print(sequence_results[-1])
        save_json(prev_sequence_results, os.path.join(exp_path, 'prev_sequence_results.json'))
        save_json(sequence_results, os.path.join(exp_path, 'sequence_results.json'))
        torch.save(mem_embeddings, os.path.join(exp_path, 'save_embeds.pkl'))


if __name__ == '__main__':
    main()
