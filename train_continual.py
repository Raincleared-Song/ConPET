import os
import argparse
from loss_similarity import LossSimilarity
from utils import load_save_config, init_seed
from kernel import init_tokenizer_model, init_optimizer, load_checkpoint, model_list_test, train, load_model_list, \
    test, get_split_by_path, select_continual_samples
from preprocess import init_contrastive_dataset, init_contrastive_dataloader, init_type_descriptions

import warnings
import logging

warnings.filterwarnings(action='once')
logging.getLogger("pytorch_pretrained_bert.tokenization").setLevel(logging.ERROR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help='config path', type=str, default='configs/fewrel_continual_typing.yaml')
    parser.add_argument('--checkpoint', '-ch', help='checkpoint path', type=str, default='')
    parser.add_argument('--type_checkpoint', '-tch', help='checkpoint paths with type embeds', type=str, default='')
    parser.add_argument('--grad_checkpoint', '-gch', help='path of grad_mean and grad_fisher', type=str, default='')
    parser.add_argument('--select_checkpoint', '-sch', help='path of expert selector', type=str, default='')
    parser.add_argument('--is_test', '--test', '-t', help='do test', action='store_true')
    parser.add_argument('--generate_grad', help='if generate grad and fisher', action='store_true')
    parser.add_argument('--generate_logit', help='if generate logit for old class', action='store_true')
    parser.add_argument('--select_sample', help='if select reserved samples', action='store_true')
    parser.add_argument('--sample_num', help='the number of reserved samples per class', type=int, default=50)
    parser.add_argument('--device', '-d', help='gpu device', type=str, default='cuda:0')
    parser.add_argument('--infer_device', '-id', help='gpu device for inference', type=str, default='cuda:1')
    args = parser.parse_args()

    config: dict = load_save_config(args)

    init_seed(config)
    assert not (args.type_checkpoint != '' and args.grad_checkpoint != '')
    checkpoints = [s for s in args.type_checkpoint.split(',') if s]
    type_splits = [get_split_by_path(config, path) for path in checkpoints]
    assert not (config['train']['continual_method'] == 'ewc' and config['dataset']['extra_special_part'] != '')
    assert config['select_sample'] or not (config['dataset']['special_part'] not in ['p1', 'p2']
                                           and config['train']['use_expert_selector'] and args.select_checkpoint == '')
    # init tokenizer and model
    tokenizer, model = init_tokenizer_model(config)
    init_type_descriptions(config)
    datasets, label_num = init_contrastive_dataset(config, type_splits)
    config['label_num'], config['vocab_size'] = label_num, len(tokenizer)
    config['hidden_size'] = model.config.hidden_size if config['plm']['model_name'] != 'cpm' else model.config.dim_model
    loss_sim = LossSimilarity(config)
    config['embed_size'] = loss_sim.embed_size
    data_loaders, train_sz = init_contrastive_dataloader(config, datasets, tokenizer)
    optimizer, scheduler = init_optimizer(config, model, train_sz)
    if args.select_sample:
        assert args.checkpoint != ''
        trained_epoch, global_step, extra_cp = \
            load_checkpoint(config, args.checkpoint, model, loss_sim, optimizer, scheduler)
        print(f'selecting samples with epoch {trained_epoch} step {global_step}')
        select_continual_samples(config, data_loaders, model, tokenizer, loss_sim)
        results = f'select success: {args.sample_num} per class'
    elif args.is_test:
        test_output_path = os.path.join(config['logging']['path_base'], config['logging']['unique_string'], 'test')
        if config['task'] != 'fewshot' and args.type_checkpoint != '':
            assert args.type_checkpoint != '' and len(checkpoints) == 3, 'test mode must have 3 checkpoints'
            type_embeds, type_counter, extra_module_info = load_model_list(config, checkpoints)
            extra_module_states = [info[-1] for info in extra_module_info]
            assert type_splits == [info[0] for info in extra_module_info]
            assert [f'p{idx+1}' in checkpoints[idx] for idx in range(3)], args.type_checkpoint
            results, _, _ = model_list_test(
                config, data_loaders['test_groups'], model, extra_module_states, tokenizer, loss_sim, test_output_path,
                type_embeds=type_embeds, type_counter=type_counter)
        else:
            _, _, params = load_checkpoint(config, args.checkpoint, model, loss_sim, optimizer, scheduler)
            type_embeds, type_counter, extra_module_info = load_model_list(config, checkpoints)
            results, _, _ = test(config, data_loaders['test_groups'], None, model, tokenizer,
                                 loss_sim, 'test', test_output_path,
                                 type_embeds=params['type_embeds'], type_counter=params['type_counter'],
                                 extra_module_info=extra_module_info)
    else:
        not_to_load = config['train']['continual_method'] in ['ewc', 'lwf', 'emr', 'emr_abl'] \
                  and config['logging']['unique_string'] not in args.checkpoint
        if not_to_load:
            trained_epoch, global_step, extra_cp = load_checkpoint(config, args.checkpoint, model, loss_sim)
        else:
            trained_epoch, global_step, extra_cp = \
                load_checkpoint(config, args.checkpoint, model, loss_sim, optimizer, scheduler)
        type_embeds, type_counter, extra_module_info = load_model_list(config, checkpoints)
        assert type_splits == [info[0] for info in extra_module_info]
        best_results = extra_cp['best_results'] if 'best_results' in extra_cp else None
        if not_to_load:
            trained_epoch, global_step, best_results = 0, 0, None
        global_step, average_loss, best_step, best_step_acc, best_epoch, best_epoch_acc, valid_results, results = \
            train(config, data_loaders, model, tokenizer, loss_sim, optimizer, scheduler, trained_epoch, global_step,
                  best_results=best_results, type_embeds=type_embeds, type_counter=type_counter,
                  extra_module_info=extra_module_info)
        print('global step:', global_step)
        print('average loss:', average_loss)
        print('best step:', best_step, 'best step acc:', best_step_acc)
        print('best epoch:', best_epoch, 'best epoch acc:', best_epoch_acc)
    print('test results:', results)


if __name__ == '__main__':
    main()
