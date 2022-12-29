import os
import torch
from utils import load_json
from torch.optim import AdamW
from loss_similarity import LossSimilarity
from models import BertForMaskedLMLoRA, BertLoRAWithSelector
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForMaskedLM


def init_tokenizer_model(config):
    model_name, model_path = config['plm']['model_name'], config['plm']['model_path']
    max_seq_len = config['dataloader']['max_seq_length']
    dataset_name = config['dataset']['dataset_name']
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    if model_name == 't5':
        tokenizer = T5Tokenizer.from_pretrained(model_path, model_max_length=max_seq_len)
        model = T5ForConditionalGeneration.from_pretrained(model_path).to(config['device'])
        model.config.max_length = max_seq_len
    elif model_name in ['bert', 'roberta']:
        tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=max_seq_len)
        tokenizer.add_special_tokens({"additional_special_tokens": [f"[unused{idx}]" for idx in range(5)]})
        assert all(len(tokenizer.tokenize(f"[unused{idx}]")) == 1 for idx in range(5)) and \
               len(tokenizer.tokenize("[unused2]"))
        if config['plm']['apply_lora'] or config['plm']['apply_adapter']:
            assert not (config['plm']['apply_lora'] and config['plm']['apply_adapter'])
            use_expert_selector = config['train']['train_expert_selector'] > 0 or config['train']['use_expert_selector']
            use_expert_selector &= config['train']['continual_method'] == 'our'
            if config['dataset']['method_type'] == 'linear':
                split_to_tags = load_json(f'scripts/{dataset_name}_class_split_'
                                          f'{config["dataset"]["total_parts"]}_tags.json')
                if config['train']['continual_method'] == 'our':
                    linear_dim = len(split_to_tags[config['dataset']['special_part']])
                    linear_dim_exp = int(config['dataset']['special_part'][1:])
                else:
                    assert config['train']['continual_method'] in ['ewc', 'lwf', 'emr', 'emr_abl']
                    num_labels = sum(len(tags) for split, tags in split_to_tags.items() if split != 'all')
                    linear_dim = num_labels
                    linear_dim_exp = -1
            else:
                linear_dim = linear_dim_exp = -1
            if use_expert_selector:
                model = BertLoRAWithSelector(config, with_selector=True, linear_dim=linear_dim,
                                             linear_dim_exp=linear_dim_exp).to(config['device'])
            else:
                model = BertLoRAWithSelector(config, with_selector=False, linear_dim=linear_dim,
                                             linear_dim_exp=linear_dim_exp).to(config['device'])
            if use_expert_selector and config['select_checkpoint']:
                assert isinstance(model, BertLoRAWithSelector)
                print('loading expert selector from:', config['select_checkpoint'])
                model_state = torch.load(config['select_checkpoint'], map_location='cpu')['model']
                err_msg = model.load_selector_state_dict(model_state, strict=False)
                assert len(err_msg.unexpected_keys) == 0 and \
                       all('lora' not in key for key in err_msg.missing_keys)
        else:
            model = AutoModelForMaskedLM.from_pretrained(model_path).to(config['device'])
            model.config.max_length = max_seq_len
    else:
        raise NotImplementedError(f'invalid model name {model_name}')
    return tokenizer, model


def init_optimizer(config, model, data_len):
    opt_config = config['plm']['optimize']
    optimizer = AdamW([param for param in model.parameters() if param.requires_grad],
                      lr=opt_config['lr'], weight_decay=opt_config['weight_decay'])
    scheduler = None
    if 'scheduler' in opt_config:
        num_training_steps = data_len * config['train']['num_epochs'] // config['train']['gradient_accumulation_steps']
        num_warmup_steps = int(opt_config['scheduler']['warmup_ratio'] * num_training_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps,
        )
    return optimizer, scheduler


def load_checkpoint(config, cp_path, model, loss_sim: LossSimilarity = None, optimizer=None, scheduler=None):
    if not cp_path:
        return 0, 0, {}
    trained_epoch, global_step = 0, 0
    if hasattr(model, 'module'):
        model_to_load = model.module
    else:
        model_to_load = model

    print(f'load model from {cp_path} ......')
    if os.path.isdir(cp_path):
        meta_path = os.path.join(cp_path, 'meta.pkl')
        if os.path.exists(meta_path):
            params = torch.load(meta_path)
        else:
            print(f'Warning: meta {meta_path} does not exist.')
            params = {}
        use_new_model = config['plm']['apply_lora'] or config['plm']['apply_adapter']
        target_class = T5ForConditionalGeneration if config['plm']['model_name'] == 't5' else \
            (BertForMaskedLMLoRA if use_new_model else AutoModelForMaskedLM)
        model_to_load.load_state_dict(target_class.from_pretrained(cp_path).state_dict())
    else:
        params = torch.load(cp_path, map_location='cpu')
        strict = not (config['plm']['apply_lora'] or config['plm']['apply_adapter'])
        model_to_load.load_state_dict(params['model'], strict=strict)

    if params == {}:
        return trained_epoch, global_step, {}

    trained_epoch = params['trained_epoch'] + 1
    # ALERT: This condition need to be checked
    if optimizer is not None and params['optimizer']['state'] and \
            not (config['task'] == 'fewshot' and config['is_test']):
        if all(group['lr'] > 0 for group in params['optimizer']['param_groups']):
            optimizer.load_state_dict(params['optimizer'])
    if optimizer is not None and not config['is_test']:
        for group in optimizer.param_groups:
            assert group['lr'] > 0
    if loss_sim is not None and 'loss_sim' in params:
        loss_sim.extra_module.load_state_dict(params['loss_sim'])
    if 'global_step' in params:
        global_step = params['global_step']
    if 'scheduler' in params and scheduler is not None and all(lr > 0 for lr in params['scheduler']['_last_lr']):
        scheduler.load_state_dict(params['scheduler'])
    if 'type_embeds' in params:
        params['type_embeds'] = [embed.to(config['device']) for embed in params['type_embeds']]
    else:
        params['type_embeds'], params['type_counter'] = None, None
    return trained_epoch, global_step, params


def get_split_by_path(config, path: str):
    tmp_path = path.replace('/', '_') + '_'
    cur_split, upper_bound = '', int(config['dataset']['total_parts'][1:])
    for idx in range(1, upper_bound):
        if f'_p{idx}_' in tmp_path:
            assert cur_split == '', path
            cur_split = f'p{idx}'
    assert cur_split != '', path
    return cur_split


def load_model_list(config, cp_path_list):
    # ALERT: multiple loss_sim is not supported!
    embed_size, label_num = config['embed_size'], config['label_num']
    if config['task'] in ['contrastive', 'continual']:
        type_embeds = torch.stack([torch.zeros(embed_size).to(config['device']) for _ in range(label_num)])
        type_counter = [0 for _ in range(label_num)]
        type_embeds.requires_grad = False
    else:
        type_embeds, type_counter = None, None
    extra_module_info = []
    if len(cp_path_list) == 0:
        print('load_model_list: no type embedding checkpoint is specified!')
        return type_embeds, type_counter, extra_module_info
    for cp_path in cp_path_list:
        params = torch.load(cp_path, map_location='cpu')
        new_state_dict = {}
        for key, val in params['model'].items():
            if key.startswith('bert_lora'):
                new_state_dict[key[10:]] = val
            elif not key.startswith('bert_selector'):
                new_state_dict[key] = val
        extra_module_info.append((get_split_by_path(config, cp_path), new_state_dict))
        if config['task'] == 'fewshot':
            continue
        cur_embed, cur_counter = params['type_embeds'], params['type_counter']
        assert len(cur_counter) == label_num
        for idx in range(label_num):
            if cur_counter[idx] > 0:
                assert type_counter[idx] == 0 and cur_embed[idx][0].item() != 0.
                type_counter[idx] = cur_counter[idx]
                type_embeds[idx] = cur_embed[idx]
    if config['task'] in ['contrastive', 'continual']:
        positive_cls_num = sum(cnt > 0 for cnt in type_counter)
        assert positive_cls_num > 0
        print(f'load_model_list: {positive_cls_num} categories have initialized embedding ......')
    return type_embeds, type_counter, extra_module_info
