import torch
import torch.nn as nn
from bmt_models import LlamaConfigLoRA, LlamaForCausalLMLoRA
from transformers import LlamaConfig, LlamaForCausalLM


class LlamaLoRAWithSelector(nn.Module):
    def __init__(self, config, is_infer: bool, with_selector: bool,
                 linear_dim: int, linear_dim_exp: int, model_state: dict = None):
        super().__init__()
        self.extra_config = config
        self.is_infer = is_infer
        self.with_selector = with_selector

        # initialize model weights and config
        max_seq_len, model_path = config['dataloader']['max_seq_length'], config['plm']['model_path']
        model_config = LlamaConfig.from_pretrained(model_path)
        model_config.max_length = max_seq_len
        config_dict = model_config.to_dict()
        config_dict['apply_lora'], config_dict['lora_r'] = config['plm']['apply_lora'], config['plm']['lora_r']
        config_dict['apply_adapter'], config_dict['adapter_type'], config_dict['adapter_size'] = \
            config['plm']['apply_adapter'], config['plm']['adapter_type'], config['plm']['adapter_size']
        self.config = LlamaConfigLoRA.from_dict(config_dict)

        # set alignment and the linear dim
        self.linear_dim = linear_dim
        self.linear_dim_exp = linear_dim_exp if with_selector and linear_dim_exp > 0 else -1
        self.add_alignment = config['train']['continual_method'] == 'eaemr' or \
            'is_eaemr' in config and config['is_eaemr']

        self.backbone = LlamaForCausalLMLoRA(self.config)
        if model_state is None:
            model = LlamaForCausalLM.from_pretrained(model_path)
            model_state = model.state_dict()
            del model
        err_msg = self.backbone.load_state_dict(model_state, strict=False)
        assert len(err_msg.unexpected_keys) == 0 and all('lora' in key or 'adapter' in key for key in err_msg.missing_keys)

        if self.linear_dim > 0:
            self.lora_projector = nn.Linear(self.config.hidden_size, self.linear_dim, dtype=torch.half)
        if self.linear_dim_exp > 0:
            self.lora_exp_projector = nn.Linear(self.config.hidden_size, self.linear_dim_exp, dtype=torch.half)
        if self.add_alignment:
            self.lora_alignment = nn.Linear(self.config.hidden_size, self.config.hidden_size, dtype=torch.half)
        if self.is_infer:
            self.split_to_lora_linear = {}

        self.cur_loaded_split = ''

    def init_alignment(self):
        assert hasattr(self, 'lora_alignment'), 'alignment config error'

    def load_selector_state_dict(self, state_dict):
        assert self.with_selector and self.linear_dim_exp > 0
        prefix = 'backbone.'
        backbone_state_dict = {key[len(prefix):]: val for key, val in state_dict.items() if prefix in key}
        err_msg = self.backbone.load_state_dict(backbone_state_dict, strict=False)
        assert len(err_msg.unexpected_keys) == 0 and all('lora' not in key and 'adapter' not in key for key in err_msg.missing_keys)

        prefix = 'lora_exp_projector.'
        sel_state_dict = {key[len(prefix):]: val for key, val in state_dict.items() if prefix in key}
        assert len(sel_state_dict) == 2 and hasattr(self, 'lora_exp_projector')
        old_weight, old_bias = sel_state_dict['weight'], sel_state_dict['bias']
        basic_dim, input_dim = old_weight.shape
        if self.linear_dim_exp > basic_dim:
            new_weight, new_bias = self.lora_exp_projector.weight.cpu().detach(), \
                self.lora_exp_projector.bias.cpu().detach()
            new_weight[:basic_dim, :] = old_weight
            new_bias[:basic_dim] = old_bias
            sel_state_dict['weight'] = new_weight
            sel_state_dict['bias'] = new_bias
        self.lora_exp_projector.load_state_dict(sel_state_dict, strict=True)

    def load_delta(self, split: str) -> nn.Linear:
        """
            split: selector / p[1-10]
        """
        assert self.is_infer, 'load delta is only available under inference mode'
        lora_state, linear_layer = self.split_to_lora_linear[split]
        if split != self.cur_loaded_split:
            err_msg = self.backbone.load_state_dict(lora_state, strict=False)
            assert len(err_msg.unexpected_keys) == 0 and all('lora' not in key and 'adapter' not in key for key in err_msg.missing_keys)
            self.cur_loaded_split = split
        return linear_layer

    def add_delta(self, split: str, lora_state: dict, linear_layer: nn.Linear):
        if split in self.split_to_lora_linear:
            return
        self.split_to_lora_linear[split] = (lora_state, linear_layer)

    def forward(self, *args, **kwargs):
        if 'output_hidden_states' not in kwargs:
            kwargs['output_hidden_states'] = True
        return self.backbone(*args, **kwargs)['hidden_states'][-1]
