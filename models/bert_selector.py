import copy
import torch.nn as nn
import loralib as lora
from transformers import AutoModelForMaskedLM
from .bert_lora import BertForMaskedLMLoRA, BertLoRAConfig


class BertLoRAWithSelector(nn.Module):
    def __init__(self, config, with_selector: bool, linear_dim: int, linear_dim_exp: int):
        super().__init__()
        self.extra_config = config

        # initialize model weights and config
        max_seq_len, model_path = config['dataloader']['max_seq_length'], config['plm']['model_path']
        model = AutoModelForMaskedLM.from_pretrained(model_path)
        model.config.max_length = max_seq_len
        state_dict, config_dict = model.state_dict(), model.config.to_dict()
        del model
        config_dict['apply_lora'], config_dict['lora_r'] = config['plm']['apply_lora'], config['plm']['lora_r']
        config_dict['apply_adapter'], config_dict['adapter_type'], config_dict['adapter_size'] = \
            config['plm']['apply_adapter'], config['plm']['adapter_type'], config['plm']['adapter_size']
        self.config = BertLoRAConfig.from_dict(config_dict)

        self.with_selector = with_selector
        self.bert_lora = BertForMaskedLMLoRA(self.config)
        self.linear_dim = linear_dim
        if linear_dim > 0:
            self.lora_linear_out = nn.Linear(self.bert_lora.config.hidden_size, linear_dim)
        if with_selector:
            self.bert_selector = BertForMaskedLMLoRA(self.config)
            selector_dict = lora.lora_state_dict(self.bert_selector) if config_dict['apply_lora'] \
                else self.bert_selector.state_dict()
            self.selector_dict = copy.deepcopy(selector_dict)
            self.linear_dim_exp = linear_dim_exp
            if linear_dim_exp > 0:
                self.lora_linear_selector = nn.Linear(self.bert_lora.config.hidden_size, linear_dim_exp)

        err_msg_lora, err_msg_exp = self.load_both_state_dict(state_dict, strict=False)
        assert len(err_msg_lora.unexpected_keys) == 0 and \
               all('lora' in key for key in err_msg_lora.missing_keys)
        assert err_msg_exp is None or len(err_msg_exp.unexpected_keys) == 0 and \
               all('lora' in key for key in err_msg_exp.missing_keys)

        self.lora_alignment = None

    def init_alignment(self):
        if self.lora_alignment is None:
            self.lora_alignment = nn.Linear(self.bert_lora.config.hidden_size, self.bert_lora.config.hidden_size)
            self.lora_alignment = self.lora_alignment.to(self.extra_config['device'])

    def resize_token_embeddings(self, new_num_tokens: int):
        self.bert_lora.resize_token_embeddings(new_num_tokens)
        if self.with_selector:
            self.bert_selector.resize_token_embeddings(new_num_tokens)

    def clone(self):
        return copy.deepcopy(self.bert_lora)

    def load_selector_state_dict(self, state_dict, strict=True):
        # if the output dimension of lora_linear_out is different, extent the weight and bias instead of re-initialize
        assert self.with_selector
        new_state_dict, prefix = {}, 'bert_selector.'
        for key, val in state_dict.items():
            if key.startswith(prefix):
                new_state_dict[key[len(prefix):]] = val
        err_msg = self.bert_selector.load_state_dict(new_state_dict, strict=strict)
        self.selector_dict = copy.deepcopy(new_state_dict)
        if self.linear_dim_exp > 0:
            linear_state_dict, prefix = {}, 'lora_linear_selector.'
            for key, val in state_dict.items():
                if key.startswith(prefix):
                    linear_state_dict[key[len(prefix):]] = val
            old_weight, old_bias = linear_state_dict['weight'], linear_state_dict['bias']
            basic_dim, input_dim = old_weight.shape
            assert self.linear_dim_exp >= basic_dim
            if self.linear_dim_exp > basic_dim:
                new_weight, new_bias = self.lora_linear_selector.weight, self.lora_linear_selector.bias
                new_weight, new_bias = new_weight.cpu().detach(), new_bias.cpu().detach()
                new_weight[:basic_dim, :] = old_weight
                new_bias[:basic_dim] = old_bias
                linear_state_dict['weight'], linear_state_dict['bias'] = new_weight, new_bias
            self.lora_linear_selector.load_state_dict(linear_state_dict, strict=True)
        return err_msg

    def load_both_state_dict(self, state_dict, strict=True):
        err_msg_lora = self.bert_lora.load_state_dict(state_dict, strict=strict)
        err_msg_exp = None
        if self.with_selector:
            err_msg_exp = self.bert_selector.load_state_dict(state_dict, strict=strict)
        return err_msg_lora, err_msg_exp

    def forward(self, *args, **kwargs):
        return self.bert_lora(*args, **kwargs)

    def forward_select(self, *args, **kwargs):
        assert self.with_selector
        return self.bert_selector(*args, **kwargs)
