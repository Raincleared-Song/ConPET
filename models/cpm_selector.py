import torch
import torch.nn as nn
# from opendelta import LoraModel
from bmt_models import CPMBeeConfig, CPMBeeTorch


class CPMLoRAWithSelector(nn.Module):
    def __init__(self, config, is_infer: bool, with_selector: bool,
                 linear_dim: int, linear_dim_exp: int, model_state: dict = None):
        super().__init__()
        self.extra_config = config
        self.is_infer = is_infer

        # initialize model weights and config
        self.config = CPMBeeConfig.from_json_file('bmt_models/bee_config.json')
        self.config.lora_r = config['plm']['lora_r'] if config['plm']['apply_lora'] else -1
        self.with_selector = with_selector

        # set alignment and the linear dim
        self.linear_dim = linear_dim
        self.linear_dim_exp = linear_dim_exp if with_selector and linear_dim_exp > 0 else -1
        self.add_alignment = config['train']['continual_method'] == 'eaemr' or \
            'is_eaemr' in config and config['is_eaemr']

        self.backbone = CPMBeeTorch(self.config)
        if model_state is None:
            model_state = torch.load(config['plm']['model_path'], map_location='cpu')
        self.resize_token_embeddings(model_state)
        err_msg = self.backbone.load_state_dict(model_state, strict=False)
        assert len(err_msg.unexpected_keys) == 0 and all('lora' in key or 'adapter' in key for key in err_msg.missing_keys)

        # self.delta_model = LoraModel(backbone_model=self.backbone, lora_r=config['plm']['lora_r'],
        #                              modified_modules=['project_q', 'project_v'])
        # self.delta_model.create_config_from_model()
        # self.delta_model.freeze_module(exclude=['deltas'], set_state_dict=True)
        # self.delta_model.log()

        if self.linear_dim > 0:
            self.lora_projector = nn.Linear(self.config.dim_model, self.linear_dim, dtype=torch.half)
        if self.linear_dim_exp > 0:
            self.lora_exp_projector = nn.Linear(self.config.dim_model, self.linear_dim_exp, dtype=torch.half)
        if self.add_alignment:
            self.lora_alignment = nn.Linear(self.config.dim_model, self.config.dim_model, dtype=torch.half)
        if self.is_infer:
            self.split_to_lora_linear = {}

        self.cur_loaded_split = ''

    def resize_token_embeddings(self, state_dict):
        new_vocab_size = self.config.vocab_size
        embed_key = 'input_embedding.weight'
        old_vocab_size = state_dict[embed_key].shape[0]
        assert old_vocab_size in [86585, 86591], str(old_vocab_size)
        if new_vocab_size == old_vocab_size:
            return state_dict
        assert new_vocab_size > old_vocab_size
        new_embeddings = nn.Embedding(new_vocab_size, self.config.dim_model).weight.detach()
        new_embeddings[:old_vocab_size, :] = state_dict[embed_key]
        state_dict[embed_key] = new_embeddings

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
        return self.backbone(*args, **kwargs)
