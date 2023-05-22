import torch.nn as nn
from transformers import AutoModel
from .bert_lora import BertModel, BertLoRAConfig
from .llama_lora import LlamaConfigLoRA, LlamaForCausalLMLoRA
from transformers import LlamaConfig, LlamaForCausalLM


class BertEncoder(nn.Module):

    def __init__(self, args):
        super(BertEncoder, self).__init__()

        max_seq_len, model_path = args.max_length, args.bert_path
        model = AutoModel.from_pretrained(model_path)
        model.config.max_length = max_seq_len
        state_dict, config_dict = model.state_dict(), model.config.to_dict()
        del model
        config_dict['apply_lora'], config_dict['lora_r'] = True, 4
        config_dict['apply_adapter'], config_dict['adapter_type'], config_dict['adapter_size'] = False, 'houlsby', 64
        self.bert_config = BertLoRAConfig.from_dict(config_dict)

        # load model
        self.encoder = BertModel(self.bert_config)

        err_msg_lora = self.load_both_state_dict(state_dict, strict=False)
        assert len(err_msg_lora.unexpected_keys) == 0 and \
               all('lora' in key for key in err_msg_lora.missing_keys)

        # the dimension for the final outputs
        args.encoder_output_size = self.output_size = self.bert_config.hidden_size
        # self.out_dim = self.output_size
        # self.linear_transform = nn.Linear(self.bert_config.hidden_size, self.output_size, bias=True)

    def load_both_state_dict(self, state_dict, strict=True):
        err_msg_lora = self.encoder.load_state_dict(state_dict, strict=strict)
        return err_msg_lora

    def forward(self, input_ids, attention_mask, mask_ids):
        # generate representation under a certain encoding strategy
        output = self.encoder(input_ids, attention_mask)[0]
        output = output[mask_ids]
        return output


class LlamaEncoder(nn.Module):
    def __init__(self, args):
        super(LlamaEncoder, self).__init__()

        # initialize model weights and config
        max_seq_len, model_path = args.max_length, args.bert_path
        model_config = LlamaConfig.from_pretrained(model_path)
        model_config.max_length = max_seq_len
        config_dict = model_config.to_dict()
        config_dict['apply_lora'], config_dict['lora_r'] = True, 4
        self.config = LlamaConfigLoRA.from_dict(config_dict)

        self.encoder = LlamaForCausalLMLoRA(self.config)
        model = LlamaForCausalLM.from_pretrained(model_path)
        model_state = model.state_dict()
        del model
        err_msg = self.encoder.load_state_dict(model_state, strict=False)
        assert len(err_msg.unexpected_keys) == 0 and all('lora' in key for key in err_msg.missing_keys)

        # the dimension for the final outputs
        args.encoder_output_size = self.output_size = model_config.hidden_size
        # self.out_dim = self.output_size
        # self.linear_transform = nn.Linear(self.bert_config.hidden_size, self.output_size, bias=True)

    def forward(self, input_ids, attention_mask, mask_ids):
        # generate representation under a certain encoding strategy
        output = self.encoder(input_ids, attention_mask, output_hidden_states=True)['hidden_states'][-1]
        output = output[mask_ids]
        return output
