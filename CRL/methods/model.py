import torch.nn as nn
import torch.nn.functional as F
from .backbone import BertEncoder, LlamaEncoder


class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        if 'bert' in args.bert_path:
            self.encoder = BertEncoder(args)
        elif 'llama' in args.bert_path:
            self.encoder = LlamaEncoder(args)
        else:
            raise NotImplementedError(f'invalid bert_path: {self.args.bert_path}')
        dim_in = self.output_size = self.encoder.output_size
        self.lora_head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, args.feat_dim)
        )

    def bert_forward(self, input_ids, attention_mask, mask_ids):
        out = self.encoder(input_ids, attention_mask, mask_ids)
        xx = self.lora_head(out)
        xx = F.normalize(xx, p=2, dim=1)
        return out, xx
