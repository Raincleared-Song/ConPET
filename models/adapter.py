import torch.nn as nn
from transformers.activations import get_activation


class Adapter(nn.Module):
    def __init__(self, dim, r, act):
        super().__init__()
        self.adapter_A = nn.Linear(dim, r)
        self.act = get_activation(act)
        self.adapter_B = nn.Linear(r, dim)

    def forward(self, x, residual):
        result = self.adapter_A(x)
        result = self.act(result)
        result = self.adapter_B(result)
        return result + residual


def mark_only_adapter_lora_as_trainable(model):
    for n, p in model.named_parameters():
        if 'adapter' not in n and 'lora' not in n:
            p.requires_grad = False


def adapter_lora_state_dict(model):
    my_state_dict = model.state_dict()
    return {k: my_state_dict[k] for k in my_state_dict if 'adapter' in k or 'lora' in k}
