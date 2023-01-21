import os
import sys
import math
import copy
import torch
import random
import numpy as np
import torch.nn as nn
import loralib as lora
from global_var import GLOBAL
import torch.nn.functional as F
from transformers import T5Tokenizer, T5ForConditionalGeneration


def save_model(path: str, model, optimizer_name, optimizer, scheduler,
               trained_epoch: int, global_step: int, extra=None):
    ret = {
        'optimizer_name': optimizer_name,
        'optimizer': optimizer.state_dict(),
        'trained_epoch': trained_epoch,
        'global_step': global_step
    }
    if model is not None:
        if hasattr(model, 'module'):
            model = model.module
        model_state_dict = model.state_dict()
        if any('lora_A' in key for key in model_state_dict.keys()):
            model_state_dict = lora.lora_state_dict(model)
        elif any('adapter' in key for key in model_state_dict.keys()):
            model_state_dict = model.state_dict()
            model_state_dict = {k: model_state_dict[k] for k in model_state_dict if 'adapter' in k}
        ret['model'] = model_state_dict
    if scheduler is not None:
        ret['scheduler'] = scheduler.state_dict()
    if extra is not None:
        assert isinstance(extra, dict)
        ret.update(extra)
    try:
        torch.save(ret, path)
    except Exception as err:
        print(f'Save model failure with error {err}', file=sys.stderr)


class JensenShannonDivergence(nn.Module):
    """
    Jensen-Shannon Divergence Loss
    logits1, logits2: logits shape like [batch_size, class_number]
    reduction: batch_reserve for output a [batch_size] shape loss vector
    """
    def __init__(self, reduction="batch_reserve", base=2.0):
        super().__init__()
        self.factor = math.log(base)
        self.reduction = reduction

    @staticmethod
    def relative_entropy(probs_p, probs_q):
        return probs_p * (torch.log(probs_p) - torch.log(probs_q))

    def forward(self, logits1, logits2):
        probs1 = F.softmax(logits1, dim=1)
        probs2 = F.softmax(logits2, dim=1)

        probs_m = (probs1 + probs2) / 2
        loss = (self.relative_entropy(probs1, probs_m) + self.relative_entropy(probs2, probs_m)) / 2 / self.factor
        if self.reduction == "batch_reserve":
            loss = torch.sum(loss, dim=1)
        return loss


def gather_t5_result(raw_result: str) -> str:
    start, s_key = raw_result.find('<extra_id_0>'), '<extra_id_0>'
    if start == -1:
        start, s_key = raw_result.find('<pad>'), '<pad>'
    if start == -1:
        start, s_key = 0, ''
    start = start + len(s_key)
    end, e_key = raw_result.find('<extra_id_1>'), '<extra_id_1>'
    if end == -1:
        end, e_key = raw_result.find('</s>'), '</s>'
    if end == -1:
        end = len(raw_result)
    return raw_result[start:end].strip().strip('.')


def init_seed(config):
    device = config['device']
    torch.cuda.set_device(device)
    seed = config['reproduce']['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    determine = torch.use_deterministic_algorithms if hasattr(torch, 'use_deterministic_algorithms') \
        else getattr(torch, 'set_deterministic')
    determine(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'


def batch_shuffle(*args):
    arg_len = len(args[0])
    for arg in args:
        assert len(arg) == arg_len, f'{len(arg)} | {arg_len}'
    idx_ls = list(range(arg_len))
    random.shuffle(idx_ls)
    ret_args = []
    for arg in args:
        ret_args.append([arg[idx] for idx in idx_ls])
    return tuple(ret_args)


def dot_product(left_logits, right_logits):
    assert left_logits.shape == right_logits.shape and left_logits.ndim == 2
    return torch.sum(left_logits * right_logits, dim=1)


def load_partial_checkpoint(config, cp_path: str, model, optimizer, scheduler=None):
    params = torch.load(cp_path, map_location='cpu')
    ori_state_dict, ori_opt_dict, ori_sche_dict = params['model'], params['optimizer'], None
    strict = not (config['plm']['apply_lora'] or config['plm']['apply_adapter'])
    model.load_state_dict(ori_state_dict, strict=strict)
    # ALERT: This condition need to be checked
    if params['optimizer']['state'] and not (config['task'] == 'fewshot' and config['is_test']):
        optimizer.load_state_dict(ori_opt_dict)
    else:
        ori_opt_dict = copy.deepcopy(optimizer.state_dict())
    if scheduler is not None:
        assert 'scheduler' in params
        ori_sche_dict = params['scheduler']
        scheduler.load_state_dict(ori_sche_dict)
    return ori_state_dict, ori_opt_dict, ori_sche_dict


def get_gpu_usage():
    cmd = "echo -e `nvidia-smi | grep 'Default' | grep '[0-9]*MiB /' -o | grep '[0-9]*' -o`"
    fin = os.popen(cmd)
    res = fin.read()
    fin.close()
    res = res.replace("-e", "").strip()
    return res


def update_tag_loss_count(tags, loss_vec, tag_to_loss_count):
    assert len(tags) == len(loss_vec)
    for tag, loss in zip(tags, loss_vec):
        tag_to_loss_count[tag][0] += 1
        tag_to_loss_count[tag][1] += loss
