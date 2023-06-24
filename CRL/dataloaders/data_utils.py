import numpy as np
from transformers import BertTokenizer, LlamaTokenizer


def complex_sample(samples, size=None, replace=True, p=None):
    sampled_idx = np.random.choice(len(samples), size=size, replace=replace, p=p)
    if size is None:
        return samples[sampled_idx]
    res = []
    for idx in sampled_idx:
        res.append(samples[idx])
    return res


def get_tokenizer(args):
    if 'bert' in args.bert_path:
        tokenizer = BertTokenizer.from_pretrained(args.bert_path, model_max_length=args.max_length)
        tokenizer.add_special_tokens({"additional_special_tokens": [f"[unused{idx}]" for idx in range(5)]})
        assert all(len(tokenizer.tokenize(f"[unused{idx}]")) == 1 for idx in range(5))
    elif 'llama' in args.bert_path:
        tokenizer = LlamaTokenizer.from_pretrained(args.bert_path)
        tokenizer.unk_token, tokenizer.unk_token_id = '<unk>', 0
        tokenizer.pad_token, tokenizer.pad_token_id = '<unk>', 0
        tokenizer.bos_token, tokenizer.eos_token, tokenizer.mask_token = '<s>', '</s>', '<0x05>'
        tokenizer.padding_side = 'right'
        tokenizer.add_special_tokens({
            "additional_special_tokens": ['<s>', '</s>', '<unk>'] + [f'<0x{idx:02}>' for idx in range(6)]})
        assert all(len(tokenizer.tokenize(f"<0x{idx:02}>")) == 1 for idx in range(6))
    else:
        raise NotImplementedError(f'invalid bert_path: {args.bert_path}')
    return tokenizer
