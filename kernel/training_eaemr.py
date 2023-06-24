import torch
import random
from tqdm import tqdm, trange
from loss_similarity import LossSimilarity
from preprocess import init_contrastive_dataloader


@torch.no_grad()
def get_embeddings(config, memorized_samples, model, tokenizer, loss_sim: LossSimilarity,
                   before_reverse=False, return_tags=False):
    if isinstance(memorized_samples, dict):
        data_loader = memorized_samples['train_infer']
    else:
        data_loader = init_contrastive_dataloader(
            config, {'train': [], 'test': memorized_samples}, tokenizer)[0]['test']
    features, key2tags = {}, {}
    backup_alignment = model.lora_alignment
    if before_reverse:
        model.lora_alignment = None
    for batch in tqdm(data_loader, desc='embedding'):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(config['device'], non_blocking=True)
        hidden_mask = batch['input_ids'] == tokenizer.mask_token_id
        model_name = config['plm']['model_name']
        model_linear_out = model.lora_linear_out if model_name not in ['cpm', 'llama'] else model.lora_projector
        feature = loss_sim.forward_model_logits(
            '', model_linear_out, batch, model, hidden_mask, is_selector=False)[1]
        feature = feature.detach().cpu()
        assert len(feature) == len(batch['sample_keys']) == len(batch['tags'])
        for fea, sample_key, tag in zip(feature, batch['sample_keys'], batch['tags']):
            features[sample_key] = fea
            key2tags[sample_key] = int(tag)
    if before_reverse:
        model.lora_alignment = backup_alignment
    return (features, key2tags) if return_tags else features


def update_alignment_model(config, cur_embeddings: dict, mem_embeddings: dict, lora_alignment):
    # train selector from initialization
    assert config['train']['train_expert_selector'] == 0 and not config['train']['use_expert_selector']
    assert config['train']['loss_adverse_step'] <= 0

    batch_size = 32
    learning_rate = 0.0001
    epoch_num = 20
    loss_func = torch.nn.MSELoss()
    device = config['device']
    optimizer = torch.optim.Adam(lora_alignment.parameters(), lr=learning_rate)

    assert cur_embeddings.keys() == mem_embeddings.keys()
    key_set = list(cur_embeddings.keys())
    random.shuffle(key_set)
    train_size = len(key_set)

    for epoch in trange(epoch_num, desc='alignment'):
        start_idx = 0
        while start_idx < train_size:
            cur_inputs = [cur_embeddings[key] for key in key_set[start_idx:start_idx+batch_size]]
            cur_targets = [mem_embeddings[key] for key in key_set[start_idx:start_idx+batch_size]]
            cur_inputs, cur_targets = torch.stack(cur_inputs).to(device), torch.stack(cur_targets).to(device)

            optimizer.zero_grad()
            outputs = lora_alignment(cur_inputs)
            loss = loss_func(outputs, cur_targets)
            loss.backward()
            optimizer.step()

            start_idx += batch_size

    optimizer.zero_grad()
