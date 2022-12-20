import torch
from tqdm import tqdm
import loralib as lora
from preprocess import get_tag_set_by_dataset
from .adapter import mark_only_adapter_as_trainable


def get_model_mean_fisher(config, train_loader, model, tokenizer, loss_sim,
                          type_embeds=None, type_counter=None,
                          extra_module_info=None, tag_to_loss_weight=None):
    # get grad model fisher
    fisher_batch_size = 1  # can be adjusted
    batch_epoch = (len(train_loader) - 1) // fisher_batch_size + 1
    train_tag_set = get_tag_set_by_dataset(train_loader)
    print('label number in training set:', len(train_tag_set), '......')

    model.eval()
    if config['plm']['apply_lora']:
        lora.mark_only_lora_as_trainable(model)
    elif config['plm']['apply_adapter']:
        mark_only_adapter_as_trainable(model)

    grad_params = [param for param in model.parameters() if param.requires_grad]
    grad_mean, grad_fisher = [param.clone() for param in grad_params], []

    epoch_iterator = tqdm(train_loader, desc="data iteration") if len(train_loader) > 20 else train_loader
    for step, batch in enumerate(epoch_iterator):
        model.zero_grad()
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(config['device'], non_blocking=True)

        loss, _ = loss_sim.forward_similarity_train(model, tokenizer, batch,
                                                    type_embeds, type_counter,
                                                    extra_module_info, tag_to_loss_weight)

        loss.backward()
        grad_params = [param for param in model.parameters() if param.requires_grad]
        if len(grad_fisher) == 0:
            grad_fisher = [param.grad ** 2 / batch_epoch for param in grad_params]
        else:
            grad_fisher = [grad_fisher[idx] + param.grad ** 2 / batch_epoch for idx, param in enumerate(grad_params)]

    return grad_mean, grad_fisher
