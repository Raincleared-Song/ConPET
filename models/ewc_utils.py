import torch
from tqdm import tqdm
import loralib as lora
from preprocess import get_tag_set_by_dataset
from .adapter import mark_only_adapter_lora_as_trainable


def get_model_mean_fisher(config, train_loader, model, tokenizer, loss_sim,
                          extra_module_info=None, tag_to_loss_weight=None):
    if config['dataset']['method_type'] != 'linear':
        raise NotImplementedError('invalid method_type')
    # get grad model fisher
    fisher_batch_size = 1  # can be adjusted
    batch_epoch = (len(train_loader) - 1) // fisher_batch_size + 1
    train_tag_set = get_tag_set_by_dataset(train_loader)
    print('label number in training set:', len(train_tag_set), '......')

    model.eval()
    if config['plm']['apply_lora']:
        lora.mark_only_lora_as_trainable(model)
    elif config['plm']['apply_adapter']:
        mark_only_adapter_lora_as_trainable(model)

    grad_mean, grad_fisher = {}, {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            n = n.replace('.', '__')
            grad_mean[n] = p.cpu()
            grad_fisher[n] = p.detach().clone().zero_()
    loss_func = torch.nn.CrossEntropyLoss()

    epoch_iterator = tqdm(train_loader, desc="data iteration") if len(train_loader) > 20 else train_loader
    for step, batch in enumerate(epoch_iterator):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(config['device'], non_blocking=True)

        logits, loss, _ = loss_sim.generate_continual_logits(
            'train', model, tokenizer, batch, extra_module_info, tag_to_loss_weight, lwf_logit=True)
        with torch.no_grad():
            label_weights = torch.softmax(logits, dim=1)
            label_weights = torch.mean(label_weights, dim=0)

        batch_sz, label_num = logits.shape
        for label_index in range(label_num):
            label = torch.LongTensor([label_index] * batch_sz).to(config['device'])
            neg_likelihood = loss_func(logits, label)  # --> get neg log-likelihoods for this class
            # Calculate gradient of negative loglikelihood
            model.zero_grad()
            neg_likelihood.backward(retain_graph=True if (label_index + 1) < logits.shape[1] else False)
            # Square gradients and keep running sum (using the weights)
            for n, p in model.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    if p.grad is not None:
                        grad_fisher[n] += label_weights[label_index] * (p.grad.detach() ** 2)
    grad_fisher = {n: p.cpu() / batch_epoch for n, p in grad_fisher.items()}

    return grad_mean, grad_fisher
