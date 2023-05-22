import os
import torch
import shutil
from tqdm import tqdm
import loralib as lora
from loss_similarity import LossSimilarity
from .testing import valid_save, test_by_best
from preprocess import get_tag_set_by_dataset
from preprocess import init_contrastive_dataloader
from models import BertLoRAWithSelector, mark_only_adapter_as_trainable


@torch.no_grad()
def get_prototypes(config, proto_memory, model, tokenizer, loss_sim: LossSimilarity):
    memset, rangeset = [], [0]
    for data in proto_memory:
        memset += data
        rangeset.append(rangeset[-1] + len(data))
    data_loader = init_contrastive_dataloader(config, {'train': [], 'test': memset}, tokenizer)[0]['test']
    features = []
    for batch in tqdm(data_loader, desc='prototype'):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(config['device'], non_blocking=True)
        hidden_mask = batch['input_ids'] == tokenizer.mask_token_id
        feature = loss_sim.forward_model_logits(
            '', model.lora_projector, batch, model, hidden_mask, is_selector=False)[1]
        features.append(feature.detach().cpu())
    features = torch.cat(features, dim=0)
    prototypes, hidden_size = [], config['hidden_size']
    print("proto_instances:%d" % len(features))
    for i in range(len(proto_memory)):
        if rangeset[i] == rangeset[i+1]:
            prototypes.append(torch.zeros((1, hidden_size)))
        else:
            prototypes.append(features[rangeset[i]:rangeset[i+1], :].mean(0, keepdims=True))
    prototypes = torch.cat(prototypes, 0).to(config['device'])
    return prototypes


def train_emar(config, data_loaders1, data_loaders2, model, tokenizer, loss_sim: LossSimilarity,
               optimizer, trained_epoch, global_step, best_results=None, proto_memory=None):
    # train selector from initialization
    assert config['train']['train_expert_selector'] == 0 and not config['train']['use_expert_selector']
    assert config['train']['loss_adverse_step'] <= 0

    # mark only model itself as trainable
    if isinstance(model, BertLoRAWithSelector):
        for n, p in model.named_parameters():
            p.requires_grad = 'bert_lora' in n or 'lora_linear_out' in n

    exp_path = os.path.join(config['logging']['path_base'], config['logging']['unique_string'])
    model_path = os.path.join(exp_path, 'models')
    os.makedirs(model_path, exist_ok=True)
    train_loader1, valid_groups1, test_groups1, train_infer1 = data_loaders1['train_infer'], \
        data_loaders1['valid_groups'], data_loaders1['test_groups'], data_loaders1['train_infer']
    train_loader2, valid_groups2, test_groups2, train_infer2 = data_loaders2['train_infer'], \
        data_loaders2['valid_groups'], data_loaders2['test_groups'], data_loaders2['train_infer']
    assert config['task'] == 'fewshot'

    num_epoch = config['train']['num_epochs']
    assert config['train']['save_step'] == config['train']['num_steps'] == -1
    grad_accu_step = config['train']['gradient_accumulation_steps']
    if best_results is None:
        print('WARNING: best_results is None ......')
        best_results = {
            'best_step': -1,
            'best_step_acc': 0.,
            'best_epoch': -1,
            'best_epoch_acc': 0.,
        }
    else:
        print('loaded best results:', str(best_results))
    train_loss = 0.0
    model.zero_grad()
    if config['train']['save_option'] >= 1 and config['plm']['model_name'] != 'cpm':
        tokenizer.save_pretrained(model_path)

    train_tag_set = list(get_tag_set_by_dataset(train_loader1))
    print('label number in training set:', len(train_tag_set), '......')
    tag_to_loss_weight = {tag: 1. for tag in train_tag_set}

    for epoch in range(trained_epoch, int(num_epoch) + trained_epoch):  # ALERT: different!

        # I. get current prototypes
        cur_prototypes = get_prototypes(config, proto_memory, model, tokenizer, loss_sim)

        # II. train_simple_model
        epoch_iterator1 = tqdm(train_loader1, desc="data iteration 1") if len(train_loader1) > 20 else train_loader1
        for step, batch in enumerate(epoch_iterator1):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(config['device'], non_blocking=True)

            model.train()
            if config['plm']['apply_lora']:
                lora.mark_only_lora_as_trainable(model)
            elif config['plm']['apply_adapter']:
                mark_only_adapter_as_trainable(model)

            try:
                loss, loss_vec = loss_sim.forward_similarity_train(model, tokenizer, batch,
                                                                   tag_to_loss_weight=tag_to_loss_weight,
                                                                   prototypes=None)
            except RuntimeError as err:
                from IPython import embed
                embed()
                raise err

            if grad_accu_step > 1:
                loss /= grad_accu_step

            loss.backward()
            train_loss += loss.item()

            if (step + 1) % grad_accu_step == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # 梯度累加
                optimizer.step()
                model.zero_grad()
                global_step += 1

        # III. train_prototype_model
        epoch_iterator2 = tqdm(train_loader2, desc="data iteration 2") if len(train_loader2) > 20 else train_loader2
        for step, batch in enumerate(epoch_iterator2):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(config['device'], non_blocking=True)

            model.train()
            if config['plm']['apply_lora']:
                lora.mark_only_lora_as_trainable(model)
            elif config['plm']['apply_adapter']:
                mark_only_adapter_as_trainable(model)

            try:
                loss, loss_vec = loss_sim.forward_similarity_train(model, tokenizer, batch,
                                                                   tag_to_loss_weight=tag_to_loss_weight,
                                                                   prototypes=cur_prototypes)
            except RuntimeError as err:
                from IPython import embed
                embed()
                raise err

            if grad_accu_step > 1:
                loss /= grad_accu_step

            loss.backward()
            train_loss += loss.item()

            if (step + 1) % grad_accu_step == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # 梯度累加
                optimizer.step()
                model.zero_grad()
                global_step += 1

        valid_save(config, model, tokenizer, loss_sim, is_step=False,
                   best_results=best_results, cur_step=global_step, cur_epoch=epoch,
                   optimizer=optimizer, scheduler=None,
                   valid_loader=valid_groups1, valid_infer=None, train_infer=train_infer1,
                   extra_module_info=None, train_expert_selector=False, prototypes=cur_prototypes)
    average_loss = train_loss / global_step if global_step > 0 else 0
    best_step, best_step_acc, best_epoch, best_epoch_acc = best_results['best_step'], best_results['best_step_acc'], \
        best_results['best_epoch'], best_results['best_epoch_acc']

    print('got best epoch:', best_epoch, 'accuracy:', best_epoch_acc)
    print('got best step:', best_step, 'accuracy:', best_step_acc)
    if best_epoch_acc > best_step_acc:
        best_key = 'epoch-model'
        shutil.copy(os.path.join(model_path, 'epoch-best.pkl'), os.path.join(model_path, 'tot-best.pkl'))
    else:
        best_key = 'step-model'
        shutil.copy(os.path.join(model_path, 'step-best.pkl'), os.path.join(model_path, 'tot-best.pkl'))
    cur_prototypes = get_prototypes(config, proto_memory, model, tokenizer, loss_sim)
    test_results = test_by_best(config, model, tokenizer, loss_sim,
                                best_epoch, best_epoch_acc, best_step, best_step_acc,
                                test_groups1, test_infer=None, train_infer=train_infer1,
                                best_model=best_results[best_key],
                                extra_module_info=None, prototypes=cur_prototypes)
    with open(os.path.join(exp_path, 'flag'), 'w') as fout:
        fout.write('complete\n')
    return global_step, average_loss, \
        best_step, best_step_acc, best_epoch, best_epoch_acc, best_results, test_results
