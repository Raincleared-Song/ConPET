import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from global_var import GLOBAL
from sklearn.cluster import KMeans
from models import BertLoRAWithSelector, CPMLoRAWithSelector, LlamaLoRAWithSelector
from utils import JensenShannonDivergence, dot_product, complex_sample, write_cache, read_cache


class LossSimilarity:
    def __init__(self, config):
        self.config = config
        self.sim_type = config['train']['similarity']
        self.negative_label = 0 if self.sim_type != 'cosine' else -1
        self.negative_logit = float(config['train']['null_expert_logit'])
        self.embed_size = config['vocab_size'] if self.sim_type == 'divergence' else config['hidden_size']
        self.curr_bound = int(config['dataset']['special_part'][1:])
        self.loss_func, self.extra_module = self.get_similarity_by_config()
        self.use_linear = config['dataset']['method_type'] == 'linear'
        self.model_name = config['plm']['model_name']
        self.continual_method = config['train']['continual_method']
        self.split_to_model_cache = {}
        self.prompt_token_map_cache = {}
        self.continual_logit_cache = {}
        self.float_type = torch.float if self.model_name not in ['cpm', 'llama'] else torch.half
        self.np_float_type = np.float16 if self.model_name not in ['cpm', 'llama'] else np.float32
        assert self.use_linear

    def get_similarity_by_config(self):
        extra_module = None
        if self.sim_type == 'cosine':
            loss_func = nn.CosineEmbeddingLoss(reduction="none")
        elif self.sim_type == 'divergence':
            loss_func = JensenShannonDivergence(reduction="batch_reserve")
        elif self.sim_type == 'sigmoid_distance':
            loss_func = None
            extra_module = nn.Linear(self.embed_size, 1).to(self.config['device'])
        elif self.sim_type == 'dot':
            loss_func = dot_product
        else:
            raise NotImplementedError('invalid similarity function: ' + self.sim_type)
        return loss_func, extra_module

    def get_hidden_mask(self, batch, tokenizer):
        if self.config['dataset']['method_type'] in ['prompt', 'linear']:
            use_mask = self.config['dataset']['use_mask']
            mask_id = tokenizer.mask_id if self.model_name == 'cpm' else tokenizer.mask_token_id
            cls_id = tokenizer.bos_id if self.model_name == 'cpm' \
                else (tokenizer.bos_token_id if self.model_name == 'llama' else tokenizer.cls_token_id)
            hidden_mask = torch.eq(batch['input_ids'], (mask_id if use_mask else cls_id))
        else:
            raise NotImplementedError('invalid method_type: ' + self.config['method_type'])
        return hidden_mask

    def forward_by_sim_mat(self, left_logits, right_logits, sim_mat):
        negative_weight = self.config['train']['negative_weight']
        if self.sim_type == 'cosine':
            cos_mat = self.loss_func(left_logits, right_logits, sim_mat)
            weight_mat = sim_mat.float()
            weight_mat[sim_mat == -1] = negative_weight
            loss = torch.mean(cos_mat * weight_mat)
        elif self.sim_type == 'divergence':
            div = self.loss_func(left_logits, right_logits)
            loss = torch.mean(- torch.log(1 - div) * sim_mat - negative_weight * torch.log(div) * (1 - sim_mat))
        elif self.sim_type == 'sigmoid_distance':
            vec_distance = torch.square(left_logits - right_logits)

            # same sample with sigmoid distance 0, different with sigmoid distance 1
            sim_score = torch.sigmoid(self.extra_module(vec_distance).squeeze(1))
            entropy_mat = - torch.log(2 - sim_score) * sim_mat - torch.log(sim_score) * (1 - sim_mat)
            # sim_score = vec_distance
            # vec_distance = torch.sum(vec_distance, dim=1)
            # entropy_mat = vec_distance * sim_mat - vec_distance * (1 - sim_mat)

            weight_mat = sim_mat.float()
            weight_mat[sim_mat == 0] = negative_weight
            loss = torch.mean(entropy_mat * weight_mat)
        elif self.sim_type == 'dot':
            dot_products = self.loss_func(left_logits, right_logits)
            positive_mask, negative_mask = sim_mat == 1, sim_mat == 0
            dot_products[positive_mask] = torch.maximum(200 - dot_products[positive_mask], torch.tensor(0))
            dot_products[negative_mask] = torch.maximum(dot_products[negative_mask], torch.tensor(0))
            weight_mat = sim_mat.float()
            weight_mat[sim_mat == 0] = negative_weight
            loss = torch.mean(dot_products * weight_mat)
        else:
            raise NotImplementedError('invalid similarity function: ' + self.sim_type)
        return loss

    def forward_similarity_train(self, model, tokenizer, batch,
                                 type_embeds=None, type_counter=None,
                                 extra_module_info=None, tag_to_loss_weight=None, prototypes=None):
        if extra_module_info is None:
            extra_module_info = []
        train_target_pos = self.config['train']['train_target_pos']  # ALERT: need to be confirmed
        loss_vec = None
        if self.config['task'] in ['contrastive', 'continual']:
            if self.model_name in ['cpm', 'llama']:
                raise NotImplementedError('not implemented for CPM or LLaMA')
            batch_tag_set = [tag.item() for tag in batch['tags']]
            hidden_mask = self.get_hidden_mask(batch, tokenizer)
            outputs = model(
                input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=None,
                output_hidden_states=True, return_dict=True)
            past_tag_set = [tag for tag, cnt in enumerate(type_counter) if cnt > 0] if type_counter is not None else []
            if self.sim_type != 'divergence':
                if self.config['plm']['model_name'].startswith('t5'):
                    logits = outputs.decoder_hidden_states[-1][:, train_target_pos, :]
                else:
                    logits = outputs['hidden_states'][-1][hidden_mask]
            else:
                if self.config['plm']['model_name'].startswith('t5'):
                    logits = outputs.logits[:, train_target_pos, :]
                else:
                    logits = outputs['logits'][hidden_mask]
            batch_sz, hidden_sz = logits.shape
            left_logits = logits[torch.arange(0, batch_sz, 2), :]
            right_logits = logits[torch.arange(1, batch_sz, 2), :]
            loss = self.forward_by_sim_mat(left_logits, right_logits, batch['sim_mat'])
            assert all(tag not in past_tag_set for tag in batch_tag_set)
            past_weight = self.config['train']['past_weight'] if 'past_weight' in self.config['train'] else 0
            if len(past_tag_set) > 0 and past_weight > 0:
                compare_tags = complex_sample(past_tag_set, size=batch_sz, replace=True)
                compare_logits = torch.stack([type_embeds[tag] for tag in compare_tags])
                compare_sim_mat = torch.LongTensor([self.negative_label for _ in range(batch_sz)]).to(batch['sim_mat'])
                compare_loss = self.forward_by_sim_mat(logits, compare_logits, compare_sim_mat)
                loss += compare_loss * past_weight
        elif self.config['task'] == 'fewshot':
            _, loss, loss_vec = self.generate_continual_logits('train', model, tokenizer, batch,
                                                               extra_module_info, tag_to_loss_weight,
                                                               prototypes=prototypes)
            if self.config['train']['train_expert_selector'] == 1:
                # train expert selector
                _, loss_selector, t_loss_vec = self.generate_expert_selector_logits(
                    'train', model, tokenizer, batch, tag_to_loss_weight)
                loss += loss_selector * self.config['train']['expert_selector_factor']
                assert len(loss_vec) == len(t_loss_vec)
                loss_vec = [loss_vec[i] + t_loss_vec[i] for i in range(len(loss_vec))]
        else:
            raise NotImplementedError('invalid task name: ' + self.config['task'])
        if self.continual_method == 'ewc' and not self.config['generate_grad']:
            assert len(extra_module_info) == 0
            ewc_loss = self.calculate_ewc_loss(model)
            loss += ewc_loss
        return loss, loss_vec

    def calculate_ewc_loss(self, model):
        assert self.continual_method == 'ewc'
        p_lambda = self.config['train']['ewc_lambda']
        if 'grad_means' not in self.split_to_model_cache:
            assert 'grad_fishers' not in self.split_to_model_cache
            grad_checkpoints = [s for s in self.config['grad_checkpoint'].split(',') if s]
            grad_means, grad_fishers = [], []
            for grad_checkpoint in grad_checkpoints:
                print(f'loading {grad_checkpoint} ......')
                grad_mean, grad_fisher = torch.load(grad_checkpoint, map_location='cpu')
                grad_mean = {n: p.to(self.config['device']) for n, p in grad_mean.items()}
                grad_fisher = {n: p.to(self.config['device']) for n, p in grad_fisher.items()}
                grad_means.append(grad_mean)
                grad_fishers.append(grad_fisher)
            self.split_to_model_cache.update({'grad_means': grad_means, 'grad_fishers': grad_fishers})
        else:
            assert 'grad_fishers' in self.split_to_model_cache
        grad_means, grad_fishers = \
            self.split_to_model_cache['grad_means'], self.split_to_model_cache['grad_fishers']
        assert len(grad_means) == len(grad_fishers)
        params = {n.replace('.', '__'): p for n, p in model.named_parameters() if p.requires_grad}
        ewc_loss = torch.tensor(0., dtype=self.float_type).to(self.config['device'])
        # loop over all previous contexts as each context has separate penalty term
        for mean, fisher in zip(grad_means, grad_fishers):
            assert mean.keys() == fisher.keys() == params.keys()
            for n in mean.keys():
                ewc_loss += 0.5 * (fisher[n] * (params[n] - mean[n]) ** 2).sum()
        ewc_loss *= p_lambda
        return ewc_loss

    def calculate_lwf_loss(self, sample_keys, whole_logits):
        assert self.continual_method == 'lwf'
        past_splits = [f'p{idx}' for idx in range(1, self.curr_bound)]  # only consider old tasks
        last_target_tokens, _ = self.generate_continual_token_map(past_splits, strict=True)
        if 'lwf_logit' not in self.split_to_model_cache:
            grad_checkpoints = [s for s in self.config['grad_checkpoint'].split(',') if s]
            assert len(grad_checkpoints) == 1
            print(f'loading lwf logit from {grad_checkpoints[0]} ......')
            lwf_logit = torch.load(grad_checkpoints[0], map_location='cpu')
            lwf_logit = {key: torch.softmax(val.to(self.config['device'])[last_target_tokens], dim=0)
                         for key, val in lwf_logit.items()}
            self.split_to_model_cache['lwf_logit'] = lwf_logit
        lwf_logit = self.split_to_model_cache['lwf_logit']
        last_lwf_logit = torch.stack([lwf_logit[sample_key] for sample_key in sample_keys])

        cur_lwf_logit = whole_logits[:, last_target_tokens]
        lwf_loss = (- last_lwf_logit * torch.log_softmax(cur_lwf_logit, dim=1)).sum()
        lwf_loss *= self.config['train']['ewc_lambda']
        assert not torch.isnan(lwf_loss)
        return lwf_loss

    def generate_continual_experts(self, model, extra_module_info):
        split_to_tags = GLOBAL['continual_split_to_tags']
        assert model is None or isinstance(model, (BertLoRAWithSelector, CPMLoRAWithSelector, LlamaLoRAWithSelector))
        for split, state in extra_module_info:
            if split in self.split_to_model_cache:
                continue
            linear_dim = len(split_to_tags[split]) if self.use_linear else -1
            if self.model_name not in ['cpm', 'llama']:
                split_model = BertLoRAWithSelector(self.config, with_selector=False,
                                                   linear_dim=linear_dim, linear_dim_exp=-1).to(self.config['device'])
                split_model.resize_token_embeddings(model.config.vocab_size)
                bert_state = {key: val for key, val in state.items() if key.startswith('bert')}
                err_msg = split_model.bert_lora.load_state_dict(bert_state, strict=False)
                assert len(err_msg.unexpected_keys) == 0 and all('lora' not in key and 'adapter' not in key for key in err_msg.missing_keys)
                if self.use_linear:
                    prefix = 'lora_linear_out.'
                    linear_state = {key[len(prefix):]: val for key, val in state.items() if key.startswith(prefix)}
                    split_model.lora_linear_out.load_state_dict(linear_state, strict=True)
                    print(f'loaded linear weight for {split} ......')
                split_model.zero_grad()
                split_model.eval()
                self.split_to_model_cache[split] = split_model
            else:
                prefix = 'backbone.'
                split_state_dict = {key[len(prefix):]: val for key, val in state.items() if key.startswith(prefix)}
                if self.use_linear:
                    split_linear = nn.Linear(self.config['hidden_size'], linear_dim, dtype=torch.half)
                    prefix = 'lora_projector.'
                    linear_state = {key[len(prefix):]: val for key, val in state.items() if key.startswith(prefix)}
                    split_linear.load_state_dict(linear_state, strict=True)
                    split_linear.zero_grad()
                    split_linear.eval()
                    split_linear = split_linear.to(self.config['infer_device'])
                    # print(f'loaded linear weight for {split} ......')
                    model.add_delta(split, split_state_dict, split_linear)
                else:
                    raise NotImplementedError('CPM without linear is not implemented')

    def generate_continual_token_map(self, splits: list, strict=False):
        """
        ALERT: for self.continual_method not in ['our', 'our_abl', 'our_sim_pro', 'lwf'], the result is different from
        simply changing the splits to ['p1'-total_parts]!
        """
        if not strict and self.continual_method not in ['our', 'our_abl', 'our_sim_pro', 'lwf']:
            label_num = self.config['label_num']
            target_tokens = list(range(label_num))
            token_to_tid = {idx: tag for idx, tag in enumerate(target_tokens)}
            return target_tokens, token_to_tid
        if not strict and self.continual_method == 'lwf':
            splits = [f'p{sid+1}' for sid in range(self.curr_bound)]
        continual_split_to_targets = GLOBAL['continual_split_to_tags']
        target_tokens, token_to_tid = [], {}
        for split in splits:
            cur_targets = continual_split_to_targets[split]
            for wid in cur_targets:
                token_to_tid[wid] = len(target_tokens)
                target_tokens.append(wid)
        return target_tokens, token_to_tid

    def get_current_splits(self, extra_module_info, train_expert_selector):
        if extra_module_info is None:
            extra_module_info = []
        if train_expert_selector:
            assert self.config['train']['train_expert_selector'] and \
                   self.config['train']['verbalizer_strategy'] == 'mean'
            expert_selector_groups = [f'p{idx}' for idx in range(1, self.curr_bound + 1)]
            return expert_selector_groups, expert_selector_groups
        current_splits = [f'p{self.curr_bound}']
        extra_special_part = [s for s in self.config['dataset']['extra_special_part'].split(',') if s]
        for split in extra_special_part:
            if split not in current_splits:
                current_splits.append(split)
        past_split_set = [info[0] for info in extra_module_info]
        for current_split in current_splits:
            assert all(current_split != split for split in past_split_set)
        current_splits.sort()
        total_split_set = past_split_set + current_splits
        if self.config['is_test']:  # ALERT: for all is_test
            for idx in range(1, self.curr_bound + 1):
                if f'p{idx}' not in total_split_set:
                    total_split_set.append(f'p{idx}')
        total_split_set.sort()
        return total_split_set, current_splits

    def forward_model_logits(self, split: str, linear_layer, batch, model, hidden_mask, is_selector: bool):
        if hidden_mask.device != batch['input_ids'].device:
            hidden_mask = hidden_mask.to(batch['input_ids'].device)
        if self.model_name not in ['cpm', 'llama']:
            forward_model = model.bert_selector if is_selector else model.bert_lora
            forward_linear = model.lora_linear_selector if is_selector else model.lora_linear_out
            representation = forward_model(
                input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                output_hidden_states=True, return_dict=True)['hidden_states'][-1][hidden_mask]
            if model.lora_alignment is not None:
                representation = model.lora_alignment(representation)
            logits = forward_linear(representation)
        else:
            if model.is_infer:
                linear_layer = model.load_delta(split)
            input_batch = {key: val for key, val in batch.items() if key not in ['tags', 'sample_keys']}
            representation = model(**input_batch)[hidden_mask]
            logits = linear_layer(representation)
        return logits, representation

    @torch.no_grad()
    def select_topk_experts(self, mode, model, tokenizer, batch, k=1):
        total_splits = [f'p{idx}' for idx in range(1, self.curr_bound + 1)]
        hidden_mask = self.get_hidden_mask(batch, tokenizer)

        # if p2, do not use topk experts
        assert self.curr_bound != 1
        if k >= len(total_splits):
            return [total_splits for _ in range(len(batch['sample_keys']))]

        assert self.config['train']['train_expert_selector'] >= 1
        assert isinstance(model, (BertLoRAWithSelector, CPMLoRAWithSelector, LlamaLoRAWithSelector))
        if self.config['train']['train_expert_selector'] == 1:
            raise NotImplementedError('wait for implementation')

        split_to_tags, tag_to_split = GLOBAL['continual_split_to_tags'], GLOBAL['continual_tag_to_split']
        preds, tags = [], batch['tags'].cpu().tolist()

        assert self.model_name not in ['cpm', 'llama'] or model.is_infer
        logits, _ = self.forward_model_logits('selector', None, batch, model, hidden_mask, is_selector=True)
        assert len(logits) == len(tags)
        for logit, tag in zip(logits, tags):
            cur_probs = []
            for split in total_splits:
                split_label = int(split[1:]) - 1
                cur_probs.append((split_label, logit[split_label]))
            cur_probs = sorted(cur_probs, key=lambda x: x[1], reverse=True)
            assert len(cur_probs) >= k
            cur_preds = [f'p{split_tag + 1}' for split_tag, _ in cur_probs[:k]]
            if mode == 'train' and self.config['train']['teacher_forcing']:
                # must choose the true expert
                true_split = tag_to_split[tag]
                if true_split not in cur_preds:
                    cur_preds = cur_preds[:-1] + [true_split]
            preds.append(cur_preds)
        return preds

    def generate_expert_selector_logits(self, mode, model, tokenizer, batch, tag_to_loss_weight=None):
        tag_to_split, split_to_tags = GLOBAL['continual_tag_to_split'], GLOBAL['continual_split_to_tags']
        total_splits, current_splits = self.get_current_splits([], train_expert_selector=True)
        batch_sz = batch['input_ids'].shape[0]

        # forward by split
        hidden_mask = self.get_hidden_mask(batch, tokenizer)
        assert self.model_name not in ['cpm', 'llama'] or not model.is_infer
        exp_linear_out = model.lora_linear_selector if self.model_name not in ['cpm', 'llama'] else model.lora_exp_projector
        logits, _ = self.forward_model_logits('', exp_linear_out, batch, model, hidden_mask, is_selector=True)
        logits = torch.log_softmax(logits, dim=1)

        if mode != 'train':
            return logits

        # generate verb_mat
        verb_mat = []
        assert tag_to_loss_weight is not None
        for tag in batch['tags']:
            tag, word_ids = tag.item(), []
            split_label = int(tag_to_split[tag][1:]) - 1
            cur_prob = torch.tensor([0. for _ in range(len(total_splits))], dtype=self.float_type)
            cur_prob[split_label] = 1.
            verb_mat.append(cur_prob * tag_to_loss_weight[tag])
        verb_mat = torch.stack(verb_mat).to(self.config['device'])
        flat_loss = torch.sum(logits * verb_mat, dim=1)
        loss = - torch.sum(flat_loss) / batch_sz
        assert not (torch.isnan(loss) or torch.isinf(loss))
        return logits, loss, flat_loss.cpu().tolist()

    def convert_infer_batch(self, batch):
        if self.model_name not in ['cpm', 'llama']:
            infer_batch = batch
        else:
            infer_batch = {}
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    val = val.to(self.config['infer_device'])
                infer_batch[key] = val
        return infer_batch

    def generate_continual_logits(self, mode, model, tokenizer, batch,
                                  extra_module_info=None, tag_to_loss_weight=None,
                                  lwf_logit=False, prototypes=None):
        if extra_module_info is None:
            extra_module_info = []
        # check split_to_model_cache
        infer_model = model if self.model_name not in ['cpm', 'llama'] else GLOBAL['infer_model']
        self.generate_continual_experts(infer_model, extra_module_info)

        tag_to_split, split_to_tags = GLOBAL['continual_tag_to_split'], GLOBAL['continual_split_to_tags']
        total_splits, current_splits = self.get_current_splits(extra_module_info, train_expert_selector=False)
        target_tokens, token_to_tid = self.generate_continual_token_map(total_splits)
        batch_sz, target_sz = batch['input_ids'].shape[0], len(target_tokens)
        if self.continual_method == 'lwf':
            target_sz = self.config['label_num']

        if self.config['train']['use_expert_selector']:
            assert self.continual_method.startswith('our') and len(current_splits) == 1
            # user expert selector as filter
            key_set = [f'{key}_es' for key in batch['sample_keys']]
            if all(key in self.continual_logit_cache for key in key_set):
                selected_experts = [self.continual_logit_cache[key] for key in key_set]
            else:
                selected_experts = self.select_topk_experts(
                    mode, infer_model, tokenizer, self.convert_infer_batch(batch), self.config['train']['expert_topk'])
                assert len(key_set) == len(selected_experts)
                for key, experts in zip(key_set, selected_experts):
                    self.continual_logit_cache[key] = experts
            split_groups = {}
            for bid, experts in enumerate(selected_experts):
                for expert in experts:
                    if expert not in split_groups:
                        split_groups[expert] = []
                    split_groups[expert].append(bid)
        else:
            extra_splits = [info[0] for info in extra_module_info]
            selector_splits = [f'p{idx}' for idx in range(1, self.curr_bound + 1)]
            assert not self.continual_method.startswith('our') or extra_splits == selector_splits[:-1]
            split_groups = {split: list(range(batch_sz)) for split in selector_splits}

        # forward by split
        hidden_mask = self.get_hidden_mask(batch, tokenizer)
        # low enough score
        logits = torch.full((batch_sz, target_sz), self.negative_logit,
                            dtype=self.float_type, requires_grad=True).to(self.config['device'])
        for split, bids in split_groups.items():
            if not self.continual_method.startswith('our') or split in current_splits:
                continue
            split_targets = split_to_tags[split]
            split_tids = [token_to_tid[target] for target in split_targets]

            with torch.no_grad():
                loc_batch = {}
                for key, val in batch.items():
                    if isinstance(val, torch.Tensor):
                        val = val[bids]
                    loc_batch[key] = val
                sample_keys = [batch['sample_keys'][bid] for bid in bids]
                loc_hidden_mask = hidden_mask[bids, :]

                assert self.continual_method.startswith('our')
                cache_results = [read_cache(split, sample_key, self.np_float_type) for sample_key in sample_keys]
                if all(res is not None for res in cache_results):
                    loc_outputs = torch.tensor(cache_results, dtype=self.float_type).to(logits)
                else:
                    split_model = self.split_to_model_cache[split] \
                        if self.model_name not in ['cpm', 'llama'] else infer_model
                    assert self.model_name not in ['cpm', 'llama'] or split_model.is_infer
                    loc_outputs, _ = self.forward_model_logits(
                        split, None, self.convert_infer_batch(loc_batch), split_model,
                        loc_hidden_mask, is_selector=False)
                    loc_outputs = loc_outputs.to(self.config['device'])
                    assert loc_outputs.shape[0] == len(sample_keys)
                    for sample_key, logit in zip(sample_keys, loc_outputs):
                        write_cache(split, sample_key, logit.cpu().numpy(), dtype=self.np_float_type)
                temp_logits = logits[bids, :]
                temp_logits[:, split_tids] = loc_outputs
                logits[bids, :] = temp_logits

        assert self.continual_method.startswith('emr') or len(current_splits) == 1
        current_targets = split_to_tags[current_splits[0]]
        current_tids = [token_to_tid[target] for target in current_targets]
        outputs = None
        assert self.model_name not in ['cpm', 'llama'] or not model.is_infer
        model_linear_out = model.lora_linear_out if self.model_name not in ['cpm', 'llama'] else model.lora_projector
        if not self.continual_method.startswith('our'):
            assert not self.config['train']['use_expert_selector']
            outputs, representation = self.forward_model_logits(
                '', model_linear_out, batch, model, hidden_mask, is_selector=False)
            if prototypes is None:
                logits = outputs
            else:
                rep = representation.view(representation.shape[0], 1, representation.shape[-1])
                proto = prototypes.view(1, -1, prototypes.shape[-1])
                logits = (rep * proto).sum(-1)

        elif not self.config['train']['use_expert_selector'] or current_splits[0] in split_groups:
            bids = split_groups[current_splits[0]]
            loc_batch = {}
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    val = val[bids]
                loc_batch[key] = val
            loc_hidden_mask = hidden_mask[bids, :]
            outputs, _ = self.forward_model_logits(
                '', model_linear_out, loc_batch, model, loc_hidden_mask, is_selector=False)
            temp_logits = logits[bids, :]
            temp_logits[:, current_tids] = outputs
            logits[bids, :] = temp_logits
        else:
            assert self.config['dataset']['batch_limit_policy'] == 0 or \
                   not self.config['train']['teacher_forcing'] or mode != 'train'
        logits = torch.log_softmax(logits, dim=1)
        if mode != 'train':
            return outputs if lwf_logit else logits

        # generate verb_mat
        verb_mat = []
        assert tag_to_loss_weight is not None
        for tag in batch['tags']:
            tag = tag.item()
            cur_prob = torch.tensor([0. for _ in range(target_sz)], dtype=self.float_type)
            cur_prob[token_to_tid[tag]] = 1.
            verb_mat.append(cur_prob * tag_to_loss_weight[tag])
        verb_mat = torch.stack(verb_mat).to(self.config['device'])
        flat_loss = torch.sum(logits * verb_mat, dim=1)
        loss = - torch.sum(flat_loss) / batch_sz
        assert not (torch.isnan(loss) or torch.isinf(loss))

        if self.continual_method == 'lwf' and self.curr_bound != 1:
            lwf_loss = self.calculate_lwf_loss(batch['sample_keys'], outputs)
            loss += lwf_loss

        return outputs if lwf_logit else logits, loss, flat_loss.cpu().tolist()

    @torch.no_grad()
    def generate_inference_logits(self, model, tokenizer, batch):
        inference_target_pos = self.config['train']['inference_target_pos']  # ALERT: need to be confirmed
        if self.config['plm']['model_name'].startswith('t5'):
            outputs = model.generate(input_ids=batch['input_ids'], max_length=4, output_scores=True,
                                     output_hidden_states=True, return_dict_in_generate=True)
            if self.sim_type in ['cosine', 'sigmoid_distance', 'dot']:
                logits = outputs.decoder_hidden_states[inference_target_pos][-1].squeeze(1)
            elif self.sim_type == 'divergence':
                logits = outputs.scores[inference_target_pos]
            else:
                raise NotImplementedError('invalid similarity function: ' + self.sim_type)
        else:
            outputs = model(
                input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                output_hidden_states=True, return_dict=True)
            hidden_mask = self.get_hidden_mask(batch, tokenizer)
            if self.sim_type in ['cosine', 'sigmoid_distance', 'dot']:
                logits = outputs['hidden_states'][-1][hidden_mask]
            elif self.sim_type == 'divergence':
                logits = outputs['logits'][hidden_mask]
            else:
                raise NotImplementedError('invalid similarity function: ' + self.sim_type)
        return logits

    @torch.no_grad()
    def forward_similarity_test(self, logit, type_embeds, target_ones=None):
        if self.sim_type == 'cosine':
            losses = self.loss_func(logit, type_embeds, target_ones)
        elif self.sim_type == 'divergence':
            losses = self.loss_func(logit, type_embeds)
        elif self.sim_type == 'sigmoid_distance':
            vec_distance = torch.square(logit - type_embeds)
            losses = torch.sigmoid(self.extra_module(vec_distance).squeeze(1))
            # losses = torch.sum(vec_distance, dim=1)
        elif self.sim_type == 'dot':
            losses = 200 - self.loss_func(logit, type_embeds)
        else:
            raise NotImplementedError('invalid similarity function: ' + self.sim_type)
        return losses

    @torch.no_grad()
    def forward_select_continual_sample(self, model, tokenizer, batch, write_db=False):
        hidden_mask = self.get_hidden_mask(batch, tokenizer)
        model_linear_out = model.lora_linear_out if self.model_name not in ['cpm', 'llama'] else model.lora_projector
        target_logits, _ = self.forward_model_logits(
            '', model_linear_out, batch, model, hidden_mask, is_selector=False)
        target_logits = target_logits.cpu().numpy()
        batch_tags, sample_keys = batch['tags'].cpu().tolist(), batch['sample_keys']
        assert len(target_logits) == len(batch_tags) == len(sample_keys)

        cur_split = self.config['dataset']['special_part']
        for key, tag, state in zip(sample_keys, batch_tags, target_logits):
            if write_db:
                write_cache(cur_split, key, state, self.np_float_type)
                continue
            if tag not in self.continual_logit_cache:
                self.continual_logit_cache[tag] = []
            self.continual_logit_cache[tag].append((key, state))

    def obtain_select_continual_sample(self):
        selected_sample_keys = {'train_infer': set(), 'valid_groups': set()}
        for tag, samples in tqdm(self.continual_logit_cache.items()):
            cur_tag_samples_keys = []
            sample_keys = [sample[0] for sample in samples]
            sample_states = np.stack([sample[1] for sample in samples])
            num_clusters = min(self.config['sample_num'], len(sample_keys))
            print(f'clustering tag {tag} sample number {len(sample_keys)}')
            distances = KMeans(n_clusters=num_clusters,
                               random_state=self.config['dataset']['seed']).fit_transform(sample_states)
            for cid in range(num_clusters):
                sel_index = np.argmin(distances[:, cid])
                cur_tag_samples_keys.append(sample_keys[sel_index])
            train_samples = complex_sample(range(len(cur_tag_samples_keys)),
                                           size=int(np.ceil(len(cur_tag_samples_keys) * 0.8)), replace=False)
            train_samples = set(train_samples)
            for sid in range(len(cur_tag_samples_keys)):
                if sid in train_samples:
                    selected_sample_keys['train_infer'].add(cur_tag_samples_keys[sid])
                else:
                    selected_sample_keys['valid_groups'].add(cur_tag_samples_keys[sid])
        selected_sample_keys['valid_groups'] -= selected_sample_keys['train_infer']
        return selected_sample_keys
