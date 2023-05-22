import os
import json
import math
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
from tqdm import tqdm
from dataloaders.data_loader import get_data_loader
from dataloaders.sampler import DataSampler
from .model import Encoder
from .utils import Moment, dot_dist, osdist
import loralib as lora
from global_var import get_batch_size_map


def get_emr_replay_frequency(dataset_name: str, memory_train_size: int, big_model: bool):
    if memory_train_size == 0:
        return 1
    batch_size_map = get_batch_size_map(big_model)
    batch_limit_map = {
        'fewnerd': 2500,
        'ontonotes': 1250,
        'bbn': 500,
        'fewrel': 400,
        'tacred': 100,
        'ace': 100,
    }
    base_old_frequency_map = {
        'fewnerd': 10,
        'ontonotes': 10,
        'bbn': 5,
        'fewrel': 5,
        'tacred': 5,
        'ace': 5,
    }
    batch_limit_train = int(math.ceil(batch_limit_map[dataset_name] * 0.8))
    room_for_old_samples = batch_limit_train * batch_size_map[dataset_name]
    if room_for_old_samples <= 0:
        return base_old_frequency_map[dataset_name]
    replay_frequency = room_for_old_samples / memory_train_size
    replay_frequency = int(math.ceil(replay_frequency))
    assert replay_frequency > 0
    return replay_frequency


class Manager(object):
    def __init__(self, args):
        super().__init__()
        self.lbs = None
        self.moment = None
        self.id2rel = None
        self.rel2id = None
        self.torch_dtype = torch.half if 'llama' in args.bert_path else torch.float
        self.numpy_dtype = np.float16 if 'llama' in args.bert_path else np.float32
        self.args = args
        self.log_file = open('logs/' + args.log_path + '.log', 'a', encoding='utf-8')

    def print_log(self, *args, **kwargs):
        print(*args, **kwargs)
        kwargs['file'] = self.log_file
        kwargs['flush'] = True
        print(*args, **kwargs)

    def get_proto(self, args, encoder, mem_set):
        # aggregate the prototype set for further use.
        data_loader = get_data_loader(args, mem_set, False, False, 1)

        features = []

        encoder.eval()
        for step, batch_data in enumerate(data_loader):
            labels, tokens, attention_mask, mask_ids, ind = batch_data
            tokens = tokens.to(args.device)
            attention_mask = attention_mask.to(args.device)
            mask_ids = mask_ids.to(args.device)
            with torch.no_grad():
                feature, rep = encoder.bert_forward(tokens, attention_mask, mask_ids)
            features.append(feature.detach())
            self.lbs.append(labels.item())
        features = torch.cat(features, dim=0)

        proto = torch.mean(features, dim=0, keepdim=True)

        return proto, features

    @staticmethod
    # Use K-Means to select what samples to save, similar to at_least = 0
    def select_data(args, encoder, sample_set):
        data_loader = get_data_loader(args, sample_set, shuffle=False, drop_last=False, batch_size=1)
        features = []
        encoder.eval()
        for step, batch_data in enumerate(data_loader):
            labels, tokens, attention_mask, mask_ids, ind = batch_data
            tokens = tokens.to(args.device)
            attention_mask = attention_mask.to(args.device)
            mask_ids = mask_ids.to(args.device)
            with torch.no_grad():
                feature, rp = encoder.bert_forward(tokens, attention_mask, mask_ids)
            features.append(feature.detach().cpu())

        features = np.concatenate(features)
        num_clusters = min(args.num_protos, len(sample_set))

        while not os.path.exists('flag'):
            time.sleep(random.randint(5, 20))
        os.remove('flag')
        distances = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(features)
        with open('flag', 'w'):
            pass

        # the number of chosen samples is num_protos - 20
        mem_set = []
        current_feat = []
        for k in range(num_clusters):
            sel_index = np.argmin(distances[:, k])
            instance = sample_set[sel_index]
            mem_set.append(instance)
            current_feat.append(features[sel_index])

        current_feat = np.stack(current_feat, axis=0)
        current_feat = torch.from_numpy(current_feat)
        return mem_set, current_feat, current_feat.mean(0)

    def get_optimizer(self, args, encoder):
        self.print_log('Use {} optim!'.format(args.optim))

        # def set_param(module, lr):
        #     parameters_to_optimize = list(module.named_parameters())
        #     no_decay = ['undecay']
        #     parameters_to_optimize = [
        #         {'params': [p for n, p in parameters_to_optimize
        #                     if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': lr},
        #         {'params': [p for n, p in parameters_to_optimize
        #                     if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': lr}
        #     ]
        #     return parameters_to_optimize

        # params = set_param(encoder, args.learning_rate)

        if args.optim == 'adam':
            pytorch_optim = optim.Adam
        elif args.optim == 'adamw':
            pytorch_optim = optim.AdamW
        else:
            raise NotImplementedError
        optimizer = pytorch_optim(
            encoder.parameters(), lr=args.learning_rate, weight_decay=0.01, betas=(0.9, 0.999), eps=0.0001
        )
        return optimizer

    def train_simple_model(self, args, encoder, training_data, epochs):
        data_loader = get_data_loader(args, training_data, shuffle=True)
        encoder.train()
        lora.mark_only_lora_as_trainable(encoder)

        optimizer = self.get_optimizer(args, encoder)

        def train_data(data_loader_, name="train_simple", is_mem=False):
            losses = []
            td = tqdm(data_loader_, desc=name)
            for step, batch_data in enumerate(td):
                optimizer.zero_grad()
                labels, tokens, attention_mask, mask_ids, ind = batch_data
                tokens = tokens.to(args.device)
                attention_mask = attention_mask.to(args.device)
                mask_ids = mask_ids.to(args.device)
                labels = labels.to(args.device)
                hidden, reps = encoder.bert_forward(tokens, attention_mask, mask_ids)
                loss = self.moment.loss(reps, labels)
                if torch.sum(torch.isnan(loss)) > 0:
                    print('simple nan loss ......')
                    from IPython import embed
                    embed()
                    exit()
                losses.append(loss.item())
                td.set_postfix(loss=np.array(losses).mean())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
                optimizer.step()
                # update moemnt
                if is_mem:
                    self.moment.update_mem(ind, reps.detach().cpu())
                else:
                    self.moment.update(ind, reps.detach().cpu())
            self.print_log(f"{name} loss is {np.array(losses).mean()}")

        for epoch_i in range(epochs):
            train_data(data_loader, "init_train_{}".format(epoch_i), is_mem=False)

    def train_mem_model(self, args, encoder, mem_data, proto_mem, epochs, seen_relations,
                        history_rel_num, replay_frequency: int = 1):
        # history_nums = len(seen_relations) - args.rel_per_task
        if len(proto_mem) > 0:
            proto_mem = F.normalize(proto_mem, p=2, dim=1)
            dist = dot_dist(proto_mem, proto_mem)
            dist = dist.to(args.device)

        mem_loader = get_data_loader(args, mem_data, shuffle=True)
        encoder.train()
        lora.mark_only_lora_as_trainable(encoder)
        temp_rel2id = [self.rel2id[x] for x in seen_relations]
        map_relid2tempid = {k: v for v, k in enumerate(temp_rel2id)}
        # map_tempid2relid = {k: v for k, v in map_relid2tempid.items()}
        optimizer = self.get_optimizer(args, encoder)

        def train_data(data_loader_, name="train_mem", is_mem=False):
            losses = []
            kl_losses = []
            td = tqdm(data_loader_, desc=name)
            for step, batch_data in enumerate(td):

                optimizer.zero_grad()
                cpu_labels, tokens, attention_mask, mask_ids, ind = batch_data
                tokens = tokens.to(args.device)
                attention_mask = attention_mask.to(args.device)
                mask_ids = mask_ids.to(args.device)
                labels = cpu_labels.to(args.device)
                zz, reps = encoder.bert_forward(tokens, attention_mask, mask_ids)
                hidden = reps

                need_ratio_compute = ind < history_rel_num * args.num_protos
                for lab in temp_rel2id[history_rel_num:]:
                    need_ratio_compute = torch.logical_and(need_ratio_compute, torch.not_equal(cpu_labels, lab))
                total_need = need_ratio_compute.sum()

                if total_need > 0:
                    try:
                        # Knowledge Distillation for Relieve Forgetting
                        # need_ind = ind[need_ratio_compute]
                        need_labels = labels[need_ratio_compute]
                        temp_labels = [map_relid2tempid[x.item()] for x in need_labels]
                        gold_dist = dist[temp_labels]
                        current_proto = self.moment.get_mem_proto()[:history_rel_num]
                        this_dist = dot_dist(hidden[need_ratio_compute], current_proto.to(args.device))
                        loss1 = self.kl_div_loss(gold_dist, this_dist, t=args.kl_temp)
                        loss1.backward(retain_graph=True)
                    except Exception as err:
                        print(err)
                        from IPython import embed
                        embed()
                        exit()
                    if torch.sum(torch.isnan(loss1)) > 0:
                        print('mem nan loss1 ......')
                        from IPython import embed
                        embed()
                        exit()
                else:
                    loss1 = 0.0

                #  Contrastive Replay
                cl_loss = self.moment.loss(reps, labels, is_mem=True)

                if isinstance(loss1, float):
                    kl_losses.append(loss1)
                else:
                    kl_losses.append(loss1.item())
                loss = cl_loss
                if torch.sum(torch.isnan(loss)) > 0:
                    print('mem nan loss ......')
                    from IPython import embed
                    embed()
                    exit()
                if isinstance(loss, float):
                    losses.append(loss)
                    td.set_postfix(loss=np.array(losses).mean(), kl_loss=np.array(kl_losses).mean())
                    # update moemnt
                    if is_mem:
                        self.moment.update_mem(ind, reps.detach().cpu(), hidden.detach().cpu())
                    else:
                        self.moment.update(ind, reps.detach().cpu())
                    continue
                losses.append(loss.item())
                td.set_postfix(loss=np.array(losses).mean(), kl_loss=np.array(kl_losses).mean())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
                optimizer.step()

                # update moemnt
                if is_mem:
                    self.moment.update_mem(ind, reps.detach().cpu())
                else:
                    self.moment.update(ind, reps.detach().cpu())
            self.print_log(f"{name} loss is {np.array(losses).mean()}")

        for epoch_i in range(epochs * replay_frequency):
            train_data(mem_loader, "memory_train_{}".format(epoch_i), is_mem=True)

    @staticmethod
    def kl_div_loss(x1, x2, t=10):

        batch_dist = F.softmax(t * x1, dim=1)
        temp_dist = F.log_softmax(t * x2, dim=1)
        loss = F.kl_div(temp_dist, batch_dist, reduction="batchmean")
        return loss

    @torch.no_grad()
    def evaluate_strict_model(self, args, encoder, test_data, protos4eval, _, seen_relations, sampler: DataSampler):
        data_loader = get_data_loader(args, test_data, batch_size=args.batch_size)
        encoder.eval()
        temp_rel2id = [self.rel2id[x] for x in seen_relations]
        map_relid2tempid = {k: v for v, k in enumerate(temp_rel2id)}
        seen_relation_ids = [self.rel2id[relation] for relation in seen_relations]
        seen_relation_ids = [map_relid2tempid[x] for x in seen_relation_ids]
        result_list = [[0, 0] for _ in range(sampler.total_split)]
        for step, batch_data in tqdm(enumerate(data_loader), desc='test', total=len(data_loader)):
            cpu_labels, tokens, attention_mask, mask_ids, ind = batch_data
            tokens = tokens.to(args.device)
            attention_mask = attention_mask.to(args.device)
            mask_ids = mask_ids.to(args.device)
            labels = cpu_labels.to(args.device)
            hidden, reps = encoder.bert_forward(tokens, attention_mask, mask_ids)
            labels = [map_relid2tempid[x.item()] for x in labels]
            logits = -osdist(hidden, protos4eval)
            seen_sim = logits[:, seen_relation_ids]
            seen_sim = seen_sim.cpu().data.numpy()
            max_smi = np.max(seen_sim, axis=1)
            logits = logits.cpu().data.numpy()
            assert len(max_smi) == len(seen_sim) == len(cpu_labels) == len(logits) == len(labels)
            for cur_max, cur_label, ori_label, logit in zip(max_smi, labels, cpu_labels, logits):
                ori_sid = sampler.rel2split[ori_label.item()]
                cur_pred = logit[cur_label]
                if cur_pred >= cur_max:
                    # correct prediction
                    result_list[ori_sid][0] += 1
                result_list[ori_sid][1] += 1
        group_num = sum(instance_num > 0 for correct_num, instance_num in result_list)
        assert group_num > 0
        average_acc = sum((correct_num * 100 / instance_num) for correct_num, instance_num in result_list
                          if instance_num > 0) / group_num
        total_correct_num = sum(correct_num for correct_num, instance_num in result_list)
        total_instance_num = sum(instance_num for correct_num, instance_num in result_list)
        result_dict = {
            'splits': result_list,
            'average_accuracy': round(average_acc, 2),
            'total_accuracy': round(total_correct_num * 100 / total_instance_num, 2),
        }
        return result_dict

    def train(self, args):
        load_checkpoint, skip_round, skip_step = None, -1, -1
        if args.load_path != "":
            load_checkpoint = torch.load(args.load_path, map_location='cpu')
            skip_round, skip_step = load_checkpoint['round'], load_checkpoint['step']
            if skip_step != args.task_length - 1:
                skip_round -= 1
            else:
                skip_step = -1
        # set training batch
        for i in range(args.total_round):
            if i <= skip_round:
                assert load_checkpoint is not None
                continue

            test_ave_cur, test_tot_cur = [], []
            test_ave_his, test_tot_his = [], []
            # set random seed
            random.seed(args.seed + i * 100)

            # sampler setup
            sampler = DataSampler(args=args, seed=args.seed + i * 100)
            self.id2rel = sampler.id2rel
            self.rel2id = sampler.rel2id
            # encoder setup
            encoder = Encoder(args=args).to(device=args.device, dtype=self.torch_dtype)
            lora.mark_only_lora_as_trainable(encoder)

            # initialize memory and prototypes
            # num_class = len(sampler.id2rel)
            memorized_samples = {}

            # load data and start computation
            history_relation = []
            proto4repaly = []

            # load checkpoint
            if load_checkpoint is not None:
                test_cur, test_total = load_checkpoint['test_cur'], load_checkpoint['test_total']
                err_msg = encoder.load_state_dict(load_checkpoint['encoder'], strict=False)
                assert len(err_msg.unexpected_keys) == 0 and \
                       all('lora' not in key for key in err_msg.missing_keys)
                memorized_samples = load_checkpoint['memorized_samples']
                history_relation = load_checkpoint['history_relation']
                proto4repaly = load_checkpoint['proto4repaly'].to(args.device)

                prefix = 'manager.'
                for key, val in load_checkpoint.items():
                    if key.startswith(prefix):
                        setattr(self, key[len(prefix):], val)

                del load_checkpoint
                load_checkpoint = None

            for steps, (training_data, valid_data, test_data,
                        current_relations, historic_valid_data, historic_test_data,
                        seen_relations, history_rel_num) in enumerate(sampler):
                if steps <= skip_step:
                    continue

                self.print_log('steps:', steps, 'relations:', current_relations)
                # Initial
                train_data_for_initial = []
                for relation in current_relations:
                    history_relation.append(relation)
                    train_data_for_initial += training_data[relation]

                # train model
                # no memory. first train with current task
                self.moment = Moment(args)
                # update features
                self.moment.init_moment(args, encoder, train_data_for_initial, is_memory=False)
                # step1, baseline training

                # use contrastive loss, compare the samples to the stored pre-calculated features
                self.train_simple_model(args, encoder, train_data_for_initial, args.step1_epochs)

                # replay
                if len(memorized_samples) > 0:
                    # select current task sample
                    # select num_proto samples for each relation
                    for relation in current_relations:
                        memorized_samples[relation], _, _ = self.select_data(args, encoder, training_data[relation])

                    train_data_for_memory = []
                    for relation in history_relation:
                        train_data_for_memory += memorized_samples[relation]

                    # update mem_features
                    self.moment.init_moment(args, encoder, train_data_for_memory, is_memory=True)

                    replay_frequency = get_emr_replay_frequency(
                        args.dataset, len(train_data_for_memory), args.big_model) if args.replay else 1

                    # seen_valid_data = []
                    # for relation in seen_relations:
                    #     seen_valid_data += historic_valid_data[relation]
                    #
                    # cur_acc = self.evaluate_strict_model(args, encoder, test_data_1, protos4eval, featrues4eval,
                    #                                      seen_relations)
                    # total_acc = self.evaluate_strict_model(args, encoder, test_data_2, protos4eval, featrues4eval,
                    #                                        seen_relations)

                    torch.cuda.empty_cache()
                    # step2: distilling
                    self.train_mem_model(args, encoder, train_data_for_memory, proto4repaly, args.step2_epochs,
                                         seen_relations, history_rel_num, replay_frequency)

                feat_mem = []
                proto_mem = []

                for relation in current_relations:
                    memorized_samples[relation], feat, temp_proto = self.select_data(args, encoder,
                                                                                     training_data[relation])
                    feat_mem.append(feat)
                    proto_mem.append(temp_proto)

                # feat_mem = torch.cat(feat_mem, dim=0)
                temp_proto = torch.stack(proto_mem, dim=0)

                protos4eval = []
                featrues4eval = []
                self.lbs = []
                for relation in history_relation:
                    if relation not in current_relations:
                        # calculate the protos for those not in current_relations
                        protos, featrues = self.get_proto(args, encoder, memorized_samples[relation])
                        protos4eval.append(protos)
                        featrues4eval.append(featrues)

                if protos4eval:

                    protos4eval = torch.cat(protos4eval, dim=0).detach()
                    protos4eval = torch.cat([protos4eval, temp_proto.to(args.device)], dim=0)

                else:
                    protos4eval = temp_proto.to(args.device)
                proto4repaly = protos4eval.clone()

                test_data_1 = []
                for relation in current_relations:
                    test_data_1 += test_data[relation]
                self.print_log('current relations:', current_relations, len(current_relations))
                self.print_log('current test data:', len(test_data_1))

                test_data_2 = []
                for relation in seen_relations:
                    test_data_2 += historic_test_data[relation]
                self.print_log('seen relations:', seen_relations, len(seen_relations))
                self.print_log('seen test data:', len(test_data_2))

                cur_res_dict = self.evaluate_strict_model(args, encoder, test_data_1, protos4eval, featrues4eval,
                                                          seen_relations, sampler)
                his_res_dict = self.evaluate_strict_model(args, encoder, test_data_2, protos4eval, featrues4eval,
                                                          seen_relations, sampler)
                cur_ave_acc, cur_tot_acc = cur_res_dict['average_accuracy'], cur_res_dict['total_accuracy']
                his_ave_acc, his_tot_acc = his_res_dict['average_accuracy'], his_res_dict['total_accuracy']

                self.print_log(f'Restart Num {i + 1}')
                self.print_log(f'task--{steps + 1}:')
                self.print_log(f'current test average acc: {cur_ave_acc}')
                self.print_log(f'current test total acc: {cur_tot_acc}')
                self.print_log(f'history test average acc: {his_ave_acc}')
                self.print_log(f'history test total acc: {his_tot_acc}')
                test_ave_cur.append(cur_ave_acc)
                test_tot_cur.append(cur_tot_acc)
                test_ave_his.append(his_ave_acc)
                test_tot_his.append(his_tot_acc)

                self.print_log(test_ave_cur)
                self.print_log(test_tot_cur)
                self.print_log(test_ave_his)
                self.print_log(test_tot_his)
                del self.moment

                model_dir = f'checkpoints/{args.log_path}'
                os.makedirs(model_dir, exist_ok=True)
                model_path = f'{model_dir}/round_{i}_step_{steps}.pkl'
                save_dict = {
                    'round': i, 'step': steps,
                    'test_ave_cur': test_ave_cur, 'test_tot_cur': test_tot_cur,
                    'test_ave_his': test_ave_his, 'test_tot_his': test_tot_his,
                    'cur_res_dict': cur_res_dict, 'his_res_dict': his_res_dict,
                    'sampler': sampler.batch,
                    'encoder': lora.lora_state_dict(encoder),
                    'memorized_samples': memorized_samples,
                    'history_relation': history_relation,
                    'proto4repaly': proto4repaly,
                }
                for name in dir(self):
                    val = getattr(self, name)
                    if not name.startswith('_') and isinstance(val, (list, torch.Tensor)):
                        key = 'manager.' + name
                        save_dict[key] = val
                torch.save(save_dict, model_path)
                metrics_dict = {
                    'round': i, 'step': steps,
                    'test_ave_cur': test_ave_cur, 'test_tot_cur': test_tot_cur,
                    'test_ave_his': test_ave_his, 'test_tot_his': test_tot_his,
                    'cur_res_dict': cur_res_dict, 'his_res_dict': his_res_dict,
                }
                metrics_path = f'{model_dir}/round_{i}_step_{steps}_metric.json'
                with open(metrics_path, 'w') as fout:
                    json.dump(metrics_dict, fout)
