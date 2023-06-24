import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from dataloaders import get_data_loader, CustomLoader, DataSampler
from .model import Encoder
from .utils import Moment, dot_dist, osdist
import loralib as lora
from typing import Optional
from global_var import get_learning_rate


class ManagerDynamic(object):
    def __init__(self, args):
        super().__init__()
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

    @staticmethod
    def get_tag_to_protos(args, encoder, custom_loader):
        # aggregate the prototype set for further use.
        tag_to_features = {}
        encoder.eval()
        for step, batch_data in enumerate(custom_loader):
            labels, tokens, attention_mask, mask_ids, ind = batch_data
            tokens = tokens.to(args.device)
            attention_mask = attention_mask.to(args.device)
            mask_ids = mask_ids.to(args.device)
            with torch.no_grad():
                feature, rep = encoder.bert_forward(tokens, attention_mask, mask_ids)
            features = feature.detach()
            assert len(labels) == len(features)
            for tag, feature in zip(labels, features):
                tag = int(tag)
                if tag not in tag_to_features:
                    tag_to_features[tag] = []
                tag_to_features[tag].append(feature)
        for tag, features in tag_to_features.items():
            tag_to_features[tag] = torch.mean(torch.stack(features), dim=0)
        return tag_to_features

    def get_optimizer(self, args, encoder, step: int):
        self.print_log('Use {} optim!'.format(args.optim))

        if args.optim == 'adam':
            pytorch_optim = optim.Adam
        elif args.optim == 'adamw':
            from .adamw import AdamW
            pytorch_optim = AdamW
        else:
            raise NotImplementedError
        args.learning_rate = get_learning_rate(args.big_model, args.dataset, step)
        print('Learning Rate:', args.learning_rate)
        optimizer = pytorch_optim(
            encoder.parameters(), lr=args.learning_rate, weight_decay=0.01, betas=(0.9, 0.999), eps=0.0001
        )
        return optimizer

    def train_simple_model(self, args, encoder, training_data, epochs, step):
        data_loader = get_data_loader(args, training_data, shuffle=True)
        encoder.train()
        if not args.not_apply_lora:
            lora.mark_only_lora_as_trainable(encoder)

        optimizer = self.get_optimizer(args, encoder, step)

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
                torch.cuda.empty_cache()
            self.print_log(f"{name} loss is {np.array(losses).mean()}")

        for epoch_i in range(epochs):
            train_data(data_loader, "init_train_{}".format(epoch_i), is_mem=False)

    def train_mem_model(self, args, encoder, custom_loader, proto_mem, epochs, seen_relations, history_rel_num, step):
        assert epochs == 1 and isinstance(custom_loader, CustomLoader)
        # history_nums = len(seen_relations) - args.rel_per_task
        if len(proto_mem) > 0:
            proto_mem = F.normalize(proto_mem, p=2, dim=1)
            dist = dot_dist(proto_mem, proto_mem)
            dist = dist.to(args.device)

        encoder.train()
        if not args.not_apply_lora:
            lora.mark_only_lora_as_trainable(encoder)
        temp_rel2id = [self.rel2id[x] for x in seen_relations]
        map_relid2tempid = {k: v for v, k in enumerate(temp_rel2id)}
        optimizer = self.get_optimizer(args, encoder, step)

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

                # need_ratio_compute marks the positions of old samples?
                need_ratio_compute = torch.ones(ind.shape, dtype=torch.bool)
                for lab in temp_rel2id[history_rel_num:]:
                    need_ratio_compute = torch.logical_and(need_ratio_compute, torch.not_equal(cpu_labels, lab))
                total_need = need_ratio_compute.sum()  # should be > 0
                if total_need <= 0:
                    from IPython import embed
                    embed()
                    exit()

                try:
                    # Knowledge Distillation for Relieve Forgetting
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

                if step % args.batch_limit == 0:
                    self.print_log(f"{name} loss is {np.array(losses).mean()}")

                torch.cuda.empty_cache()

        for epoch_i in range(epochs):
            train_data(custom_loader, "memory_train_{}".format(epoch_i), is_mem=True)

    @staticmethod
    def kl_div_loss(x1, x2, t=10):

        batch_dist = F.softmax(t * x1, dim=1)
        temp_dist = F.log_softmax(t * x2, dim=1)
        loss = F.kl_div(temp_dist, batch_dist, reduction="batchmean")
        return loss

    @torch.no_grad()
    def evaluate_strict_model(self, args, encoder, test_data, protos4eval, seen_relations, sampler: DataSampler):
        data_loader = get_data_loader(args, test_data, batch_size=args.batch_size)
        encoder.eval()
        temp_rel2id = [self.rel2id[x] for x in seen_relations]
        map_relid2tempid = {k: v for v, k in enumerate(temp_rel2id)}
        seen_relation_ids = [self.rel2id[relation] for relation in seen_relations]
        seen_relation_ids = [map_relid2tempid[x] for x in seen_relation_ids]
        result_list = [[0, 0] for _ in range(sampler.task_length)]
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
            if not args.not_apply_lora:
                lora.mark_only_lora_as_trainable(encoder)

            # load data and start computation
            train_data_seen = []
            history_relation = []
            proto4repaly = []

            # load checkpoint
            if load_checkpoint is not None:
                test_ave_cur, test_tot_cur = load_checkpoint['test_ave_cur'], load_checkpoint['test_tot_cur']
                test_ave_his, test_tot_his = load_checkpoint['test_ave_his'], load_checkpoint['test_tot_his']
                err_msg = encoder.load_state_dict(load_checkpoint['encoder'], strict=False)
                assert len(err_msg.unexpected_keys) == 0 and \
                       all('lora' not in key for key in err_msg.missing_keys)
                history_relation = load_checkpoint['history_relation']
                proto4repaly = load_checkpoint['proto4repaly'].to(args.device)
                train_data_seen = load_checkpoint['train_data_seen']

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
                cur_split_id = steps + 1

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
                self.moment.init_moment(args, encoder, datasets=train_data_for_initial, is_memory=False)
                # step1, baseline training

                # use contrastive loss, compare the samples to the stored pre-calculated features
                self.train_simple_model(args, encoder, train_data_for_initial, args.step1_epochs, steps)

                for relation in current_relations:
                    train_data_seen += training_data[relation]

                # replay with custom sampler
                custom_loader: Optional[CustomLoader] = None
                if cur_split_id > 1:
                    custom_loader = CustomLoader(args, cur_split_id, train_data_seen, args.step2_epochs,
                                                 args.batch_size, shuffle=True, num_workers=4,
                                                 drop_last=False, seen_relations=seen_relations)

                    # update mem_features
                    self.moment.init_moment(args, encoder, data_loader=custom_loader, is_memory=True)

                    torch.cuda.empty_cache()
                    # step2: distilling, only train for 1 epoch
                    self.train_mem_model(args, encoder, custom_loader, proto4repaly, 1, seen_relations, history_rel_num, steps)

                proto_loader = custom_loader if custom_loader is not None else get_data_loader(
                    args, train_data_for_initial, shuffle=False)
                tag_to_protos = self.get_tag_to_protos(args, encoder, proto_loader)
                protos4eval = []
                try:
                    for relation in history_relation:
                        # calculate the protos for those not in current_relations
                        protos4eval.append(tag_to_protos[self.rel2id[relation]])
                except KeyError as err:
                    print(err)
                    from IPython import embed
                    embed()
                    exit()
                protos4eval = torch.stack(protos4eval).detach()
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

                cur_res_dict = self.evaluate_strict_model(args, encoder, test_data_1, protos4eval,
                                                          seen_relations, sampler)
                his_res_dict = self.evaluate_strict_model(args, encoder, test_data_2, protos4eval,
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
                    'encoder': lora.lora_state_dict(encoder) if not args.not_apply_lora else encoder.state_dict(),
                    'history_relation': history_relation,
                    'proto4repaly': proto4repaly,
                    'train_data_seen': train_data_seen,
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
