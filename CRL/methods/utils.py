import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from dataloaders.data_loader import get_data_loader


class Moment:
    def __init__(self, args) -> None:

        self.hidden_features = None
        self.mem_features = None
        self.features = None
        self.labels = None
        self.mem_labels = None
        self.memlen = 0
        self.sample_k = args.sample_k
        self.temperature = args.temp
        self.device = args.device
        self.torch_dtype = torch.half if 'llama' in args.bert_path else torch.float
        self.numpy_dtype = np.float16 if 'llama' in args.bert_path else np.float32

    def get_mem_proto(self):
        c = self._compute_centroids_ind()
        return c

    def _compute_centroids_ind(self):
        cinds = []
        for x in self.mem_labels:
            if x.item() not in cinds:
                cinds.append(x.item())

        num = len(cinds)
        feats = self.mem_features
        centroids = torch.zeros((num, feats.size(1)), dtype=self.torch_dtype, device=feats.device)
        for i, c in enumerate(cinds):
            ind = np.where(self.mem_labels.cpu().numpy() == c)[0]
            centroids[i, :] = normalize_half_cpu(feats[ind, :].mean(dim=0), p=2, dim=0)
        return centroids

    def update(self, ind, feature):
        self.features[ind] = feature

    def update_mem(self, ind, feature, hidden=None):
        self.mem_features[ind] = feature
        if hidden is not None:
            self.hidden_features[ind] = hidden

    @torch.no_grad()
    def init_moment(self, args, encoder, datasets=None, data_loader=None, is_memory=False):
        encoder.eval()
        if data_loader is not None:
            assert datasets is None
            data_len = data_loader.data_len
        else:
            assert datasets is not None
            data_loader = get_data_loader(args, datasets)
            data_len = len(datasets)

        if not is_memory:
            self.features = torch.zeros(data_len, args.feat_dim, dtype=self.torch_dtype)
            td = tqdm(data_loader, desc='init_mom')
            lbs = []
            for step, batch_data in enumerate(td):
                labels, tokens, attention_mask, mask_ids, ind = batch_data
                tokens = tokens.to(args.device)
                attention_mask = attention_mask.to(args.device)
                mask_ids = mask_ids.to(args.device)
                _, reps = encoder.bert_forward(tokens, attention_mask, mask_ids)
                try:
                    self.update(ind, reps.detach().cpu())
                except Exception as err:
                    print(err)
                    from IPython import embed
                    embed()
                    exit()
                lbs.append(labels)
            self.labels = torch.cat(lbs).cpu()
        else:
            self.memlen = data_len
            self.mem_features = torch.zeros(data_len, args.feat_dim, dtype=self.torch_dtype)
            self.hidden_features = torch.zeros(data_len, args.encoder_output_size, dtype=self.torch_dtype)
            lbs = []
            td = tqdm(data_loader, desc='init_mom_mem')
            for step, batch_data in enumerate(td):
                labels, tokens, attention_mask, mask_ids, ind = batch_data
                tokens = tokens.to(args.device)
                attention_mask = attention_mask.to(args.device)
                mask_ids = mask_ids.to(args.device)
                hidden, reps = encoder.bert_forward(tokens, attention_mask, mask_ids)
                self.update_mem(ind, reps.detach().cpu(), hidden.detach().cpu())
                lbs.append(labels)
            if hasattr(data_loader, 'raw_labels'):
                self.mem_labels = data_loader.raw_labels
            else:
                self.mem_labels = torch.cat(lbs).cpu()

    def loss(self, x, labels, is_mem=False, force_all=False):
        if is_mem:
            all_features, all_labels = self.mem_features, self.mem_labels
        else:
            all_features, all_labels = self.features, self.labels

        if self.sample_k is not None and not (is_mem and force_all):
            # sample some instances to calculate the contrastive loss
            idx = list(range(len(all_features)))
            passed, ct_x, ct_y = False, None, None
            while not passed:
                if len(idx) > self.sample_k:
                    sample_id = random.sample(idx, self.sample_k)
                else:
                    sample_id = idx
                ct_x = all_features[sample_id]
                ct_y = all_labels[sample_id]
                passed = True
                for lab in labels:
                    if torch.sum(torch.eq(ct_y, lab.item())) == 0:
                        passed = False
                        break
        else:
            ct_x = all_features
            ct_y = all_labels

        ct_x, ct_y = ct_x.to(self.device), ct_y.to(self.device)

        dot_product_tempered = torch.mm(x, ct_x.T) / self.temperature  # n * m
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
                torch.exp(
                    dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0].detach()) + 1e-4
        )
        mask_combined = torch.eq(labels.unsqueeze(1).repeat(1, ct_y.shape[0]), ct_y).to(self.device)  # n*m
        cardinality_per_samples = torch.sum(mask_combined, dim=1)

        # if is_mem:
        #     log_prob = -torch.log(exp_dot_tempered / torch.mean(exp_dot_tempered, dim=1, keepdim=True))
        #     supervised_contrastive_loss_per_sample = \
        #         torch.mean(log_prob * mask_combined, dim=1) * (log_prob.shape[1] / cardinality_per_samples)
        # else:
        log_prob = -torch.log(exp_dot_tempered / torch.sum(exp_dot_tempered, dim=1, keepdim=True))
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)

        if torch.isnan(supervised_contrastive_loss):
            print('nan loss inner ......')
            from IPython import embed
            embed()
            exit()

        return supervised_contrastive_loss


def dot_dist(x1, x2):
    return torch.matmul(x1, x2.t())


def osdist(x, c):
    pairwise_distances_squared = torch.sum(x ** 2, dim=1, keepdim=True) + \
                                 torch.sum(c.t() ** 2, dim=0, keepdim=True) - \
                                 2.0 * torch.matmul(x, c.t())
    error_mask = pairwise_distances_squared <= 0.0
    pairwise_distances = pairwise_distances_squared.clamp(min=1e-16)  # .sqrt()
    pairwise_distances = torch.mul(pairwise_distances, ~error_mask)
    return pairwise_distances


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def normalize_half_cpu(inputs: torch.Tensor, p: float = 2.0, dim: int = 1, eps: float = 1e-12):
    if inputs.dtype == torch.half:
        return F.normalize(inputs.to(dtype=torch.float), p=p, dim=dim, eps=eps).to(torch.half)
    else:
        return F.normalize(inputs, p=p, dim=dim, eps=eps)
