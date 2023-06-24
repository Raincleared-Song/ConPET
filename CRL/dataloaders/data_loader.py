import torch
from torch.utils.data import Dataset, DataLoader


class data_set(Dataset):

    def __init__(self, data, config=None):
        self.data = data
        self.config = config
        self.bert = True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], idx

    def collate_fn(self, data):
        labels = torch.IntTensor([item[0]['relation'] for item in data])

        # pad tokens
        tokens = [item[0]['tokens'].tolist() for item in data]
        max_len = max(len(token) for token in tokens)
        new_tokens, attention_mask = [], []
        for token in tokens:
            attention_mask.append(torch.IntTensor([1] * len(token) + [0] * (max_len - len(token))))
            new_tokens.append(torch.IntTensor(token + [self.config.pad_token_id] * (max_len - len(token))))
        tokens = new_tokens

        mask_ids = [token == self.config.mask_token_id for token in tokens]
        assert all(torch.sum(mask_id) == 1 for mask_id in mask_ids)
        ind = torch.LongTensor([item[1] for item in data])
        assert len(labels) == len(tokens) == len(mask_ids) == len(ind)
        tokens, attention_mask = torch.stack(tokens, dim=0), torch.stack(attention_mask, dim=0)
        mask_ids = torch.stack(mask_ids, dim=0).to(dtype=torch.bool)
        return labels, tokens, attention_mask, mask_ids, ind


def get_data_loader(config, data, shuffle=False, drop_last=False, batch_size=None):
    dataset = data_set(data, config)

    if batch_size is None:
        batch_size = min(config.batch_size, len(data))
    else:
        batch_size = min(batch_size, len(data))

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=config.num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=drop_last)

    return data_loader
