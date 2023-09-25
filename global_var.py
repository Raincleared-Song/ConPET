GLOBAL = {
    "continual_split_to_tags": None,
    "continual_tag_to_split": None,
    "infer_model": None,
}


def get_epoch_map(big_model: bool):
    if big_model:
        return {
            'fewnerd': 10,
            'ontonotes': 10,
            'bbn': 10,
            'fewrel': 10,
            'tacred': 20,
            'ace': 20,
            'chent': 10,
        }
    else:
        return {
            'fewnerd': 10,
            'ontonotes': 20,
            'bbn': 20,
            'fewrel': 10,
            'tacred': 20,
            'ace': 20,
            'chent': 20,
        }


def get_batch_limit(dataset_name: str):
    return {
        'fewnerd': 2500,
        'ontonotes': 1250,
        'bbn': 500,
        'fewrel': 400,
        'tacred': 100,
        'ace': 100,
        'chent': 1250,
    }[dataset_name]


def get_batch_size_map(big_model: bool):
    if big_model:
        return {
            'fewnerd': 8,
            'ontonotes': 8,
            'bbn': 8,
            'fewrel': 8,
            'tacred': 8,
            'ace': 8,
            'chent': 8,
        }
    else:
        return {
            'fewnerd': 16,
            'ontonotes': 16,
            'bbn': 16,
            'fewrel': 16,
            'tacred': 16,
            'ace': 16,
            'chent': 16,
        }


def get_learning_rate(big_model: bool, dataset_name: str, split_id: int, method: str, is_adapter=False):
    if big_model:
        if dataset_name == 'fewnerd':
            if method not in ['emr', 'emar', 'eaemr', 'our_abl', 'emr_abl']:
                if is_adapter:
                    return 0.0001
                else:
                    return 0.0005 if split_id <= 2 else 0.0001
            else:
                return 1.0e-5 if is_adapter else 2.0e-5
        elif dataset_name == 'ontonotes':
            return 0.0005 if split_id <= 3 else 0.0001
        elif dataset_name == 'bbn':
            return 0.0002
        elif dataset_name == 'fewrel':
            return 0.0002
        elif dataset_name == 'tacred':
            return 0.0002
        elif dataset_name == 'ace':
            return 0.0002
        else:
            raise NotImplementedError(f'invalid dataset_name: {dataset_name}')
    else:
        return {
            'fewnerd': 0.00002,
            'ontonotes': 0.0001,
            'bbn': 0.00005,
            'fewrel': 0.001,
            'tacred': 0.001,
            'ace': 0.002,
            'chent': 0.0001,
        }[dataset_name]
