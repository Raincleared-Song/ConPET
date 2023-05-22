import mkl
import torch
from config import Param
from methods.utils import setup_seed
from methods.manager import Manager
from global_var import get_epoch_map, get_batch_size_map, get_learning_rate


def run(args):
    setup_seed(args.seed)
    print("hyper-parameter configurations:")
    print(str(args.__dict__))

    manager = Manager(args)
    manager.train(args)


def main():
    param = Param()  # There are detailed hyperparameter configurations.
    args = param.args
    torch.cuda.set_device(args.device)
    args.device = torch.device(args.device)
    args.n_gpu = torch.cuda.device_count()
    # args.task_name = args.dataset

    args.big_model = 'llama' in args.bert_path or 'cpm' in args.bert_path

    # modify sequence length
    max_seq_len_map = {
        'ace': 128, 'bbn': 256, 'fewnerd': 256,
        'fewrel': 128, 'ontonotes': 256, 'tacred': 128,
    }
    args.max_length = max_seq_len_map[args.dataset]

    # modify task length
    task_length_map = {
        'ace': 5, 'bbn': 10, 'fewnerd': 10,
        'fewrel': 10, 'ontonotes': 10, 'tacred': 10,
    }
    args.task_length = task_length_map[args.dataset]

    # modify num_protos
    num_protos_map = {
        'ace': 20, 'bbn': 50, 'fewnerd': 100,
        'fewrel': 50, 'ontonotes': 100, 'tacred': 20,
    }
    args.num_protos = num_protos_map[args.dataset]

    # modify epoch_num
    num_epochs = get_epoch_map(args.big_model)[args.dataset]
    if args.step1_epochs == -1:
        args.step1_epochs = num_epochs
    if args.step2_epochs == -1:
        args.step2_epochs = num_epochs
    args.batch_size = get_batch_size_map(args.big_model)[args.dataset]
    args.learning_rate = get_learning_rate(args.big_model, args.dataset)

    mkl.set_num_threads(args.num_threads)
    run(args)


if __name__ == '__main__':
    main()
