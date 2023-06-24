import argparse


class Param:
    """
    Detailed hyperparameter configurations.
    """
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser = self.all_param(parser)
        all_args, unknown = parser.parse_known_args()
        self.args = all_args

    @staticmethod
    def all_param(parser):

        # #################################common parameters####################################
        # parser.add_argument("--gpu", default=0, type=int)

        parser.add_argument("--dataset", default='ontonotes', type=str,
                            choices=['ace', 'bbn', 'fewnerd', 'fewrel', 'ontonotes', 'tacred'])

        # parser.add_argument("--task_name", default='FewRel', type=str)
        # parser.add_argument("--max_length", default=256, type=int)  # should be modified
        # parser.add_argument("--task_length", default=10, type=int)  # should be modified

        parser.add_argument("--log_path", required=True, type=str)

        parser.add_argument("--load_path", default="", type=str)

        parser.add_argument("--this_name", default="continual", type=str)

        parser.add_argument("--device", default="cuda:7", type=str)

        # ##############################   training ################################################

        parser.add_argument("--batch_size", default=8, type=int)

        parser.add_argument("--learning_rate", default=2e-4, type=float)

        parser.add_argument("--total_round", default=1, type=int)  # ALERT: modified!!!

        parser.add_argument("--pattern", default="entity_marker") 

        parser.add_argument("--encoder_output_size", default=1024, type=int)

        # parser.add_argument("--vocab_size", default=30522, type=int)

        # parser.add_argument("--marker_size", default=4, type=int)

        # Temperature parameter in CL and CR
        parser.add_argument("--temp", default=0.1, type=float)

        # the sample_k used in utils.py
        parser.add_argument("--sample_k", default=1000, type=int)

        # the minimum number of samples for each tag in CustomLoader
        parser.add_argument("--min_sample_num", default=10, type=int)

        # The projection head outputs dimensions
        parser.add_argument("--feat_dim", default=64, type=int)

        # Temperature parameter in KL
        parser.add_argument("--kl_temp", default=10, type=float)

        parser.add_argument("--num_workers", default=0, type=int)

        # epoch1
        parser.add_argument("--step1_epochs", default=-1, type=int)
        # epoch2
        parser.add_argument("--step2_epochs", default=-1, type=int)

        parser.add_argument("--seed", default=100, type=int)

        parser.add_argument("--max_grad_norm", default=10, type=float)

        parser.add_argument("--not_apply_lora", action="store_true")

        # Memory size
        # parser.add_argument("--num_protos", default=20, type=int)

        parser.add_argument("--optim", default='adamw', type=str)

        # dataset path
        parser.add_argument("--data_path", default='data/', type=str)

        # bert-base-uncased weights path
        parser.add_argument("--bert_path", default="decapoda-research/llama-7b-hf", type=str)

        parser.add_argument("--replay", action="store_true")

        parser.add_argument("--num_threads", default=8, type=int)

        parser.add_argument("--dynamic_sampling", action="store_true")

        return parser
