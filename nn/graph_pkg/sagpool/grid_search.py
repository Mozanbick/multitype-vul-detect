import os
import sys
import json
from copy import deepcopy
path = os.path.abspath(os.path.join(os.getcwd(), '../../../'))
sys.path.append(path)

from train import arg_parser as parent_parser
from nn.graph_pkg.sagpool.utils import get_stats
from nn.graph_pkg.sagpool_cli import graph_classify_task, arg_parser


def load_config(path='./grid_search_config.json'):
    with open(path, "r") as fp:
        return json.load(fp)


def run_experiments(args):
    res = []
    for i in range(args.num_trails):
        print("Trail {}/{}".format(i + 1, args.num_trails))
        logger = graph_classify_task(args)
        res.append(logger['val_acc'])

    mean, err_bd = get_stats(res, conf_interval=True)
    return mean, err_bd


def grid_search(config: dict):
    parser, _ = parent_parser()
    args = arg_parser(parser)

    best_acc, err_bd = 0.0, 0.0
    best_args = vars(args)
    for arch in config['arch']:
        args.architecture = arch
        for hidden in config['hidden']:
            args.hidden_dim = hidden
            for pool_ratio in config['pool_ratio']:
                args.pool_ratio = pool_ratio
                for lr in config['lr']:
                    args.lr = lr
                    for weight_decay in config['weight_decay']:
                        args.weight_decay = weight_decay
                        acc, bd = run_experiments(args)
                        if acc > best_acc:
                            best_acc = acc
                            err_bd = bd
                            best_args = deepcopy(vars(args))
    args.output_path = "./output"
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    args.output_path = "./output/{}.log".format(args.dataset)
    result = {
        "params": best_args,
        "result": "{:.4f}({:.4f})".format(best_acc, err_bd),
    }
    with open(args.output_path, "w") as fp:
        json.dump(result, fp, sort_keys=True, indent=4)


grid_search(load_config())
