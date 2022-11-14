import os
import sys
import json
import argparse
from copy import deepcopy
path = os.path.abspath(os.path.join(os.getcwd(), '../../../'))
sys.path.append(path)

from nn.graph_pkg.sagpool.utils import get_stats
from nn.graph_pkg.sagpool_cli import graph_classify_task, arg_parser


def parent_parser():
    parser = argparse.ArgumentParser(description="training arguments", add_help=False)
    parser.add_argument(
        '-m',
        dest='model',
        choices=[
            'rgcn', 'rgin', 'sagpool'
        ],
        help='Choose a model'
    )
    parser.add_argument('--dataset', dest='dataset', help='Input Dataset')
    parser.add_argument('--no-train', dest='no_train', action='store_const',
                        const=True, default=False, help='skip the training phase')
    parser.add_argument('--cuda', dest='cuda', type=int, help='switch cuda')
    parser.add_argument('--lr', dest='lr', type=float, help='learning rate')
    parser.add_argument('--batch-size', dest='batch_size', type=int, help='batch size')
    parser.add_argument('--epochs', dest='epochs', type=int, help='num-of-epoch')
    parser.add_argument('--train-ratio', dest='train_ratio', type=float,
                        help='ratio of training dataset split')
    parser.add_argument('--ext-test', dest='ext_test', action='store_const',
                        const=True, default=False,
                        help='split testing dataset within training dataset')
    parser.add_argument('--test-ratio', dest='test_ratio', type=float,
                        help='ratio of testing dataset split')
    parser.add_argument(
        '--weight-decay',
        dest='weight_decay',
        type=float,
        help='weight decay'
    )
    parser.add_argument(
        '--patience',
        dest='patience',
        type=int,
        help='patience for early stopping, `-1` means no early stopping'
    )
    parser.add_argument('--num-workers', dest='n_worker', type=int,
                        help='number of workers when data loading')
    parser.add_argument(
        '--train-dir',
        dest='train_dir',
        help='train dataset saving directory'
    )
    parser.add_argument(
        '--test-dir',
        dest='test_dir',
        help='test dataset saving directory'
    )
    parser.add_argument(
        '--save-dir',
        dest='save_dir',
        help='model saving directory: SAVE_DICT/DATASET'
    )
    parser.add_argument('--load-epoch', dest='load_epoch', type=int, help='load trained model params from\
                                 SAVE_DICT/DATASET/model-LOAD_EPOCH')

    parser.set_defaults(
        model='sagpool',
        dataset='fan',
        no_train=False,
        cuda=-1,
        lr=0.01,
        batch_size=20,
        epochs=5,
        train_ratio=0.8,
        ext_test=False,
        test_ratio=0.1,
        weight_decay=1e-4,
        patience=-1,
        n_worker=1,
        train_dir="input/dataset/train",
        test_dir="input/dataset/test",
        save_dir="nn/graph_pkg/model_param",
        load_epoch=-1
    )

    return parser, parser.parse_args()


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
        for batch_size in config['batch_size']:
            args.batch_size = batch_size
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
