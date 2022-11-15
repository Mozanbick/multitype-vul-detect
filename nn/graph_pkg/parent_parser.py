import argparse


def parent_parser():
    parser = argparse.ArgumentParser(description="training arguments", add_help=False)
    parser.add_argument(
        'model',
        type=str,
        choices=[
            'rgcn', 'rgin', 'sagpool'
        ],
        help='Choose a graph model'
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
        model='',
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

    return parser
