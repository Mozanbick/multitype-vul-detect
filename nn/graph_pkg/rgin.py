import os
import time
import torch as th
import dgl
import argparse
import torch.nn.functional as F
import torch.nn as nn

from configs import modelConfig as ModelConfig
from nn.graph_pkg import Logger
from dgl.dataloading import GraphDataLoader
from utils.objects.dataset import GraphDataset
from nn.graph_pkg.models import RGINModel
from torch.utils.data import random_split
from torchmetrics import Accuracy, F1Score, Precision, Recall

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
global_train_time_per_epoch = []


def arg_parse():
    parser = argparse.ArgumentParser(description='R-GCN arguments')
    parser.add_argument('--dataset', dest='dataset', help='Input Dataset')
    parser.add_argument('--no-train', dest='no_train', action='store_const',
                        const=True, default=False, help='skip the training phase')
    parser.add_argument('--cuda', dest='cuda', type=int, help='switch cuda')
    parser.add_argument('--lr', dest='lr', type=float, help='learning rate')
    parser.add_argument('--batch-size', dest='batch_size', type=int, help='batch size')
    parser.add_argument('--epochs', dest='epochs', type=int, help='num-of-epoch')
    parser.add_argument('--train-ratio', dest='train_ratio', type=float,
                        help='ratio of training dataset split')
    parser.add_argument('--ext-dtest', dest='ext_test', action='store_const',
                        const=True, default=False,
                        help='split testing dataset within training dataset')
    parser.add_argument('--dtest-ratio', dest='test_ratio', type=float,
                        help='ratio of testing dataset split')
    parser.add_argument('--num-workers', dest='n_worker', type=int,
                        help='number of workers when data loading')
    parser.add_argument('--n-hidden-layers', dest='n_hidden_layers', type=int,
                        help='number of hidden graph conv layers per batch graph')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int,
                        help='dimension of hidden layers')
    parser.add_argument('--dropout', dest='dropout', type=float, help='dropout rate')
    parser.add_argument('--bias', dest='bias', action='store_const',
                        const=True, default=False, help='switch for bias')
    parser.add_argument('--learn-eps', dest='learn_eps', action='store_const',
                        const=True, default=False, help='learn eps while training')
    parser.add_argument('--num-bases', dest='num_bases', type=int, help='number of bases')
    parser.add_argument(
        '--train-dir',
        dest='train_dir',
        help='train dataset saving directory'
    )
    parser.add_argument(
        '--dtest-dir',
        dest='test_dir',
        help='dtest dataset saving directory'
    )
    parser.add_argument(
        '--save-dir',
        dest='save_dir',
        help='model saving directory: SAVE_DICT/DATASET'
    )
    parser.add_argument('--load-epoch', dest='load_epoch', type=int, help='load trained model params from\
                             SAVE_DICT/DATASET/model-LOAD_EPOCH')

    parser.set_defaults(
        dataset='fan',
        no_train=False,
        cuda=-1,
        lr=0.01,
        batch_size=20,
        epochs=5,
        train_ratio=0.8,
        ext_test=False,
        test_ratio=0.1,
        n_worker=1,
        n_hidden_layers=3,
        hidden_dim=128,
        dropout=0.0,
        bias=False,
        num_bases=-1,
        train_dir="input/dataset/train",
        test_dir="input/dataset/dtest",
        save_dir="nn/graph_pkg/model_param/rgin",
        load_epoch=-1
    )
    return parser.parse_args()


def make_run_id(model_name: str, task_name: str, run_name=None) -> str:
    """Choose a run ID, based on the --run-name parameter and the current time."""
    if run_name is not None:
        return run_name
    else:
        return "%s_%s__%s" % (model_name, task_name, time.strftime("%Y-%m-%d_%H-%M-%S"))


def prepare_data(dataset, prog_args, train=False, pre_process=None):
    """
    preprocess dataset and load dataset into dataloader
    """
    if train:
        shuffle = True
    else:
        shuffle = False

    if pre_process:
        pre_process(dataset, prog_args)

    return GraphDataLoader(
        dataset,
        batch_size=prog_args.batch_size,
        shuffle=shuffle,
        num_workers=prog_args.n_worker
    )


def train(dataset, model, args, log, val_dataset=None):
    save_dir = args.save_dir + "/" + args.dataset
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    dataloader = dataset

    # loss function, optimizer and scheduler
    loss_fn = nn.CrossEntropyLoss()
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    early_stopping_logger = {"best_epoch": -1, "val_acc": -1}
    if args.cuda >= 0:
        th.cuda.set_device(args.cuda)
    # training loop
    for epoch in range(args.epochs):
        begin_time = time.time()
        model.train()
        accum_correct = 0
        total = 0
        total_loss = 0
        log("\nEPOCH ###### {} ######".format(epoch))
        computation_time = 0
        batch = 0
        for (batch_idx, (batch_graph, graph_labels)) in enumerate(dataloader):
            batch = batch_idx
            for (key, value) in batch_graph.ndata.items():
                batch_graph.ndata[key] = value.float()
            graph_labels = graph_labels.long()
            feat = batch_graph.ndata['feat']
            if args.cuda >= 0 and th.cuda.is_available():
                batch_graph = batch_graph.to(th.cuda.current_device())
                graph_labels = graph_labels.to(th.cuda.current_device())

            # model.zero_grad()
            compute_start = time.time()
            ypred = model(batch_graph, feat)
            indices = th.argmax(ypred, dim=1)
            correct = th.sum(indices == graph_labels).item()
            accum_correct += correct
            total += graph_labels.size()[0]
            loss = loss_fn(ypred, graph_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_compute_time = time.time() - compute_start
            computation_time += batch_compute_time
        scheduler.step()
        train_accu = accum_correct / total
        log("train accuracy for this epoch {} is {:.2f}%".format(epoch, train_accu * 100))
        elapsed_time = time.time() - begin_time
        log("loss {:.4f} with epoch time {:.4f} s & computation time {:.4f} s ".format(
            total_loss / (batch + 1), elapsed_time, computation_time))
        global_train_time_per_epoch.append(elapsed_time)
        if val_dataset is not None:
            result = evaluate(val_dataset, model, args)
            log("Validation : Accuracy {:.2f}% | F1 score {:.2f}% | Precision {:.2f}% | Recall {:.2f}%".format(
                result[0] * 100, result[1] * 100, result[2] * 100, result[3] * 100))
            if early_stopping_logger['val_acc'] <= result[0] <= train_accu or epoch == 0:
                early_stopping_logger.update(best_epoch=epoch, val_acc=result[0])
                if args.save_dir is not None:
                    th.save(model.state_dict(), args.save_dir + "/" + args.dataset
                            + "/model.iter-" + str(early_stopping_logger['best_epoch']))
            log("best epoch is EPOCH {}, val_acc is {:.2f}%".format(early_stopping_logger['best_epoch'],
                                                                    early_stopping_logger['val_acc'] * 100))
        th.cuda.empty_cache()
    return early_stopping_logger


def evaluate(dataloader, model, args, logger=None):
    if logger and args.save_dir:
        model.load_state_dict(th.load(args.save_dir + "/" + args.dataset
                                      + "/model.iter-" + str(logger['best_epoch'])))
    model.eval()
    # correct_label = 0
    test_acc = Accuracy()
    test_f1 = F1Score(average='macro', num_classes=2)
    test_pre = Precision(average='macro', num_classes=2)
    test_rec = Recall(average='macro', num_classes=2)
    with th.no_grad():
        for batch_idx, (batch_graph, graph_labels) in enumerate(dataloader):
            for (key, value) in batch_graph.ndata.items():
                batch_graph.ndata[key] = value.float()
            graph_labels = graph_labels.long()
            feat = batch_graph.ndata['feat']
            if args.cuda >= 0 and th.cuda.is_available():
                batch_graph = batch_graph.to(th.cuda.current_device())
                graph_labels = graph_labels.cuda()
                test_acc = test_acc.to(th.cuda.current_device())
                test_f1 = test_f1.to(th.cuda.current_device())
                test_pre = test_pre.to(th.cuda.current_device())
                test_rec = test_rec.to(th.cuda.current_device())
            ypred = model(batch_graph, feat)
            indices = th.argmax(ypred, dim=1)
            # correct = torch.sum(indices == graph_labels)
            # correct_label += correct.item()
            acc = test_acc(indices, graph_labels)
            f1 = test_f1(indices, graph_labels)
            pre = test_pre(indices, graph_labels)
            rec = test_rec(indices, graph_labels)
    # result = correct_label / (len(dataloader) * prog_args.batch_size)
    acc = test_acc.compute()
    # f1 = test_f1.compute()
    precision = test_pre.compute()
    recall = test_rec.compute()
    f1 = 2 * precision * recall / (precision + recall)
    return [acc, f1, precision, recall]


def graph_classify_task(args):
    run_id = make_run_id(f'Train-R-GIN_{args.dataset}', 'GraphBinaryClassify')
    log_file = os.path.join(args.save_dir, f"{run_id}.log")
    log = Logger(log_file)
    log(str(args))

    dataset = GraphDataset(args.dataset, args.train_dir)
    dataset.load()

    if not args.ext_test:
        trian_size = int(args.train_ratio * len(dataset))
        test_size = int(args.test_ratio * len(dataset))
        val_size = int(len(dataset) - trian_size - test_size)

        dataset_train, dataset_val, dataset_test = random_split(
            dataset, (trian_size, val_size, test_size)
        )
    else:
        trian_size = int(args.train_ratio * len(dataset))
        val_size = int(len(dataset) - trian_size)

        dataset_train, dataset_val = random_split(
            dataset, (trian_size, val_size)
        )

        dataset_test = GraphDataset(args.dataset, args.test_dir)
        dataset_test.load()

    train_dataloader = prepare_data(dataset_train, args, train=True)
    val_dataloader = prepare_data(dataset_val, args, train=False)
    test_dataloader = prepare_data(dataset_test, args, train=False)
    in_dim, feat_dim, out_dim, max_num_node = dataset.statistics()
    log("++++++++++STATISTICS ABOUT THE DATASET")
    log(f"dataset node dimension is {in_dim}")
    log(f"dataset node feature dimension is {feat_dim}")
    log(f"dataset label dimension is {out_dim}")
    log(f"the max num node is {max_num_node}")
    log(f"number of graphs is {len(dataset)}")

    hidden_dim = args.hidden_dim
    log("++++++++++MODEL STATISTICS++++++++")
    log(f"model hidden dim is {hidden_dim}")
    activation = F.relu

    # initial model
    model = RGINModel(
        feat_dim,
        hidden_dim,
        out_dim,
        len(ModelConfig.list_etypes),
        num_layers=args.n_hidden_layers,
        num_bases=args.num_bases,
        dropout=args.dropout,
        learn_eps=args.learn_eps
    )
    if args.load_epoch >= 0 and args.save_dir is not None:
        model.load_state_dict(th.load(args.save_dir + '/' + args.dataset
                                      + "/model.iter-" + str(args.load_epoch)))
    log("model init finished")
    log("MODEL::::::::::R-GIN")
    if args.cuda >= 0 and th.cuda.is_available():
        model = model.cuda()

    logger = train(
        train_dataloader,
        model,
        args,
        log,
        val_dataset=val_dataloader
    )
    result = evaluate(test_dataloader, model, args, logger=logger)
    log("Test : Accuracy {:.2f}% | F1 score {:.2f}% | Precision {:.2f}% | Recall {:.2f}%".format(
        result[0] * 100, result[1] * 100, result[2] * 100, result[3] * 100))

    log(
        "Train time per epoch: {:.4f}".format(sum(global_train_time_per_epoch) / len(global_train_time_per_epoch)))
    log("Max memory usage: {:.4f}".format(th.cuda.max_memory_allocated(0) / (1024 * 1024)))


def graph_classify_test(args):
    run_id = make_run_id('Test-R-GCN', 'GraphBinaryClassify')
    log_file = os.path.join(args.save_dir, f"{run_id}.log")
    log = Logger(log_file)
    log(str(args))

    dataset_test = GraphDataset(args.dataset, args.train_dir)
    dataset_test.load()
    test_dataloader = prepare_data(dataset_test, args, train=False)

    in_dim, feat_dim, out_dim, max_num_node = dataset_test.statistics()
    log("++++++++++STATISTICS ABOUT THE DATASET")
    log(f"dataset node dimension is {in_dim}")
    log(f"dataset node feature dimension is {feat_dim}")
    log(f"dataset label dimension is {out_dim}")
    log(f"the max num node is {max_num_node}")
    log(f"number of graphs is {len(dataset_test)}")

    hidden_dim = args.hidden_dim
    log("++++++++++MODEL STATISTICS++++++++")
    log(f"model hidden dim is {hidden_dim}")

    # initial model
    model = RGINModel(
        feat_dim,
        hidden_dim,
        out_dim,
        len(ModelConfig.list_etypes),
        num_layers=args.n_hidden_layers,
        num_bases=args.num_bases,
        dropout=args.dropout,
        learn_eps=args.learn_eps
    )
    if args.load_epoch >= 0 and args.save_dir is not None:
        model.load_state_dict(th.load(args.save_dir + '/' + args.dataset
                                      + "/model.iter-" + str(args.load_epoch)))
    log("model init finished")
    log("MODEL::::::::::R-GCN")

    if args.cuda >= 0 and th.cuda.is_available():
        model = model.cuda()

    result = evaluate(test_dataloader, model, args)
    log("Test : Accuracy {:.2f}% | F1 score {:.2f}% | Precision {:.2f}% | Recall {:.2f}%".format(
        result[0] * 100, result[1] * 100, result[2] * 100, result[3] * 100))


def main():
    args = arg_parse()
    # print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if args.no_train:
        graph_classify_test(args)
    else:
        graph_classify_task(args)


if __name__ == '__main__':
    main()
