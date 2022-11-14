import os
import time
import torch
import dgl
import argparse
import torch.nn.functional as F

from configs import modelConfig as ModelConfig
from nn.graph_pkg import Logger
from nn.graph_pkg.logger import make_run_id
from dgl.dataloading import GraphDataLoader
from utils.objects.dataset import GraphDataset
from torch.utils.data import random_split
from torchmetrics import Accuracy, F1Score, Precision, Recall
from nn.graph_pkg.sagpool import get_sag_network
from nn.graph_pkg.sagpool.utils import get_stats


def arg_parser(args):
    parser = argparse.ArgumentParser(parents=[args], description="Self-Attention Graph Pooling")
    parser.add_argument(
        '--pool-ratio',
        type=float,
        dest='pool_ratio',
        help='pooling ratio'
    )
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, help='hidden dimension')
    parser.add_argument(
        '--architecture',
        type=str,
        dest='architecture',
        choices=['hierarchical', 'global'],
        help='model architecture'
    )
    parser.add_argument(
        '--conv-layers',
        type=int,
        dest='conv_layers',
        help='number of conv layers'
    )
    parser.add_argument('--dropout', dest='dropout', type=float, help='dropout ratio')
    parser.add_argument(
        '--print-every',
        dest='print_every',
        type=int,
        help='print train log every k epochs, -1 for silent training'
    )
    parser.add_argument('--num-trails', dest='num_trails', type=int, help='number of trails')

    parser.set_defaults(
        pool_ratio=0.5,
        hidden_dim=128,
        architecture='hierarchical',
        conv_layers=3,
        dropout=0.5,
        print_every=10,
        num_trails=1
    )

    return parser.parse_args()


def train(model: torch.nn.Module, train_dataset, args, log, val_dataset=None):
    dataloader = train_dataset
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        model.parameters()), lr=args.lr)
    early_stopping_logger = {"best_epoch": -1, "val_acc": -1}

    last_record_epoch = 0
    last_train_loss = 0.0
    last_val_loss = 0.0
    for epoch in range(args.epochs):
        begin_time = time.time()
        model.train()
        accum_correct = 0
        total = 0
        total_loss = 0.0
        log("\nEPOCH ###### {} ######".format(epoch), step=epoch)
        total_batch = 0
        for (batch_idx, (batch_graph, graph_labels)) in enumerate(dataloader):
            for (key, value) in batch_graph.ndata.items():
                batch_graph.ndata[key] = value.float()
            graph_labels = graph_labels.long()
            if args.cuda >= 0 and torch.cuda.is_available():
                batch_graph = batch_graph.to(torch.cuda.current_device())
                graph_labels = graph_labels.to(torch.cuda.current_device())

            optimizer.zero_grad()
            yperd = model(batch_graph)
            indices = torch.argmax(yperd, dim=1)
            correct = torch.sum(indices == graph_labels).item()
            accum_correct += correct
            total += graph_labels.size()[0]
            loss = F.nll_loss(yperd, graph_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_batch += 1

        train_acc = accum_correct / total
        log("train accuracy for this epoch {} is {:.2f}%".format(epoch, train_acc * 100), step=epoch)
        elapsed_time = time.time() - begin_time
        log("loss {:.4f} with epoch time {:.4f} s ".format(total_loss / total_batch, elapsed_time), step=epoch)
        # validation or not
        if val_dataset is not None:
            result = evaluate(model, val_dataset, args, log)
            log("Validation : Accuracy {:.2f}% | F1 score {:.2f}% | Precision {:.2f}% | Recall {:.2f}%".format(
                result[0] * 100, result[1] * 100, result[2] * 100, result[3] * 100), step=epoch)
            log("loss {:.4f} in validation".format(result[4] * 100), step=epoch)
            # save model
            if early_stopping_logger['val_acc'] <= result[0] <= train_acc or epoch == 0:
                early_stopping_logger.update(best_epoch=epoch, val_acc=result[0])
                if args.save_dir is not None:
                    torch.save(model.state_dict(), args.save_dir + "/" + args.dataset
                               + "/model.iter-" + str(early_stopping_logger['best_epoch']))
                    last_record_epoch = epoch
            log("best epoch is EPOCH {}, val_acc is {:.2f}%".format(early_stopping_logger['best_epoch'],
                                                                    early_stopping_logger['val_acc'] * 100), step=epoch)
            if result[4] <= last_val_loss or epoch == 0:
                last_val_loss = result[4]
                last_record_epoch = epoch
            # early stopping
            if 0 < args.patience < epoch - last_record_epoch:
                log("early stopping at EPOCH {}, val_acc is {:.2f}%".format(
                    epoch, early_stopping_logger['val_acc'] * 100))
                break
        else:
            if total_loss / len(dataloader) <= last_train_loss or epoch == 0:
                last_train_loss = total_loss / len(dataloader)
                last_record_epoch = epoch
            # early stopping
            if 0 < args.patience < epoch - last_record_epoch:
                log("early stopping at EPOCH {}, train_acc is {:.2f}%".format(
                    epoch, train_acc * 100))

        torch.cuda.empty_cache()

    return early_stopping_logger


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataloader, args, log, logger=None):
    if logger and args.save_dir:
        model.load_state_dict(torch.load(args.save_dir + "/" + args.dataset
                                         + "/model.iter-" + str(logger['best_epoch'])))

    model.eval()
    test_acc = Accuracy()
    test_pre = Precision(average='macro', num_classes=2)
    test_rec = Recall(average='macro', num_classes=2)
    total_loss = 0.0
    total_batch = 0

    for (batch_idx, (batch_graph, graph_labels)) in enumerate(dataloader):
        for (key, value) in batch_graph.ndata.items():
            batch_graph.ndata[key] = value.float()
        graph_labels = graph_labels.long()
        if args.cuda >= 0 and torch.cuda.is_available():
            batch_graph = batch_graph.to(torch.cuda.current_device())
            graph_labels = graph_labels.to(torch.cuda.current_device())
            test_acc = test_acc.to(torch.cuda.current_device())
            test_pre = test_pre.to(torch.cuda.current_device())
            test_rec = test_rec.to(torch.cuda.current_device())
        ypred = model(batch_graph)
        indices = torch.argmax(ypred, dim=1)
        loss = F.nll_loss(ypred, graph_labels, reduction='sum')
        test_acc(indices, graph_labels)
        test_pre(indices, graph_labels)
        test_rec(indices, graph_labels)
        total_loss += loss.item()
        total_batch += 1

    acc = test_acc.compute()
    precision = test_pre.compute()
    recall = test_rec.compute()
    f1 = 2 * precision * recall / (precision + recall)
    return [acc, f1, precision, recall, total_loss / total_batch]


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


def graph_classify_task(args):
    """
    Graph binary classification task, output labels are {`0`: non-vulnerable, `1`: vulnerable}
    """
    save_dir = args.save_dir + "/" + args.dataset
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    run_id = make_run_id(f'Train-SAGPool_{args.dataset}', 'GraphBinaryClassify')
    log_file = os.path.join(args.save_dir, f"{run_id}.log")
    log = Logger(log_file, patience=args.print_every)
    log(str(args))

    dataset = GraphDataset(args.dataset, args.train_dir)
    dataset.load()

    # add self loop
    for i in range(len(dataset)):
        dataset.graph_list[i] = dgl.add_self_loop(dataset.graph_list[i])

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
    node_dim, feat_dim, out_dim, max_num_node = dataset.statistics()

    log("++++++++++STATISTICS ABOUT THE DATASET++++++++")
    log(f"dataset node dimension is {node_dim}")
    log(f"dataset node feature dimension is {feat_dim}")
    log(f"dataset label dimension is {out_dim}")
    log(f"the max num node is {max_num_node}")
    log(f"number of total graphs is {len(dataset)}")
    log(f"number of train graphs is {len(dataset_train)}")
    log("++++++++++MODEL STATISTICS++++++++")
    log(f"model hidden dim is {args.hidden_dim}")
    log(f"model conv layer number is {args.conv_layers}")
    log(f"model pool ratio is {args.pool_ratio}")

    if args.cuda >= 0:
        torch.cuda.set_device(args.cuda)

    # initial model
    model_op = get_sag_network(args.architecture)
    model = model_op(
        in_dim=feat_dim,
        hid_dim=args.hidden_dim,
        out_dim=out_dim,
        num_convs=args.conv_layers,
        pool_ratio=args.pool_ratio,
        dropout=args.dropout
    )
    if args.load_epoch >= 0 and args.save_dir is not None:
        model.load_state_dict(torch.load(args.save_dir + '/' + args.dataset
                                         + "/model.iter-" + str(args.load_epoch)))
    if args.cuda >= 0 and torch.cuda.is_available():
        model = model.to(torch.cuda.current_device())
    log("model init finished")
    log("MODEL::::::::::SAGPool")

    logger = train(
        model,
        train_dataloader,
        args,
        log,
        val_dataset=val_dataloader
    )
    result = evaluate(model, test_dataloader, args, log, logger=logger)
    log("\nTEST::::::::::")
    log("Test : Accuracy {:.2f}% | F1 score {:.2f}% | Precision {:.2f}% | Recall {:.2f}%".format(
        result[0] * 100, result[1] * 100, result[2] * 100, result[3] * 100))

    return logger


def graph_classify_test(args):
    """
    Graph binary classification task for test suite
    """
    save_dir = args.save_dir + "/" + args.dataset
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    run_id = make_run_id(f'Train-SAGPool_{args.dataset}', 'GraphBinaryClassify')
    log_file = os.path.join(args.save_dir, f"{run_id}.log")
    log = Logger(log_file)
    log(str(args))

    dataset_test = GraphDataset(args.dataset, args.train_dir)
    dataset_test.load()

    # add self loop
    for i in range(len(dataset_test)):
        dataset_test.graph_list[i] = dgl.add_self_loop(dataset_test.graph_list[i])
    test_dataloader = prepare_data(dataset_test, args, train=False)

    node_dim, feat_dim, out_dim, max_num_node = dataset_test.statistics()
    log("++++++++++STATISTICS ABOUT THE DATASET++++++++")
    log(f"dataset node dimension is {node_dim}")
    log(f"dataset node feature dimension is {feat_dim}")
    log(f"dataset label dimension is {out_dim}")
    log(f"the max num node is {max_num_node}")
    log(f"number of test graphs is {len(dataset_test)}")
    log("++++++++++MODEL STATISTICS++++++++")
    log(f"model hidden dim is {args.hidden_dim}")
    log(f"model conv layer number is {args.conv_layers}")
    log(f"model pool ratio is {args.pool_ratio}")

    if args.cuda >= 0:
        torch.cuda.set_device(args.cuda)

    # initial model
    model_op = get_sag_network(args.architecture)
    model = model_op(
        in_dim=feat_dim,
        hid_dim=args.hidden_dim,
        out_dim=out_dim,
        num_convs=args.conv_layers,
        pool_ratio=args.pool_ratio,
        dropout=args.dropout
    )
    if args.load_epoch >= 0 and args.save_dir is not None:
        model.load_state_dict(torch.load(args.save_dir + '/' + args.dataset
                                         + "/model.iter-" + str(args.load_epoch)))
    if args.cuda >= 0 and torch.cuda.is_available():
        model = model.to(torch.cuda.current_device())
    log("model init finished")
    log("MODEL::::::::::SAGPool")

    result = evaluate(model, test_dataloader, args, log)
    log("\nTEST::::::::::")
    log("Test : Accuracy {:.2f}% | F1 score {:.2f}% | Precision {:.2f}% | Recall {:.2f}%".format(
        result[0] * 100, result[1] * 100, result[2] * 100, result[3] * 100))


def main(parser):
    args = arg_parser(parser)
    if args.no_train:
        graph_classify_test(args)
    else:
        graph_classify_task(args)
