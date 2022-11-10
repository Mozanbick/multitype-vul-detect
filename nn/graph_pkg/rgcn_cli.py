import os
import time
import torch
import dgl
import argparse
import torch.nn.functional as F
import torch.nn as nn

from configs import modelConfig as ModelConfig
from nn.graph_pkg import Logger
from dgl.dataloading import GraphDataLoader
from utils.objects.dataset import GraphDataset
from nn.graph_pkg.rgcn import RGCNModel
from torch.utils.data import random_split
from torchmetrics import Accuracy, F1Score, Precision, Recall

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
global_train_time_per_epoch = []


def arg_parse(args):
    parser = argparse.ArgumentParser(parents=[args], description='R-GCN arguments')
    parser.add_argument('--num-workers', dest='n_worker', type=int,
                        help='number of workers when data loading')
    parser.add_argument('--num-bases', dest='num_bases', type=int, help='number of bases')

    parser.set_defaults(
        n_worker=1,
        num_bases=-1,
    )

    return parser.parse_args()


def make_run_id(model_name: str, task_name: str, run_name=None) -> str:
    """Choose a run ID, based on the --run-name parameter and the current time."""
    if run_name is not None:
        return run_name
    else:
        return "%s_%s__%s" % (model_name, task_name, time.strftime("%Y-%m-%d_%H-%M-%S"))


def collate(samples):
    # The input `samples` is a list of DGLGraphs
    graphs = samples
    labels = [g.label for g in graphs]
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)


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


def graph_classify_task(prog_args):
    save_dir = prog_args.save_dir + "/" + prog_args.dataset
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dataset = GraphDataset(prog_args.dataset, prog_args.train_dir)
    dataset.load()

    run_id = make_run_id(f'Train-R-GCN_{prog_args.dataset}', 'GraphBinaryClassify')
    log_file = os.path.join(prog_args.save_dir, f"{run_id}.log")
    log = Logger(log_file)
    log(str(prog_args))

    if not prog_args.ext_test:
        trian_size = int(prog_args.train_ratio * len(dataset))
        test_size = int(prog_args.test_ratio * len(dataset))
        val_size = int(len(dataset) - trian_size - test_size)

        dataset_train, dataset_val, dataset_test = random_split(
            dataset, (trian_size, val_size, test_size)
        )
    else:
        trian_size = int(prog_args.train_ratio * len(dataset))
        val_size = int(len(dataset) - trian_size)

        dataset_train, dataset_val = random_split(
            dataset, (trian_size, val_size)
        )

        dataset_test = GraphDataset(prog_args.dataset, prog_args.test_dir)
        dataset_test.load()

    train_dataloader = prepare_data(dataset_train, prog_args, train=True)
    val_dataloader = prepare_data(dataset_val, prog_args, train=False)
    test_dataloader = prepare_data(dataset_test, prog_args, train=False)
    in_dim, feat_dim, out_dim, max_num_node = dataset.statistics()
    log("++++++++++STATISTICS ABOUT THE DATASET")
    log(f"dataset node dimension is {in_dim}")
    log(f"dataset node feature dimension is {feat_dim}")
    log(f"dataset label dimension is {out_dim}")
    log(f"the max num node is {max_num_node}")
    log(f"number of graphs is {len(dataset)}")

    hidden_dim = prog_args.hidden_dim
    log("++++++++++MODEL STATISTICS++++++++")
    log(f"model hidden dim is {hidden_dim}")
    activation = F.relu

    # initial model
    model = RGCNModel(
        feat_dim,
        hidden_dim,
        out_dim,
        len(ModelConfig.list_etypes),
        num_bases=prog_args.num_bases,
        num_hidden_layers=prog_args.n_hidden_layers,
        dropout=prog_args.dropout
    )
    if prog_args.load_epoch >= 0 and prog_args.save_dir is not None:
        model.load_state_dict(torch.load(prog_args.save_dir + '/' + prog_args.dataset
                                         + "/model.iter-" + str(prog_args.load_epoch)))
    log("model init finished")
    log("MODEL::::::::::R-GCN")
    if prog_args.cuda >= 0 and torch.cuda.is_available():
        model = model.cuda()

    logger = train(
        train_dataloader,
        model,
        prog_args,
        log,
        val_dataset=val_dataloader
    )
    result = evaluate(test_dataloader, model, prog_args, log, logger=logger)
    log("Test : Accuracy {:.2f}% | F1 score {:.2f}% | Precision {:.2f}% | Recall {:.2f}%".format(
        result[0] * 100, result[1] * 100, result[2] * 100, result[3] * 100))

    log(
        "Train time per epoch: {:.4f}".format(sum(global_train_time_per_epoch) / len(global_train_time_per_epoch)))
    log("Max memory usage: {:.4f}".format(torch.cuda.max_memory_allocated(0) / (1024 * 1024)))


def graph_classify_test(prog_args):
    save_dir = prog_args.save_dir + "/" + prog_args.dataset
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dataset_test = GraphDataset(prog_args.dataset, prog_args.train_dir)
    dataset_test.load()
    test_dataloader = prepare_data(dataset_test, prog_args, train=False)

    run_id = make_run_id('Test-R-GCN', 'GraphBinaryClassify')
    log_file = os.path.join(prog_args.save_dir, f"{run_id}.log")
    log = Logger(log_file)
    log(str(prog_args))

    in_dim, feat_dim, out_dim, max_num_node = dataset_test.statistics()
    log("++++++++++STATISTICS ABOUT THE DATASET")
    log(f"dataset node dimension is {in_dim}")
    log(f"dataset node feature dimension is {feat_dim}")
    log(f"dataset label dimension is {out_dim}")
    log(f"the max num node is {max_num_node}")
    log(f"number of graphs is {len(dataset_test)}")

    hidden_dim = prog_args.hidden_dim
    log("++++++++++MODEL STATISTICS++++++++")
    log(f"model hidden dim is {hidden_dim}")

    # initial model
    model = RGCNModel(
        feat_dim,
        hidden_dim,
        out_dim,
        len(ModelConfig.list_etypes),
        num_bases=prog_args.num_bases,
        num_hidden_layers=prog_args.n_hidden_layers,
        dropout=prog_args.dropout
    )
    if prog_args.load_epoch >= 0 and prog_args.save_dir is not None:
        model.load_state_dict(torch.load(prog_args.save_dir + '/' + prog_args.dataset
                                         + "/model.iter-" + str(prog_args.load_epoch)))
    log("model init finished")
    log("MODEL::::::::::R-GCN")

    if prog_args.cuda >= 0 and torch.cuda.is_available():
        model = model.cuda()

    result = evaluate(test_dataloader, model, prog_args, log)
    log("Test : Accuracy {:.2f}% | F1 score {:.2f}% | Precision {:.2f}% | Recall {:.2f}%".format(
        result[0] * 100, result[1] * 100, result[2] * 100, result[3] * 100))


def train(dataset, model, prog_args, log, same_feat=True, val_dataset=None):
    dataloader = dataset
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        model.parameters()), lr=prog_args.lr)
    early_stopping_logger = {"best_epoch": -1, "val_acc": -1}

    if prog_args.cuda >= 0:
        torch.cuda.set_device(prog_args.cuda)
    for epoch in range(prog_args.epochs):
        begin_time = time.time()
        model.train()
        accum_correct = 0
        total = 0
        log("\nEPOCH ###### {} ######".format(epoch))
        computation_time = 0
        for (batch_idx, (batch_graph, graph_labels)) in enumerate(dataloader):
            for (key, value) in batch_graph.ndata.items():
                batch_graph.ndata[key] = value.float()
            graph_labels = graph_labels.long()
            if prog_args.cuda >= 0 and torch.cuda.is_available():
                batch_graph = batch_graph.to(torch.cuda.current_device())
                graph_labels = graph_labels.cuda()

            model.zero_grad()
            compute_start = time.time()
            ypred = model(batch_graph)
            indices = torch.argmax(ypred, dim=1)
            correct = torch.sum(indices == graph_labels).item()
            accum_correct += correct
            total += graph_labels.size()[0]
            loss = F.cross_entropy(ypred, graph_labels)
            loss.backward()
            batch_compute_time = time.time() - compute_start
            computation_time += batch_compute_time
            # nn.utils.clip_grad_norm_(model.parameters(), prog_args.clip)
            optimizer.step()

        train_accu = accum_correct / total
        log("train accuracy for this epoch {} is {:.2f}%".format(epoch, train_accu * 100))
        elapsed_time = time.time() - begin_time
        log("loss {:.4f} with epoch time {:.4f} s & computation time {:.4f} s ".format(
            loss.item(), elapsed_time, computation_time))
        global_train_time_per_epoch.append(elapsed_time)
        if val_dataset is not None:
            result = evaluate(val_dataset, model, prog_args, log)
            log("Validation : Accuracy {:.2f}% | F1 score {:.2f}% | Precision {:.2f}% | Recall {:.2f}%".format(
                result[0] * 100, result[1] * 100, result[2] * 100, result[3] * 100))
            if early_stopping_logger['val_acc'] <= result[0] <= train_accu or epoch == 0:
                early_stopping_logger.update(best_epoch=epoch, val_acc=result[0])
                if prog_args.save_dir is not None:
                    torch.save(model.state_dict(), prog_args.save_dir + "/" + prog_args.dataset
                               + "/model.iter-" + str(early_stopping_logger['best_epoch']))
            log("best epoch is EPOCH {}, val_acc is {:.2f}%".format(early_stopping_logger['best_epoch'],
                                                                    early_stopping_logger['val_acc'] * 100))
        torch.cuda.empty_cache()
    return early_stopping_logger


def evaluate(dataloader, model, prog_args, log, logger=None):
    if logger and prog_args.save_dir:
        model.load_state_dict(torch.load(prog_args.save_dir + "/" + prog_args.dataset
                                         + "/model.iter-" + str(logger['best_epoch'])))
    model.eval()
    # correct_label = 0
    test_acc = Accuracy()
    test_f1 = F1Score(average='macro', num_classes=2)
    test_pre = Precision(average='macro', num_classes=2)
    test_rec = Recall(average='macro', num_classes=2)
    with torch.no_grad():
        for batch_idx, (batch_graph, graph_labels) in enumerate(dataloader):
            for (key, value) in batch_graph.ndata.items():
                batch_graph.ndata[key] = value.float()
            graph_labels = graph_labels.long()
            if prog_args.cuda >= 0 and torch.cuda.is_available():
                batch_graph = batch_graph.to(torch.cuda.current_device())
                graph_labels = graph_labels.cuda()
                test_acc = test_acc.to(torch.cuda.current_device())
                test_f1 = test_f1.to(torch.cuda.current_device())
                test_pre = test_pre.to(torch.cuda.current_device())
                test_rec = test_rec.to(torch.cuda.current_device())
            ypred = model(batch_graph)
            indices = torch.argmax(ypred, dim=1)
            # correct = torch.sum(indices == graph_labels)
            # correct_label += correct.item()
            loss = F.cross_entropy(ypred, graph_labels)
            acc = test_acc(indices, graph_labels)
            f1 = test_f1(indices, graph_labels)
            pre = test_pre(indices, graph_labels)
            rec = test_rec(indices, graph_labels)
        log("loss {:.4f} in validation".format(loss.item()))
    # result = correct_label / (len(dataloader) * prog_args.batch_size)
    acc = test_acc.compute()
    # f1 = test_f1.compute()
    precision = test_pre.compute()
    recall = test_rec.compute()
    f1 = 2 * precision * recall / (precision + recall)
    return [acc, f1, precision, recall]


def main(parser):
    prog_args = arg_parse(parser)
    # print(prog_args)
    if prog_args.no_train:
        graph_classify_test(prog_args)
    else:
        graph_classify_task(prog_args)
