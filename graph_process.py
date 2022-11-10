import os
import time
import argparse

from configs import modelConfig
from slice2graph.slice_to_graph import *
from utils.objects import Cpg
from os.path import exists, join


def arg_parse():
    parser = argparse.ArgumentParser(description="Data pre-processing arguments")
    parser.add_argument('--dataset', dest='dataset', help='Dataset to process')
    parser.add_argument('--group', dest='group', help='Pre-processing group of selected dataset')
    parser.add_argument('--func-level', dest='func_level', action='store_const', const=True,
                        help='To pre-precessing source dataset in function level. Default is slice level.')
    parser.add_argument(
        '--nodes-dim',
        dest='nodes_dim',
        type=int,
        help='Nodes dim of each slice/function graph; preprocessing program would cut or pad the nodes matrix '
             'to meet the length.'
    )
    parser.add_argument(
        '--embed-dim',
        dest='embed_dim',
        type=int,
        help='Embedding length of each node'
    )
    parser.add_argument(
        '--vul-ratio',
        dest='vul_ratio',
        type=int,
        help='Ratio of non-vulnerable data to vulnerable data. '
             'Default is 1:1, when setting to 3, it means vul-data: non-vul-data is 1:3'
    )
    parser.add_argument(
        '--spgs-dir',
        dest='spgs_dir',
        help='Directory to save the generated slice program graphs. Default dir is `./joern/repository`'
    )
    parser.add_argument(
        '--fpgs-dir',
        dest='fpgs_dir',
        help='Directory to save the generated function program graphs. Default dir is `./joern/repository`'
    )
    parser.add_argument(
        '--ast-attr-path',
        dest='ast_attr_path',
        help='Path to ast attribute file'
    )
    parser.add_argument(
        '--gen-graph',
        dest='gen_graph',
        action='store_const', const=True,
        help='Generate slice/function program graphs from source dataset'
    )
    parser.add_argument(
        '--with-load',
        dest='with_load',
        action='store_const', const=True,
        help='Load existing slice/function graph from spgs/fpgs dir'
    )
    parser.add_argument(
        '--gen-corpus',
        dest='gen_corpus',
        action='store_const', const=True,
        help='Generate tokens for each dataset group'
    )
    parser.add_argument(
        '--gen-w2v',
        dest='gen_w2v',
        action='store_const', const=True,
        help='Generate embedding models from all slice/function program graphs. '
             'Please execute this procedure after all graphs are generated and saved.'
    )
    parser.add_argument(
        '--g2dataset',
        dest='g2dataset',
        action='store_const', const=True,
        help='Convert slice/function program graphs to model input dataset'
    )
    parser.add_argument(
        '--label-path',
        dest='label_path',
        help='Path to source dataset vulnerability label dict'
    )
    parser.add_argument(
        '--corpus-dir',
        dest='corpus_dir',
        help='Directory to save corpus. Default dir is `./input/corpus`'
    )
    parser.add_argument(
        '--w2v-dir',
        dest='w2v_dir',
        help='Directory to save embedding models. Default dir is `./input/w2v`'
    )
    parser.add_argument(
        '--dataset-dir',
        dest='dataset_dir',
        help='Directory to save input dataset'
    )

    parser.set_defaults(
        dataset='cfan',
        group='group0',
        func_level=False,
        nodes_dim=205,
        embed_dim=100,
        vul_ratio=1,
        spgs_dir="./joern/repository/",
        fpgs_dir="./joern/repository/",
        ast_attr_path="./joern/files/our_map_all.txt",
        gen_graph=False,
        with_load=False,
        gen_w2v=False,
        g2dataset=False,
        label_path="",
        corpus_dir="./input/corpus/",
        w2v_dir="./input/w2v/",
        dataset_dir="./input/dataset/"
    )

    return parser.parse_args()


def init_model_config(args):
    modelConfig.set_dataset(args.dataset)
    modelConfig.set_group(args.group)
    modelConfig.set_nodes_dim(args.nodes_dim)
    modelConfig.set_embed_dim(args.embed_dim)
    modelConfig.set_vul_ratio(args.vul_ratio)
    modelConfig.set_spgs_dir(join(args.spgs_dir, args.dataset, "spgs"))
    modelConfig.set_fpgs_dir(join(args.fpgs_dir, args.dataset, "fpgs"))


def get_suffix():
    suffix = '_'
    if 'AST' in modelConfig.list_etypes:
        suffix += 'a'
    if 'CFG' in modelConfig.list_etypes:
        suffix += 'c'
    if 'CDG' in modelConfig.list_etypes and 'DDG' in modelConfig.list_etypes:
        suffix += 'p'
    return suffix if len(suffix) != 1 else ''


def main():
    args = arg_parse()
    print(args)
    init_model_config(args)
    group = modelConfig.group
    if modelConfig.differ_edges:
        suffix = get_suffix()
    else:
        suffix = ''
    if 'train' in group:
        save_dir = join(args.dataset_dir, modelConfig.dataset, "train")
    elif 'test' in group:
        save_dir = join(args.dataset_dir, modelConfig.dataset, "test")
    else:
        save_dir = join(args.dataset_dir, modelConfig.dataset + suffix)
    points_file = f"./joern/joern-cli/results_{modelConfig.dataset}/{group}/AllVulPoints.txt"
    cpg_path = f"./joern/joern-cli/results_{modelConfig.dataset}/{group}"
    if args.label_path != "":
        label_path = args.label_path
    else:
        if modelConfig.dataset == "sard":
            label_path = "./joern/files/SARD_testcaseinfo.xml"
        elif modelConfig.dataset == "cnvd":
            label_path = "./joern/files/cnvd_diff/vul_line.pkl"
        elif modelConfig.dataset == "cfan":
            label_path = "./joern/files/cfan_diff/vul_line.pkl"
        elif modelConfig.dataset == "oldnvd":
            label_path = "./joern/files/oldnvd_diff/vul_line.pkl"
        elif modelConfig.dataset == "newnvd":
            label_path = "./joern/files/newnvd_diff/vul_line.pkl"
        else:
            label_path = ""
    corpus_path = join(args.corpus_dir, modelConfig.dataset)
    w2v_path = join(args.w2v_dir, f"w2v_model_{modelConfig.dataset}.model")
    start_time = time.time()
    cpg = Cpg(cpg_path)
    # preprocessing procedure
    graph_to_dataset_new(cpg, points_file, label_path, corpus_path, w2v_path, save_dir, args)
    end_time = time.time()
    print(f"Total process time: {end_time - start_time} s.")


def main_test():
    modelConfig.set_dataset("cnvd")
    modelConfig.set_group("test_group1")
    dataset_dir = f"./input/{modelConfig.dataset}"
    group = modelConfig.group
    if 'train' in group:
        save_dir = join(dataset_dir, "train")
    elif 'test' in group:
        save_dir = join(dataset_dir, "test")
    else:
        save_dir = dataset_dir
    points_file = f"./joern/joern-cli/results_{modelConfig.dataset}/{group}/AllVulPoints.txt"
    cpg_path = f"./joern/joern-cli/results_{modelConfig.dataset}/{group}"
    if modelConfig.dataset == "sard":
        label_path = "./joern/files/SARD_testcaseinfo.xml"
    elif modelConfig.dataset == "cnvd":
        label_path = "./joern/files/cnvd_diff/vul_line.pkl"
    elif modelConfig.dataset == "cfan":
        label_path = "./joern/files/cfan_diff/vul_line.pkl"
    elif modelConfig.dataset == "oldnvd":
        label_path = "./joern/files/oldnvd_diff/vul_line.pkl"
    elif modelConfig.dataset == "newnvd":
        label_path = "./joern/files/newnvd_diff/vul_line.pkl"
    else:
        label_path = ""
    corpus_dir = "./input/corpus"
    w2v_dir = "./input/w2v"
    corpus_path = join(corpus_dir, modelConfig.dataset)
    w2v_path = join(w2v_dir, f"w2v_model.model")
    start_time = time.time()
    cpg = Cpg(cpg_path)
    # preprocessing procedure
    # graph_to_dataset_new(cpg, points_file, label_path, corpus_path, w2v_path, save_dir, args)
    graph_to_dataset(cpg, points_file, label_path, corpus_path, w2v_path, save_dir)
    end_time = time.time()
    print(f"Total process time: {end_time - start_time} s.")


if __name__ == '__main__':
    main()
    # main_test()
