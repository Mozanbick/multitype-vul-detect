import time
import argparse

from configs import modelConfig
from slice2graph.slice_to_sequence import graph_to_sequence
from utils.objects import Cpg
from os.path import join


def arg_parse():
    parser = argparse.ArgumentParser(description="Data pre-processing arguments")
    parser.add_argument('--dataset', dest='dataset', help='Dataset to process')
    parser.add_argument('--group', dest='group', help='Pre-processing group of selected dataset')
    parser.add_argument(
        '--with-load',
        dest='with_load',
        action='store_const', const=True,
        help='Load existing slice sequences from sequence dir'
    )
    parser.add_argument(
        '--seq-dir',
        dest='seq_dir',
        help='Directory to save the generated slice sequences'
    )
    parser.add_argument(
        '--gen-seq',
        dest='gen_seq',
        action='store_const', const=True,
        help='Generate program slice sequences'
    )
    parser.add_argument(
        '--gen-w2v',
        dest='gen_w2v',
        action='store_const', const=True,
        help='Generate embedding models from all slice/function program graphs. '
             'Please execute this procedure after all graphs are generated and saved.'
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
        '--s2dataset',
        dest='s2dataset',
        action='store_const', const=True,
        help='Convert slice sequences into model input dataset'
    )
    parser.add_argument(
        '--dataset-dir',
        dest='dataset_dir',
        help='Directory to save input dataset'
    )

    parser.set_defaults(
        dataset='cfan',
        group='group0',
        with_load=False,
        seq_dir="./joern/repository/",
        gen_seq=False,
        gen_w2v=False,
        g2dataset=False,
        label_path="",
        corpus_dir="./input/corpus/",
        w2v_dir="./input/w2v/",
        s2dataset=False,
        dataset_dir="./input/dataset/",
    )

    return parser.parse_args()


def init_model_config(args):
    modelConfig.set_dataset(args.dataset)
    modelConfig.set_group(args.group)
    modelConfig.set_spgs_dir(join(args.seq_dir, args.dataset, "spgs"))


def main():
    args = arg_parse()
    print(args)
    init_model_config(args)
    group = modelConfig.group
    if 'train' in group:
        save_dir = join(args.dataset_dir, modelConfig.dataset, "train")
    elif 'dtest' in group:
        save_dir = join(args.dataset_dir, modelConfig.dataset, "dtest")
    else:
        save_dir = join(args.dataset_dir, modelConfig.dataset)
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
        else:
            label_path = ""
    seq_path = join(args.seq_dir, modelConfig.dataset, 'sequence')
    corpus_path = join(args.corpus_dir, modelConfig.dataset)
    w2v_path = join(args.w2v_dir, f"w2v_model_{modelConfig.dataset}.model")
    start_time = time.time()
    cpg = Cpg(cpg_path)
    # preprocessing procedure
    # sequence_to_dataset(cpg, points_file, label_path, seq_path, corpus_path, w2v_path, save_dir, args)
    graph_to_sequence(cpg, seq_path, corpus_path, w2v_path, save_dir, args)
    end_time = time.time()
    print(f"Total process time: {end_time - start_time} s.")


if __name__ == '__main__':
    main()
