import os
import pickle
from slice2graph.gen_slice import *
from slice2graph.slice_to_graph import program_slices_to_graphs_with_load
from utils.objects.cpg import Cpg
from utils.objects import FPG
from configs import modelConfig as ModelConfig
from utils.embeddings import Corpus, generate_w2vModel, load_w2vModel
from utils.objects.dataset import GraphDataset


def graph_to_sequence(
        cpg: Cpg,
        seq_path: str,
        corpus_path: str,
        w2v_path: str,
        save_path: str,
        args
):
    # generate program slice sequences
    if args.gen_seq:
        # program_slices_to_sequences(cpg, points_file, label_path, seq_path)
        g_list = program_slices_to_graphs_with_load()
        seq_save_dir = join(seq_path, ModelConfig.group)
        if not exists(seq_save_dir):
            os.makedirs(seq_save_dir)
        for g in g_list:
            t, lines = g.to_sequence(cpg)
            seq_save_path = join(seq_save_dir, f"{t}_slices.txt")
            with open(seq_save_path, "a", encoding="utf-8") as fp:
                fp.write("\n--------------------------------------------------\n")
                fp.write("\n".join(lines))
                fp.write("\n==================================================\n\n")
            # with open(join(seq_save_dir, "spg_info.txt"), "a", encoding="utf-8") as fp:
            #     fp.write("\n--------------------------------------------------\n")
            #     print(g, file=fp)
            #     fp.write("\n==================================================\n\n")
