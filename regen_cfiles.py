import os
import re
import random

from argparse import ArgumentParser
from os.path import join
from slice2graph.slice_to_graph import program_slices_to_graphs_with_load
from configs import modelConfig
from utils.objects.cpg import Cpg
from tqdm import tqdm


class AST:

    def __init__(self, nodes):
        self.nodes = nodes
        self.ast: dict[int, list] = {}

    def _arrange_ast(self):
        for node in self.nodes:
            if node[0] not in self.ast:
                self.ast[node[0]] = [node[1]]
            else:
                self.ast[node[0]].append(node[1])


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, help='Dataset to re-generate')
    parser.add_argument('--group', dest='group', type=str, help='Group to re-generate')
    parser.add_argument(
        '--save-dir',
        dest='save_dir',
        type=str,
        help='Directory to save re-generated C files'
    )
    parser.add_argument(
        '--spgs-dir',
        dest='spgs_dir',
        type=str,
        help='Directory to load slice program graphs'
    )
    parser.add_argument(
        '--info-dir',
        dest='info_dir',
        type=str,
        help='Directory to load function info files'
    )

    parser.set_defaults(
        dataset='fan',
        group='group0',
        save_dir='joern/data/fan',
        spgs_dir="./joern/repository/",
        info_dir="./joern/data/info/"
    )

    return parser.parse_args()


def filterNodes(list, num):
    list_new = []
    if len(list) == 0:
        return list_new
    for i in range(len(list)):
        l1 = list[i][0]
        l2 = list[i][1]
        if l1 < num & l2 < num:
            list_new.append(list[i])

    return list_new


def get_info(filepath):
    # ast
    adj_code = []
    adj_child = []
    adj_next = []
    adj_com = []
    adj_guared = []
    adj_lexical = []
    adj_jump = []
    ast_nodes = []

    # cdfg
    adj_cdfg_a = []
    adj_cdfg_c = []
    adj_cdfg_p = []

    # initialize
    label_label = False  # label
    label_code = False  # code
    label_child = False  # child
    label_next = False  # next_token
    label_from = False  # compute_form
    label_by = False  # garde_by
    label_negation = False  # garde_negation
    label_att = False  # node_attribute
    label_node = False  # ast_node
    label_use = False  # Lexical_use
    label_jump = False  # jump
    label_joern = False  # cdfg
    label_joern_word = False  # joern_word
    joern_word = []

    with open(filepath, "r") as f:
        data = f.readlines()
        for line in data:
            if line.strip().find("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^") >= 0:  # end of lines
                break
            nodes = line.split('\n')[0].split(',')
            if line.find("-----label-----") >= 0:
                label_label = True
                continue
            if label_label:
                label_single = line.strip()
                label_label = False
                continue

            if line.find("-----code-----") >= 0:
                label_code = True
                continue
            if label_code:
                if line.find("-----children-----") >= 0:
                    label_code = False
                else:
                    adj_code.append(line)

            if line.find("-----children-----") >= 0:
                label_child = True
                continue
            if label_child:
                if line.find("-----nextToken-----") >= 0:
                    label_child = False
                else:
                    adj_child.append((int(nodes[0]), int(nodes[1])))

            if line.find("-----nextToken-----") >= 0:
                label_next = True
                continue
            if label_next:
                if line.find("-----computeFrom-----") >= 0:
                    label_next = False
                else:
                    for i in range(len(nodes)):
                        if i != len(nodes) - 1:
                            adj_next.append((int(nodes[i]), int(nodes[i + 1])))

            if line.find("-----computeFrom-----") >= 0:
                label_from = True
                continue
            if label_from:
                if line.find("-----guardedBy-----") >= 0:
                    label_from = False
                else:
                    adj_com.append((int(nodes[0]), int(nodes[1])))

            if line.find("-----guardedBy-----") >= 0:
                label_by = True
                continue
            if label_by:
                if line.find("-----guardedByNegation-----") >= 0:
                    label_by = False
                else:
                    adj_guared.append((int(nodes[0]), int(nodes[1])))

            if line.find("-----guardedByNegation-----") >= 0:
                label_negation = True
                continue
            if label_negation:
                if line.find("-----lastLexicalUse-----") >= 0:
                    label_negation = False
                else:
                    adj_guared.append((int(nodes[0]), int(nodes[1])))

            if line.find("-----lastLexicalUse-----") >= 0:
                label_use = True
                continue
            if label_use:
                if line.find("-----jump-----") >= 0:
                    label_use = False
                else:
                    adj_lexical.append((int(nodes[0]), int(nodes[1])))

            if line.find("-----jump-----") >= 0:
                label_jump = True
                continue
            if label_jump:
                if line.find("-----attribute-----") >= 0:
                    label_jump = False
                else:
                    adj_jump.append((int(nodes[0]), int(nodes[1])))

            if line.find("-----attribute-----") >= 0:
                label_att = True
                continue
            if label_att:
                if line.find("-----ast_node-----") >= 0:
                    label_att = False
                else:
                    num = line.strip().split(';')
                    for x in num:
                        if x == "" or x == '\n':
                            num.remove(x)

            if line.find("-----ast_node-----") >= 0:
                label_node = True
                continue
            if label_node:
                if line.find("-----joern-----") >= 0:
                    label_node = False
                else:
                    ast_nodes.append(line)

            if line.find("-----joern-----") >= 0:
                label_joern = True
                continue
            if label_joern:
                if line.find("-----------------------------------") >= 0:
                    label_joern = False
                else:
                    threedot = line.strip().split("(")[1].split(")")[0].split(",")  # 分割成三元组

                    try:
                        adj_single = (int(threedot[0]), int(threedot[1]))
                        if int(threedot[2]) == 0:
                            adj_cdfg_a.append(adj_single)
                            continue
                        if int(threedot[2]) == 1:
                            adj_cdfg_c.append(adj_single)
                            continue
                        if int(threedot[2]) == 2:
                            adj_cdfg_p.append(adj_single)
                            continue
                    except Exception as e:
                        print("joern nodes occur errors")
                        print(e)
                        continue
            if line.find("-----------------------------------") >= 0:
                label_joern_word = True
                continue
            if label_joern_word:
                if re.search('(?<=,).*', line):
                    joern_word.append(re.search('(?<=,).*', line).group().split(')')[0])

    # nodes of ast need to be smaller than len(attribute)
    adj_len = len(num)
    adj_child = filterNodes(adj_child, adj_len)
    adj_next = filterNodes(adj_next, adj_len)
    adj_com = filterNodes(adj_com, adj_len)
    adj_guared = filterNodes(adj_guared, adj_len)
    adj_lexical = filterNodes(adj_lexical, adj_len)
    adj_jump = filterNodes(adj_jump, adj_len)

    return {
        'code': adj_code,
        'child': adj_child,
        'next': adj_next,
        'com': adj_com,
        'guared': adj_guared,
        'lexical': adj_lexical,
        'jump': adj_jump,
        'attr': num,
        'nodes': ast_nodes,
    }


def main():
    args = arg_parser()
    modelConfig.set_dataset(args.dataset)
    modelConfig.set_group(args.group)
    modelConfig.set_spgs_dir(join(args.spgs_dir, args.dataset, "spgs"))
    group = modelConfig.group
    cpg_path = f"./joern/joern-cli/results_{modelConfig.dataset}/{group}"
    cpg = Cpg(cpg_path)
    g_list = program_slices_to_graphs_with_load()
    g_list = random.sample(g_list, 10000)
    files = os.listdir(args.info_dir)
    for g in tqdm(g_list, desc='start regen c files...'):
        name = g.filename.split(os.sep)[-1]
        name = name + '.c.txt'
        if name not in files:
            # print(name)
            continue
        filepath = os.path.join(args.info_dir, name)
        # g.generate_new_c_files(cpg, name, get_info(filepath), args.save_dir)
        g.generate_new_c_files_ext(cpg, name, filepath, args.save_dir)


if __name__ == '__main__':
    main()
