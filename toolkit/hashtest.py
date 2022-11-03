# import hashlib
# import torch
import json
import os
from configs import modelConfig
from utils.objects.spg import extract_tokens
from utils.objects import Cpg
from slice2graph.slice_to_graph import program_slices_to_graphs_with_load


if __name__ == '__main__':
    # l1 = [[1, 2, 3], [5, 6, 7, 8, 9], [12, 13]]
    # l2 = [[1, 2, 3], [5, 6, 7, 8, 9], [12, 13]]
    # l3 = [[1, 2, 3], [5, 6, 7, 8, 10], [12, 13]]
    #
    # print(hashlib.md5(str(l1).encode()).hexdigest())
    # print(hashlib.md5(str(l2).encode()).hexdigest())
    # print(hashlib.md5(str(l3).encode()).hexdigest())

    # with open(os.path.join("../", modelConfig.ast_attr_path), "r") as f:
    #     s = f.read()
    #     s = s.replace('\'', '\"')
    #     mapping = json.loads(json.dumps(eval(s)))
    # print(mapping)

    modelConfig.set_dataset("tests")
    modelConfig.set_spgs_dir("../joern/repository/tests/spgs")
    modelConfig.set_group("group0")
    cpg = Cpg("../joern/joern-cli/results_tests/group0")
    spg_list = program_slices_to_graphs_with_load()
    print("start...")
    with open("corpus.txt", "w", encoding="utf-8") as fp:
        for spg in spg_list:
            fp.write("\n------------------------------\n")
            method = cpg.get_method_by_filename(spg.testID, spg.filenames.pop())
            for node in spg.node_list:
                if not node.code:
                    continue
                fp.write(str(extract_tokens(node.code, method)))
                fp.write("\n")
            fp.write("\n===============================\n")
