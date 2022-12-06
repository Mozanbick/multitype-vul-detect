import hashlib
# import torch
import json
import os
# from configs import modelConfig
# from utils.objects.spg import extract_tokens
# from utils.objects import Cpg
# from slice2graph.slice_to_graph import program_slices_to_graphs_with_load


def file_hash(path: str):
    with open(path, "r") as fp:
        return hashlib.md5(fp.read().encode()).hexdigest()


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

    md1 = hashlib.md5()
    md2 = hashlib.md5()
    file1 = "d:\\Eclipse\\workspace\\SCProcess\\convertedFAN\\DoS\\CWE-125\\CVE-2012-5110\\1\\patch.diff"
    file2 = "d:\\Eclipse\\workspace\\SCProcess\\convertedFAN\\DoS\\CWE-125\\CVE-2012-5110\\3\\patch.diff"
    print(file_hash(file1) == file_hash(file2))
