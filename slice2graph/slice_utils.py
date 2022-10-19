import hashlib
import xml.sax
import pickle
from os.path import join, exists
from utils.objects import Method, Cpg, LabelHandler, SPG
from typing import Dict, List, Any, Set
from tqdm import tqdm
from configs import modelConfig as ModelConfig


def get_points_from_file(points_file: str):
    """
    Load 6 kinds of points from the point file generated by joern scripts
    """
    if not exists(points_file):
        raise FileNotFoundError(f"Can't find vul points file {points_file}, please verify your path.")
    points = []
    with open(points_file, "r") as fp:
        lines = fp.readlines()
        for line in lines:
            line = line.strip().replace("List(", "").replace(")", "")
            items = line.split(", ")
            points.append(list(map(int, items)))
    return points


def write_slices_to_file(slices_list: Dict[int, List], save_path: str):
    """
    Write slice lists to files in point types unit
    """
    if slices_list:
        with open(save_path, "w") as fp:
            for point_id, slices in slices_list.items():
                for s in slices:
                    fp.write(str(point_id) + " : " + str(s) + "\n")


def _gen_in_fwd_slice(method: Method, point_id):
    """
    Generating in-function forward slice
    """
    if point_id not in method.node_id_set:
        return []
    if method.cdg_edges is None and method.ddg_edges is None:
        return []
    slice_fwd = set()
    node_not_traverse = set(method.get_ast_entrance(point_id))
    while node_not_traverse:
        pid = node_not_traverse.pop()
        # prevent dead recursion
        if pid in slice_fwd:
            continue
        slice_fwd.add(pid)
        # node_not_traverse.update(method.cdg_edges[pid] if pid in method.cdg_edges else [])
        node_not_traverse.update(method.ddg_edges[pid] if pid in method.ddg_edges else [])
    # traverse and add variable definition node and add control structure node
    lp_set = set()
    for nid in slice_fwd:
        # function entry not considered
        if nid == method.entry:
            continue
        # variable definition node
        ast_subtree = method.get_ast_subtree_nodes(nid)
        for ast_node in ast_subtree:
            if method.nodes[ast_node].label == "IDENTIFIER":
                if method.nodes[ast_node].name in method.local_param:
                    lp_set.add(method.local_param[method.nodes[ast_node].name][0])
        # control structure node
        fnid = method.get_father_node(nid)
        if method.nodes[fnid].label == "CONTROL_STRUCTURE":
            lp_set.add(fnid)
    return list(slice_fwd.union(lp_set))


def _gen_cross_fwd_slice(cpg: Cpg, testID, slice_fwd: List, point_id):
    """
    Generating cross-function forward slice
    """
    cross_func_fwd_slice = {}
    call_pair = []
    ret_pair = []
    cur_method = cpg.get_method_by_id(testID, point_id)
    for nid in slice_fwd:
        if nid == point_id or nid == cur_method.entry or nid == cur_method.ret:
            continue
        if cur_method.nodes[nid].label == "CONTROL_STRUCTURE":
            continue
        # If a node in slice nodes set calls another function, we take that function into consideration
        sub_ast_nodes = cur_method.get_ast_subtree_nodes(nid)  # ast sub-tree nodes
        cfwd_slice = set()
        for snid in sub_ast_nodes:
            if cpg.nodes[snid].label == "CALL" and not cpg.nodes[snid].name.startswith("<operator>"):
                if 'NEW' in cur_method.filename:
                    method = cpg.get_method(testID, cpg.nodes[snid].name, 'NEW')
                elif 'OLD' in cur_method.filename:
                    method = cpg.get_method(testID, cpg.nodes[snid].name, 'OLD')
                else:
                    method = cpg.get_method(testID, cpg.nodes[snid].name)
                if method:
                    params = method.get_params()
                    if not params:
                        params = [method.entry]
                    _fwd_slice = set()
                    _fwd_slice.add(method.entry)
                    call_pair.append((nid, method.entry))
                    for param in params:
                        _fwd_slice.update(_gen_in_fwd_slice(method, param))
                        call_pair.append((nid, param))
                    for fid in _fwd_slice:
                        if method.nodes[fid].label == "RETURN":
                            ret_pair.append((fid, nid))
                    # multi function call
                    # _fwd_slice = _gen_cross_fwd_slice(cpg, testID, list(_fwd_slice), point_id)
                    # update cross fwd slice
                    cfwd_slice.update(_fwd_slice)
                    break
        if cfwd_slice:
            cross_func_fwd_slice[nid] = sorted(list(cfwd_slice))
    # v2, not do
    # for nid in cross_func_fwd_slice:
    #     idx = slice_fwd.index(nid)
    #     slice_fwd = slice_fwd[:idx+1] + cross_func_fwd_slice[nid] + slice_fwd[idx+1:]
    return cross_func_fwd_slice, call_pair, ret_pair


def _gen_in_bwd_slice(method: Method, point_id):
    """
    Generating in-function backward slice
    """
    if point_id not in method.node_id_set:
        return []
    if method.cdg_edges is None and method.ddg_edges is None:
        return []
    slice_bwd = set()
    node_not_traverse = set(method.get_ast_entrance(point_id))
    while node_not_traverse:
        pid = node_not_traverse.pop()
        # prevent dead recursion
        if pid in slice_bwd:
            continue
        slice_bwd.add(pid)
        # both control dependencies and data dependencies
        for node_in in method.cdg_edges:
            if pid in method.cdg_edges[node_in]:
                node_not_traverse.add(node_in)
        for node_in in method.ddg_edges:
            if pid in method.ddg_edges[node_in]:
                node_not_traverse.add(node_in)
    # traverse and add variable definition node and add control structure node
    lp_set = set()
    for nid in slice_bwd:
        # function entry not considered
        if nid == method.entry:
            continue
        # variable definition node
        ast_subtree = method.get_ast_subtree_nodes(nid)
        for ast_node in ast_subtree:
            if method.nodes[ast_node].label == "IDENTIFIER" and method.nodes[ast_node].name in method.local_param:
                lp_set.add(method.local_param[method.nodes[ast_node].name][0])
        # control structure node
        fnid = method.get_father_node(nid)
        if method.nodes[fnid].label == "CONTROL_STRUCTURE":
            lp_set.add(fnid)
    return list(slice_bwd.union(lp_set))


def _gen_cross_bwd_slice(cpg: Cpg, testID, slice_bwd: List, method: Method, point_id):
    """
    Generating cross-function backward slice, based on in-function backward slice
    Traversing the function to find nodes which "CALL" other functions which has "RET" node respectively,
        generating backward slice from "RET" nodes of each function and insert them into the
        corresponding node position in current slice
    Traversing the cpg to find nodes which is "CALL" type and calls the target method,
        generate backward slice from these nodes and add to the head of current slice,
        and each node in these nodes generates different slice respectively
    """
    cross_func_bwd_outer = {}
    methods = cpg.methods[testID]
    cross_func_bwd_inner = {}
    call_pair = []
    ret_pair = []
    # first condition
    for nid in slice_bwd:
        if nid == point_id or nid == method.entry or nid == method.ret:
            continue
        if method.nodes[nid].label == "CONTROL_STRUCTURE":
            continue
        # If a node in slice nodes set calls another function, we take that function into consideration
        ast_sub_nodes = method.get_ast_subtree_nodes(nid)
        cbwd_slice = set()
        for snid in ast_sub_nodes:
            if cpg.nodes[snid].label == "CALL" and not cpg.nodes[snid].name.startswith("<operator>"):
                if 'NEW' in method.filename:
                    m = cpg.get_method(testID, cpg.nodes[snid].name, 'NEW')
                elif 'OLD' in method.filename:
                    m = cpg.get_method(testID, cpg.nodes[snid].name, 'OLD')
                else:
                    m = cpg.get_method(testID, cpg.nodes[snid].name)
                if m:
                    rets = m.get_returns()
                    if not rets:
                        continue
                    _bwd_slice = set()
                    _bwd_slice.add(m.entry)
                    call_pair.append((nid, m.entry))
                    for ret in rets:
                        _bwd_slice.update(_gen_in_bwd_slice(m, ret))
                        ret_pair.append((ret, nid))
                    for bid in _bwd_slice:
                        if method.nodes[bid].label == "METHOD_PARAMETER_IN":
                            call_pair.append((nid, bid))
                    # multi function call
                    # _fwd_slice = _gen_cross_fwd_slice(cpg, testID, list(_fwd_slice), point_id)
                    # update cross bwd slice
                    cbwd_slice.update(_bwd_slice)
                    break
        if cbwd_slice:
            cross_func_bwd_inner[nid] = sorted(list(cbwd_slice))
    # v2
    # for nid in cross_func_bwd_s1:
    #     idx = slice_bwd.index(nid)
    #     slice_bwd = slice_bwd[:idx] + cross_func_bwd_s1[nid] + slice_bwd[idx:]
    # second condition
    for m in methods:
        if m.name == method.name:
            continue
        if 'NEW' in method.filename and 'NEW' not in m.filename:
            continue
        if 'OLD' in method.filename and 'OLD' not in m.filename:
            continue
        for nid, node in m.nodes.items():
            if node.label == "CALL" and node.name == method.name:
                _bwd_slice = _gen_in_bwd_slice(m, nid)
                _fwd_slice = _gen_in_fwd_slice(m, nid)
                _in_slice = sorted(list(set(_bwd_slice + _fwd_slice)))
                # idx = _bwd_slice.index(nid)
                cross_func_bwd_outer[nid] = _in_slice
                call_pair.append((nid, method.entry))
                params = method.get_params()
                for param in params:
                    if param in _in_slice:
                        call_pair.append((nid, param))
                rets = method.get_returns()
                for ret in rets:
                    if ret in _in_slice:
                        ret_pair.append((ret, nid))
    return cross_func_bwd_outer, cross_func_bwd_inner, call_pair, ret_pair


def gen_slices(cpg: Cpg, testID, method: Method, point_id):
    """
    Generating forward and backward slice of a candidate vulnerability point while traversing the pdg graph
    ### First generate a slice node set in a method, and then extract cross-function slice nodes
    ### The structure of a slice is:
            {
                "outer_bwd": {nid: [slice]},
                "inner_bwd": {nid1: [slice1], nid2: [slice2], ...},
                "in_slice": {point_id: [id1, id2, ...]},
                "cross_fwd": {nid1: [slice1], nid2: [slice2], ...},
                "call": [(in1, out1), (in2, out2), ...],
                "ret": [(in1, out1), (in2, out2), ...]
            }
            the keys in "inner_bwd" and "cross_fwd" are corresponding to the ids in "in_slice"
    """
    try:
        slices = []
        in_fwd_slice = _gen_in_fwd_slice(method, point_id)
        in_bwd_slice = _gen_in_bwd_slice(method, point_id)
        in_slice = {point_id: sorted(list(set(in_bwd_slice + in_fwd_slice)))}
        cross_fwd, cf, rf = _gen_cross_fwd_slice(cpg, testID, in_fwd_slice, point_id)
        outer_cross_bwd, inner_cross_bwd, cb, rb = _gen_cross_bwd_slice(cpg, testID, in_bwd_slice, method, point_id)
        call_pair = cf + cb
        ret_pair = rf + rb
        for nid, bs in outer_cross_bwd.items():
            slices.append(
                {
                    "outer_bwd": {nid: bs},
                    "inner_bwd": inner_cross_bwd,
                    "in_slice": in_slice,
                    "cross_fwd": cross_fwd,
                    "call": call_pair,
                    "ret": ret_pair,
                }
            )
    except KeyError:
        return None
    if slices:
        return slices
    else:
        return [{
            "outer_bwd": {},
            "inner_bwd": inner_cross_bwd,
            "in_slice": in_slice,
            "cross_fwd": cross_fwd,
            "call": call_pair,
            "ret": ret_pair,
        }]


def get_sequences_from_file(file_path: str, line_list: List):
    """
    Returns a list of code sequences of a specific source code file which are generated according to the line_list
    """
    sequences: List[str] = []
    with open(file_path, "r") as fp:
        lines = fp.readlines()
        cur_line = 1
        for line in lines:
            if cur_line in line_list:
                sequences.append(line.strip())
            cur_line += 1
    return sequences


def query_slice_info(cpg: Cpg, slice: List):
    """
    Query for a slice's information, including file_paths and line_lists
    """
    testID, method = cpg.get_node_info(slice[0])
    methods = cpg.methods[testID]
    cur_method = method
    file_paths = []
    line_lists = []
    cur_line_list = set()
    for point_id in slice:
        if point_id not in cur_method.node_id_set:
            lp_lines = cur_method.get_local_def_lines()
            cur_line_list.update(lp_lines)
            file_paths.append(cur_method.filename)
            line_lists.append(sorted(list(cur_line_list)))
            cur_line_list = set()
            for m in methods:
                if point_id in m.node_id_set:
                    cur_method = m
        cur_line_list.add(cpg.nodes[point_id].line_number)
        # if cpg.nodes[point_id].line_number in cur_method.use_line_to_def_line:
        #     cur_line_list.update(cur_method.use_line_to_def_line[cpg.nodes[point_id].line_number])
    lp_lines = cur_method.get_local_def_lines()
    cur_line_list.update(lp_lines)
    file_paths.append(cur_method.filename)
    line_lists.append(sorted(list(cur_line_list)))
    return file_paths, line_lists


def query_slice_info_ext(cpg: Cpg, slice: List):
    """
    Query for a slice's information, including file_paths, line_lists and methods
    """
    testID, method = cpg.get_node_info(slice[0])
    methods = cpg.methods[testID]
    cur_method = method
    in_methods = []
    file_paths = []
    line_lists = []
    cur_line_list = set()
    for point_id in slice:
        if point_id not in cur_method.node_id_set:
            lp_lines = cur_method.get_local_def_lines()
            cur_line_list.update(lp_lines)
            file_paths.append(cur_method.filename)
            line_lists.append(sorted(list(cur_line_list)))
            in_methods.append(cur_method)
            cur_line_list = set()
            for m in methods:
                if point_id in m.node_id_set:
                    cur_method = m
        if cpg.nodes[point_id].line_number:
            cur_line_list.add(cpg.nodes[point_id].line_number)
        # if cpg.nodes[point_id].line_number in cur_method.use_line_to_def_line:
        #     cur_line_list.update(cur_method.use_line_to_def_line[cpg.nodes[point_id].line_number])
    lp_lines = cur_method.get_local_def_lines()
    cur_line_list.update(lp_lines)
    file_paths.append(cur_method.filename)
    line_lists.append(sorted(list(cur_line_list)))
    in_methods.append(cur_method)
    return file_paths, line_lists, in_methods


def query_slice_info_v2(cpg: Cpg, testID, slice: Dict[str, Dict]):
    """
    Version 2, 2022/10/7
    """
    line_dict = {}
    cover_methods = []
    outer_bwd = slice["outer_bwd"]
    assert len(outer_bwd) <= 1  # may be 0 or 1
    line_dict["outer_bwd"] = {}
    for nid, s in outer_bwd.items():
        method = cpg.get_method_by_id(testID, nid)
        if method not in cover_methods:
            cover_methods.append(method)
        cur_lines = set()
        for pid in s:
            cur_lines.add(method.nodes[pid].line_number)
        line_dict["outer_bwd"][method.filename] = sorted(list(cur_lines))
    inner_bwd = slice["inner_bwd"]
    line_dict["inner_bwd"] = {}
    for nid, s in inner_bwd.items():
        method = cpg.get_method_by_id(testID, s[0])
        if method not in cover_methods:
            cover_methods.append(method)
        cur_lines = set()
        for pid in s:
            cur_lines.add(method.nodes[pid].line_number)
        line_dict["inner_bwd"][method.filename] = sorted(list(cur_lines))
    in_slice = slice["in_slice"]
    line_dict["in_slice"] = {}
    assert len(in_slice) == 1  # must be 1
    for nid, s in in_slice.items():
        method = cpg.get_method_by_id(testID, nid)
        if method not in cover_methods:
            cover_methods.append(method)
        cur_lines = set()
        for pid in s:
            cur_lines.add(method.nodes[pid].line_number)
        line_dict["in_slice"][method.filename] = sorted(list(cur_lines))
    cross_fwd = slice["cross_fwd"]
    line_dict["cross_fwd"] = {}
    for nid, s in cross_fwd.items():
        method = cpg.get_method_by_id(testID, s[0])
        if method not in cover_methods:
            cover_methods.append(method)
        cur_lines = set()
        for pid in s:
            cur_lines.add(method.nodes[pid].line_number)
        line_dict["cross_fwd"][method.filename] = sorted(list(cur_lines))
    return line_dict, cover_methods


def make_label_per_slice_xml(file_paths: List[str], line_lists: List[List[int]],
                             label_dict: Dict[str, Dict[str, List]]):
    """
    Make label per slice, items is a list: List[str, str, set, list]
    """
    for idx, filepath in enumerate(file_paths):
        filename = filepath.split("/")[-1]
        items = filename.split("@@")
        testID = items[0]
        name = items[1]
        start_line = int(items[2])
        lines = set(map(lambda x: x + start_line - 1, line_lists[idx]))
        if testID in label_dict and name in label_dict[testID]:
            vul_lines = label_dict[testID][name]
            if vul_lines:
                if len(vul_lines) == 1 and vul_lines[0] == 0:
                    return 1
                if set(vul_lines) & lines:
                    return 1
    return 0


def make_label_per_slice_diff(file_paths: List[str], line_lists: List[List[int]],
                              label_dict: Dict[str, Dict[str, List]]):
    for idx, filepath in enumerate(file_paths):
        filename = filepath.split("/")[-1]
        items = filename.split("@@")
        testID = items[0]
        vul = int(items[1])
        if vul == 0:
            return 0
        name = items[2]
        if ModelConfig.dataset == "cfan":
            name = testID + '_' + name
        elif ModelConfig.dataset == "oldnvd":
            name = filename
        lines = set(line_lists[idx])
        if testID in label_dict and name in label_dict[testID]:
            vul_lines = label_dict[testID][name]
            if vul_lines:
                if len(vul_lines) == 1 and vul_lines[0] == 0:
                    return 1
                if set(vul_lines) & lines:
                    return 1
    return 0


def make_label_per_slice_v2(line_dict: Dict[str, Dict[str, List]], label_dict: Dict[str, Dict[str, List]]):
    outer_bwd = line_dict["outer_bwd"]
    inner_bwd = line_dict["inner_bwd"]
    in_slice = line_dict["in_slice"]
    cross_fwd = line_dict["cross_fwd"]

    def check_vuln_line(lines: Dict[str, List]) -> bool:
        if not lines:
            return False
        for filepath, ls in lines.items():
            filename = filepath.split("/")[-1]
            items = filename.split("@@")
            testID = items[0]
            vul = int(items[1])
            if vul == 0:
                return False
            name = items[2]
            if ModelConfig.dataset == "cfan":
                name = testID + '_' + name
            elif ModelConfig.dataset == "oldnvd":
                name = filename
            ls = set(ls)
            if testID in label_dict and name in label_dict[testID]:
                vul_lines = label_dict[testID][name]
                if vul_lines:
                    if len(vul_lines) == 1 and vul_lines[0] == 0:
                        return True
                    if set(vul_lines) & ls:
                        return True
        return False

    if check_vuln_line(outer_bwd) or check_vuln_line(inner_bwd) or check_vuln_line(in_slice) or check_vuln_line(cross_fwd):
        return 1
    else:
        return 0


def gen_sequences_from_slice(cpg: Cpg, slices: Dict[int, List], save_path: str):
    """
    Generating code sequences in order from the given slices, and then write to file
    """
    with open(save_path, "w") as fp:
        for point_id, slices in slices.items():
            testID, method = cpg.get_node_info(point_id)
            for slice in slices:
                file_paths, line_lists = query_slice_info(cpg, slice)
                if len(file_paths) != len(line_lists):
                    # this may be something wrong in this slice
                    continue
                sequences = []
                for idx, file_path in enumerate(file_paths):
                    sequences += get_sequences_from_file(file_path, line_lists[idx])
                # write to file
                fp.write("\n--------------------------------------------------\n")
                slice_info = testID + "   " + method.filename + "   "
                fp.write(slice_info)
                if cpg.nodes[point_id].code:
                    fp.write(cpg.nodes[point_id].code + "   ")
                if cpg.nodes[point_id].line_number:
                    fp.write(f"line:{cpg.nodes[point_id].line_number}" + "\n")
                else:
                    fp.write(f"node-id:{point_id}" + "\n")
                fp.write("@@@\n")
                for sequence in sequences:
                    fp.write(sequence + "\n")
                fp.write("\n")


def gen_sequences(cpg: Cpg, slices: Dict[int, List], label_dict: Dict[str, Dict[str, List]], save_path: str):
    """
    Generate sequences from slices, these operations are involved:
    + slice data cleaning and deduplication
    + labeling (vulnerable 1 or non-vulnerable 0)
    + querying sequences from source file
    + write to slice file (.txt format)
    ### slice_set is a dict of the Dict structure:
        {
            hash(slice): [testID, filename, set(point_id), list(slice)]
        }
    """
    slice_set = {}
    # slice data cleaning and deduplication
    print("Applying slice data cleaning and deduplication...")
    for point_id, slice_list in slices.items():
        if not slice_list:
            continue
        testID, method = cpg.get_node_info(point_id)
        if not method:
            continue
        if testID not in slice_set:
            slice_set[testID] = {}
        try:
            for slice in slice_list:
                file_paths, line_lists = query_slice_info(cpg, slice)
                if len(file_paths) != len(line_lists):
                    # this may be something wrong in this slice
                    continue
                slice_hash = hashlib.md5(str(line_lists).encode()).hexdigest()
                if slice_hash not in slice_set[testID]:
                    slice_set[testID][slice_hash] = [testID, method.filename, {point_id}, file_paths, line_lists]
                else:
                    slice_set[testID][slice_hash][2].add(point_id)
        except:
            continue
    # get label info
    vul_count = 0
    non_vul_count = 0
    with open(save_path, "w") as fp:
        for testID, dicts in tqdm(slice_set.items(), desc="Labeling and Sequencing"):
            for slice_hash, items in tqdm(dicts.items(), leave=False):
                testID, filename, point_set, file_paths, line_lists = items
                # label = make_label_per_slice_xml(file_paths, line_lists, label_dict)
                label = make_label_per_slice_diff(file_paths, line_lists, label_dict)
                if label == 1:
                    vul_count += 1
                else:
                    non_vul_count += 1
                sequences = []
                for idx, file_path in enumerate(file_paths):
                    sequences += get_sequences_from_file(file_path, line_lists[idx])
                # write to file
                fp.write("\n--------------------------------------------------\n")
                slice_info = testID + "   " + filename + "\n"
                fp.write(slice_info)
                fp.write("focus on lines:   " + "   ".join(
                    str(cpg.nodes[point_id].line_number) for point_id in point_set) + "\n")
                fp.write(f"Label: {label}\n")
                fp.write("@@@\n")
                for sequence in sequences:
                    fp.write(sequence + "\n")
                fp.write("==================================================\n\n")
    return vul_count, non_vul_count


def get_vul_lines_from_xml(xml_path: str) -> Dict[str, Dict[str, List]]:
    label_dict = {}
    # make xml parser
    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)  # turn off namespaces
    handler = LabelHandler(label_dict)
    parser.setContentHandler(handler)  # rewrite handler
    # parser xml file
    parser.parse(xml_path)
    return label_dict


def get_vul_lines_from_pkl(filepath: str) -> Dict[str, Dict[str, List]]:
    with open(filepath, "rb") as fp:
        label_dict = pickle.load(fp)
    return label_dict


def _gen_hash(slice: Dict[str, Any]):
    hash_set = []
    outer_bwd = slice["outer_bwd"]
    inner_bwd = slice["inner_bwd"]
    in_slice = slice["in_slice"]
    cross_fwd = slice["cross_fwd"]
    call = slice["call"]
    ret = slice["ret"]

    def add_dict(sd: Dict[str, List]):
        if not sd:
            return
        for nid, s in sd.items():
            hash_set.append(s)

    def add_list(sl: List[tuple]):
        if not sl:
            return
        sl.sort()
        hash_set.append(sl)

    add_dict(outer_bwd)
    add_dict(inner_bwd)
    add_dict(in_slice)
    add_dict(cross_fwd)
    add_list(call)
    add_list(ret)

    return hashlib.md5(str(hash_set).encode()).hexdigest()


def gen_graphs(cpg: Cpg, slices: Dict[int, List], label_dict: Dict[str, Dict[str, List]], types: str):
    """
    Generate slice program graphs from slices, these operations are involved:
    + slice data cleaning and deduplication
    + labeling (vulnerable 1 or non-vulnerable 0)
    + extract nodes and edges from method's cpg
    ### slice_set is a dict of the Dict structure:
        {
            hash(slice): [testID, set(line_number), list(slice)]
        }
    """
    slice_set = {}
    # slice data cleaning and deduplication
    print("Applying slice data cleaning and deduplication...")
    for point_id, slice_list in slices.items():
        if not slice_list:
            continue
        testID, method = cpg.get_node_info(point_id)
        if not method:
            continue
        if testID not in slice_set:
            slice_set[testID] = {}
        for slice in slice_list:
            slice_hash = _gen_hash(slice)
            line = method.nodes[point_id].line_number
            if slice_hash not in slice_set[testID]:
                slice_set[testID][slice_hash] = [testID, {method.filename}, {line}, slice]
            else:
                slice_set[testID][slice_hash][1].add(method.filename)
                slice_set[testID][slice_hash][2].add(line)
    # get label info
    vul_count = 0
    non_vul_count = 0
    spg_list = []
    for testID, dicts in tqdm(slice_set.items(), desc="Labeling"):
        for slice_hash, items in tqdm(dicts.items(), leave=False):
            try:
                testID, name_set, line_set, slice = items
                if not slice:
                    continue
                # file_paths, line_lists, methods = query_slice_info_ext(cpg, slice)
                # if len(file_paths) != len(line_lists):
                #     this may be something wrong in this slice
                # continue
                # label = make_label_per_slice_xml(file_paths, line_lists, label_dict)
                # label = make_label_per_slice_diff(file_paths, line_lists, label_dict)
                # v2
                line_dict, methods = query_slice_info_v2(cpg, testID, slice)
                label = make_label_per_slice_v2(line_dict, label_dict)
                if label == 1:
                    vul_count += 1
                else:
                    non_vul_count += 1
                spg = SPG(testID, name_set, methods, line_set, slice, label, types)
                if len(spg.node_list) > 0:
                    spg_list.append(spg)
            except (KeyError, IndexError, TypeError):
                continue
    return spg_list, vul_count, non_vul_count