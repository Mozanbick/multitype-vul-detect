import torch
import os
import uuid
import platform
from gensim.models.keyedvectors import Word2VecKeyedVectors
from utils.embeddings import NodesEmbedding, GraphsEmbedding
from utils.objects.cpg import Node, Method, Cpg
from utils.objects.cpg.types import TYPES
from dgl import DGLGraph
from configs import modelConfig as ModelConfig
from typing import List, Set, Dict, Tuple
from utils.embeddings.embed_utils import extract_tokens


def _add_edges_to_map_v2(cur_method: Method, nid: int, slice_list: List, edge_map: Dict[str, List]):
    # ast edges
    if 'AST' in ModelConfig.list_etypes and nid in cur_method.ast_edges:
        for ae in cur_method.ast_edges[nid]:
            if ae in slice_list:
                edge_map["AST"].append((nid, ae))
    # cfg edges
    if 'CFG' in ModelConfig.list_etypes and nid in cur_method.cfg_edges:
        for fe in cur_method.cfg_edges[nid]:
            if fe in slice_list:
                edge_map["CFG"].append((nid, fe))
    # cdg edges
    if 'CDG' in ModelConfig.list_etypes and nid in cur_method.cdg_edges:
        for cde in cur_method.cdg_edges[nid]:
            if cde in slice_list:
                edge_map["CDG"].append((nid, cde))
    # ddg edges
    if 'DDG' in ModelConfig.list_etypes and nid in cur_method.ddg_edges:
        for dde in cur_method.ddg_edges[nid]:
            if dde in slice_list:
                edge_map["DDG"].append((nid, dde))


def _arrange_nodes_adj(methods: List[Method], slice_list: list):
    """
    extract nodes in slice
    arrange adjacency matrix
    """
    node_list = []
    edge_map = {"AST": [], "CFG": [], "CDG": [], "DDG": [], "CALL": [], "RET": []}
    m_stack = []
    for nid in slice_list:
        cur_method = None
        for method in methods:
            if nid in method.node_id_set:
                cur_method = method
                break
        if not cur_method:
            continue
        # not pdg node
        if cur_method.nodes[nid].node_type & TYPES.CDG == 0 and cur_method.nodes[nid].node_type & TYPES.DDG == 0:
            continue
        node_list.append(cur_method.nodes[nid])
        _add_edges_to_map_v2(cur_method, nid, slice_list, edge_map)
        # init methods stack
        if not m_stack:
            m_stack.append([cur_method, nid])
            continue
        if cur_method == m_stack[-1][0]:  # in same method, update position
            m_stack[-1][1] = nid
        else:  # into another function, corresponding to CALL or RET condition
            if len(m_stack) > 1 and cur_method == m_stack[-2][0]:  # RET condition
                edge_map["RET"].append((m_stack[-1][1], m_stack[-2][1]))
                m_stack.pop()
            else:  # CALL condition
                edge_map["CALL"].append((m_stack[-1][1], nid))
                m_stack.append([cur_method, nid])
    return node_list, edge_map


def _arrange_nodes_adj_v2(methods: List[Method], slice: Dict):
    """
    extract nodes in slice
    arrange adjacency matrix
    version 2, 2022/10/11
    """
    node_list = []
    edge_map = {"AST": [], "CFG": [], "CDG": [], "DDG": [], "CALL": [], "RET": []}

    outer_bwd = slice["outer_bwd"]
    inner_bwd = slice["inner_bwd"]
    in_slice = slice["in_slice"]
    cross_fwd = slice["cross_fwd"]
    call = slice["call"]
    ret = slice["ret"]

    def add_edges_4types_key(sd: Dict[str, List]):
        if not sd:
            return
        for nid, s in sd.items():
            cur_method = None
            for method in methods:
                if nid in method.node_id_set:
                    cur_method = method
                    break
            if not cur_method:
                continue
            for pid in s:
                node_list.append(cur_method.nodes[pid])
                _add_edges_to_map_v2(cur_method, pid, s, edge_map)

    def add_edges_4types_value(sd: Dict[str, List]):
        """
        The same with the method above, the only different thing is that this method uses the first item of
        dict value to get the target cpg method while above method uses the key of dict
        """
        if not sd:
            return
        for nid, s in sd.items():
            cur_method = None
            for method in methods:
                if s[0] in method.node_id_set:
                    cur_method = method
                    break
            if not cur_method:
                continue
            for pid in s:
                node_list.append(cur_method.nodes[pid])
                _add_edges_to_map_v2(cur_method, pid, s, edge_map)

    def add_edges_call_ret(sl_call: List[Tuple], sl_ret: List[Tuple]):
        for tp in sl_call:
            edge_map["CALL"].append(tp)
        for tp in sl_ret:
            edge_map["RET"].append(tp)

    add_edges_4types_key(outer_bwd)
    add_edges_4types_value(inner_bwd)
    add_edges_4types_key(in_slice)
    add_edges_4types_value(cross_fwd)
    add_edges_call_ret(call, ret)

    return node_list, edge_map


def _query_line_info_v2(cpg: Cpg, testID, slice: Dict[str, Dict]):
    line_dict = {}
    cover_methods = []
    outer_bwd = slice["outer_bwd"]
    assert len(outer_bwd) <= 1
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
    assert len(in_slice) == 1
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


def _query_nodes_by_filename(cpg: Cpg, testID: str, filename: str):
    methods = cpg.methods[testID]
    for m in methods:
        if m.filename == filename:
            return m
    return None


def _filter_node_by_code(s_nodes: List[str], n_nodes: List[Node]):
    idx = 0
    i = 0
    filter_node = []
    filter_nid = []
    matched = False
    last_node = None
    while i < len(s_nodes):
        if idx >= len(n_nodes):
            break
        node_code = n_nodes[idx].code.strip().replace(';', '')
        src_code = s_nodes[i].strip().replace(';', '')
        if '{' in src_code:
            src_code = src_code[:src_code.index('{')]
        if src_code == '':
            i += 1
            continue
        node_code = set(extract_tokens(node_code))
        src_code = set(extract_tokens(src_code))
        if src_code == node_code:
            filter_node.append(i + 1)
            filter_nid.append(n_nodes[idx].id)
            if not matched:
                matched = True
            i += 1
            idx += 1
            last_node = node_code
        else:
            if matched and last_node:
                if len(src_code.intersection(last_node)) == len(src_code):
                    filter_node.append(i + 1)
                    filter_nid.append(n_nodes[idx].id)
                    i += 1
                else:
                    i += 1
                    matched = False
            else:
                i += 1
                matched = False
    return filter_node, filter_nid


def _filter_edge_by_node(e_list: List, f_nodes: List[int]):
    if not e_list or len(e_list[0]) != 2:
        return []
    new_e_list = []
    for e in e_list:
        if e[0] in f_nodes and e[1] in f_nodes:
            new_e_list.append(e)
    return new_e_list


def modify_paths_for_win(path: str):
    base_dir = os.path.join(os.getcwd(), "joern", "data")
    items = path.split('/')
    cur_path = os.path.join(base_dir, items[-3], items[-2], items[-1])
    return cur_path


class SPG:
    """
    A class for slice program graph structure
    """

    def __init__(self, testID, names: Set[str], methods: List[Method], line_set: Set, slice, label, types: str):
        self.testID = testID
        self.filenames = names
        self.line_set = line_set
        self.slice = slice
        self.type = types
        self.node_list, self.edge_map = _arrange_nodes_adj_v2(methods, slice)

        # assert len(self.node_list) > 0

        if len(self.node_list) > ModelConfig.nodes_dim:
            self.node_list = self.node_list[:ModelConfig.nodes_dim]
        self.label = label
        self.node_vector = None
        self.adj = None
        self.g = None

    def embed(self, nodes_dim, method: Method, w2v_keyed_vectors: Word2VecKeyedVectors):
        graph = DGLGraph()

        node_embed = NodesEmbedding(nodes_dim, w2v_keyed_vectors)
        self.node_vector = node_embed(self.node_list, method)
        edge_embed = GraphsEmbedding(None, self.edge_map)
        self.adj = edge_embed(self.node_list)

        etype_dict = {
            'AST': 0,
            'CFG': 1,
            'CDG': 2,
            'DDG': 3,
            'CALL': 4,
            'RET': 5,
        }

        # add nodes
        graph.add_nodes(nodes_dim)
        graph.ndata['feat'] = self.node_vector
        # add edges
        edge_type = []
        total_elen = 0
        for et, coo in self.adj.items():
            graph.add_edges(coo[0], coo[1])
            edge_type += [etype_dict[et]] * len(coo[0])
            total_elen += len(coo[0])
        graph.edata.update({'rel_type': torch.Tensor(edge_type), 'norm': torch.ones([total_elen, 1])})

        self.g = graph

    def to_sequence(self, cpg: Cpg):
        line_dict, methods = _query_line_info_v2(cpg, self.testID, self.slice)
        outer_bwd = self.slice["outer_bwd"]
        inner_bwd = self.slice["inner_bwd"]
        in_slice = self.slice["in_slice"]
        cross_fwd = self.slice["cross_fwd"]

        # info
        lines = [self.testID]
        for name in self.filenames:
            lines.append(name)
        for line in self.line_set:
            lines.append(str(line))
        lines.append("Label: " + str(self.label))
        lines.append("@@@")
        # collect inner_bwd lines
        li_dict = {}
        call_nodes = []
        for fn, ls in line_dict["inner_bwd"].items():
            method = cpg.get_method_by_filename(self.testID, fn)
            if platform.system() == "Windows":
                fn = modify_paths_for_win(fn)
            with open(fn, "r", encoding="utf-8") as fpi:
                li = fpi.readlines()
            target_li = []
            for l in ls:
                target_li.append(li[l - 1].strip())
            for nid in inner_bwd.keys():
                if nid in method.node_id_set:
                    li_dict[nid] = target_li
                    call_nodes.append(nid)
                    break
        # collect cross_fwd lines
        lc_dict = {}
        for fn, ls in line_dict["cross_fwd"].items():
            method = cpg.get_method_by_filename(self.testID, fn)
            if platform.system() == "Windows":
                fn = modify_paths_for_win(fn)
            with open(fn, "r", encoding="utf-8") as fpc:
                lc = fpc.readlines()
            target_lc = []
            for l in ls:
                target_lc.append(lc[l - 1].strip())
            for nid in cross_fwd.keys():
                if nid in method.node_id_set:
                    lc_dict[nid] = target_lc
                    call_nodes.append(nid)
                    break
        # collect outer_bwd lines
        lo_after = []  # after call in slice lines
        for fn, ls in line_dict["outer_bwd"].items():
            method_o = cpg.get_method_by_filename(self.testID, fn)
            if platform.system() == "Windows":
                fn = modify_paths_for_win(fn)
            with open(fn, "r", encoding="utf-8") as fpo:
                lo = fpo.readlines()
            called = False
            for nido, so in outer_bwd.items():
                for soo in so:
                    line_number = method_o.nodes[soo].line_number
                    # after call in slice
                    if line_number and line_number in ls:
                        if not called:
                            lines.append(lo[line_number - 1].strip())
                        else:
                            lo_after.append(lo[line_number - 1].strip())
                        ls.remove(line_number)
                    # traverse to call node
                    if nido == soo:
                        called = True
        # in slice
        for fni, lsi in line_dict["in_slice"].items():
            method_i = cpg.get_method_by_filename(self.testID, fni)
            if platform.system() == "Windows":
                fni = modify_paths_for_win(fni)
            with open(fni, "r", encoding="utf-8") as fpi:
                lis = fpi.readlines()
            for nidi, si in in_slice.items():
                for sii in si:
                    line_number = method_i.nodes[nidi].line_number
                    # different part
                    if sii in call_nodes:
                        # add sii node first
                        if line_number and line_number in lsi:
                            lines.append(lis[line_number - 1].strip())
                            lsi.remove(line_number)
                        # add inner_bwd or cross_fwd
                        if sii in li_dict:
                            lines += li_dict[sii]
                        elif sii in lc_dict:
                            lines += lc_dict[sii]
                        continue
                    if line_number and line_number in lsi:
                        lines.append(lis[line_number - 1].strip())
                        lsi.remove(line_number)
        # add after call outer_bwd lines last
        lines += lo_after

        return self.type, lines

    def __str__(self):
        r"""
        Print necessary information of this slice program graph
        """
        node_info = ""
        edge_info = ""
        names = "\n".join(self.filenames)
        lines = ", ".join(list(map(str, self.line_set)))
        for node in self.node_list:
            node_info += f'{node.id} = ({node.node_attr}, "{node.code}", {node.line_number})\n'
        for et in self.edge_map:
            for e in self.edge_map[et]:
                edge_info += f"{e[0]} -> {e[1]}, type = {et}\n"
        return f"TestID: {self.testID}\nFiles: {names}\nFocus Line Numbers: {lines}\nLabel: {self.label}\n" + node_info + edge_info

    def generate_new_c_files(self, cpg: Cpg, filename: str, info: dict[str, list], save_dir: str):
        """
        Generate new C file from slice program graph by extracting corresponding code lines
        """
        method = _query_nodes_by_filename(cpg, self.testID, self.filenames.pop())
        if not method:
            return
        node_set = method.node_id_set
        l_node_set = set()
        for node in self.node_list:
            if node.id in node_set:
                l_node_set.add(node.id)
        l_node_set = sorted(list(l_node_set))
        node_set = [method.nodes[idx] for idx in l_node_set]
        content_code = info['code']
        adj_child = info['child']
        adj_next = info['next']
        adj_com = info['com']
        adj_guared = info['guared']
        adj_lexical = info['lexical']
        adj_jump = info['jump']
        attr = info['attr']
        nodes = info['nodes']
        # filtered_nodes, filtered_nid = _filter_node_by_code(nodes, node_set)
        # adj_child = _filter_edge_by_node(adj_child, filtered_nodes)
        # adj_next = _filter_edge_by_node(adj_next, filtered_nodes)
        # adj_com = _filter_edge_by_node(adj_com, filtered_nodes)
        # adj_guared = _filter_edge_by_node(adj_guared, filtered_nodes)
        # adj_lexical = _filter_edge_by_node(adj_lexical, filtered_nodes)
        # adj_jump = _filter_edge_by_node(adj_jump, filtered_nodes)
        # attr = [attr[i-1] for i in filtered_nodes]
        # nodes = [nodes[i-1] for i in filtered_nodes]
        adj_cdfg = []
        node_map = {nid: idx for idx, nid in enumerate(l_node_set)}
        em = {'AST': 0, 'CFG': 1, 'CDG': 2, 'DDG': 2}
        for edge in self.edge_map:
            try:
                if edge not in em:
                    continue
                emap = self.edge_map[edge]
                for e in emap:
                    if e[0] in node_map and e[1] in node_map:
                        adj_cdfg.append((node_map[e[0]], node_map[e[1]], em[edge]))
            except KeyError:
                continue
        node_cdfg = []
        for nid in l_node_set:
            n = method.nodes[nid]
            try:
                code = n.code if n.code is not None else ''
                node_cdfg.append((node_map[n.id], code))
            except KeyError:
                continue
        # rewrite c files
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # filename = str(uuid.uuid4()) + '.txt'
        with open(os.path.join(save_dir, filename), "w") as fp:
            fp.write("-----label-----\n")
            fp.write(str(self.label) + "\n")
            fp.write("-----code-----\n")
            for code in content_code:
                fp.write(code)
            fp.write("-----children-----\n")
            for child in adj_child:
                fp.write(f"{child[0]},{child[1]}\n")
            fp.write("-----nextToken-----\n")
            for nt in adj_next:
                fp.write(f"{nt[0]},")
            if adj_next:
                fp.write(f"{adj_next[-1][1]}\n")
            fp.write("-----computeFrom-----\n")
            for cf in adj_com:
                fp.write(f"{cf[0]},{cf[1]}\n")
            fp.write("-----guardedBy-----\n")
            for gb in adj_guared:
                fp.write(f"{gb[0]},{gb[1]}\n")
            fp.write("-----guardedByNegation-----\n")
            fp.write("-----lastLexicalUse-----\n")
            for lu in adj_lexical:
                fp.write(f"{lu[0]},{lu[1]}\n")
            fp.write("-----jump-----\n")
            for jp in adj_jump:
                fp.write(f"{jp[0]},{jp[1]}\n")
            fp.write("-----attribute-----\n")
            fp.write(",".join(attr))
            fp.write("\n")
            fp.write("-----ast_node-----\n")
            for n in nodes:
                fp.write(n)
            fp.write("-----joern-----\n")
            for e in adj_cdfg:
                fp.write(f"({e[0]},{e[1]},{e[2]})\n")
            fp.write("-----------------------------------\n")
            for nc in node_cdfg:
                fp.write(f"({nc[0]},{nc[1]})\n")
            fp.write("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

    def generate_new_c_files_ext(self, cpg: Cpg, filename: str, filepath: str, save_dir: str):
        method = _query_nodes_by_filename(cpg, self.testID, self.filenames.pop())
        if not method:
            return
        node_set = method.node_id_set
        l_node_set = set()
        for node in self.node_list:
            if node.id in node_set:
                l_node_set.add(node.id)
        l_node_set = sorted(list(l_node_set))
        # load exist info
        with open(filepath, "r") as fr:
            contents = fr.read()
        adj_cdfg = []
        node_map = {nid: idx for idx, nid in enumerate(l_node_set)}
        em = {'AST': 0, 'CFG': 1, 'CDG': 2, 'DDG': 2}
        for edge in self.edge_map:
            try:
                if edge not in em:
                    continue
                emap = self.edge_map[edge]
                for e in emap:
                    if e[0] in node_map and e[1] in node_map:
                        adj_cdfg.append((node_map[e[0]], node_map[e[1]], em[edge]))
            except KeyError:
                continue
        node_cdfg = []
        for nid in l_node_set:
            n = method.nodes[nid]
            try:
                code = n.code if n.code is not None else ''
                node_cdfg.append((node_map[n.id], code))
            except KeyError:
                continue
        # rewrite c files
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filename = str(uuid.uuid4()) + '.txt'
        with open(os.path.join(save_dir, filename), "w") as fp:
            fp.write(contents)
            fp.write("-----joern-----\n")
            for e in adj_cdfg:
                fp.write(f"({e[0]},{e[1]},{e[2]})\n")
            fp.write("-----------------------------------\n")
            for nc in node_cdfg:
                fp.write(f"({nc[0]},{nc[1]})\n")
            fp.write("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
