import torch
from gensim.models.keyedvectors import Word2VecKeyedVectors
from utils.embeddings import NodesEmbedding, GraphsEmbedding
from utils.objects.cpg import Node, Method
from utils.objects.cpg.types import TYPES
from dgl import DGLGraph
from configs import modelConfig as ModelConfig
from typing import List


def _arrange_nodes_adj(method: Method, slice_list: list):
    """
    extract nodes in function
    arrange adjacency matrix
    """
    node_set = set()
    edge_map = {"AST": [], "CFG": [], "CDG": [], "DDG": [], "CALL": [], "RET": []}
    for nid in slice_list:
        if nid in method.node_id_set:
            if method.nodes[nid].node_type & TYPES.CDG == 0 and method.nodes[nid].node_type & TYPES.DDG == 0:
                continue
            if not method.nodes[nid].code:
                continue
            node_set.update(method.get_ast_subtree_nodes(nid))
            # ast edges
            if 'AST' in ModelConfig.list_etypes and nid in method.ast_edges:
                for ae in method.ast_edges[nid]:
                    if ae in slice_list:
                        edge_map["AST"].append((nid, ae))
            # cfg edges
            if 'CFG' in ModelConfig.list_etypes and nid in method.cfg_edges:
                for fe in method.cfg_edges[nid]:
                    if fe in slice_list:
                        edge_map["CFG"].append((nid, fe))
            # cdg edges
            if 'CDG' in ModelConfig.list_etypes and nid in method.cdg_edges:
                for cde in method.cdg_edges[nid]:
                    if cde in slice_list:
                        edge_map["CDG"].append((nid, cde))
            # ddg edges
            if 'DDG' in ModelConfig.list_etypes and nid in method.ddg_edges:
                for dde in method.ddg_edges[nid]:
                    if dde in slice_list:
                        edge_map["DDG"].append((nid, dde))
    node_list = [method.nodes[nid] for nid in node_set]
    return node_list, edge_map


class FPG:
    """
    A class for function program graph structure
    """

    def __init__(self, testID, method: Method, slice: list, label):
        self.testID = testID
        self.filename = method.filename
        self.node_list, self.edge_map = _arrange_nodes_adj(method, slice)

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

    def __str__(self):
        r"""
        Print necessary information of this slice program graph
        """
        node_info = ""
        edge_info = ""
        for node in self.node_list:
            node_info += f'{node.id} = ({node.node_attr}, "{node.code}", {node.line_number})\n'
        for et in self.edge_map:
            for e in self.edge_map[et]:
                edge_info += f"{e[0]} -> {e[1]}, type = {et}\n"
        return f"TestID: {self.testID}\nFilename: {self.filename}\nLabel: {self.label}\n" + node_info + edge_info
