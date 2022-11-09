import numpy as np
import torch
import json
from configs import modelConfig as ModelConfig
from utils.objects.cpg import Node, Method
from utils.embeddings.embed_utils import extract_tokens
from typing import List, Dict, Any
from gensim.models.keyedvectors import Word2VecKeyedVectors

node_type_embed = {
    'CALL': 1.,
    'BLOCK': 2.,
    'CONTROL_STRUCTURE': 3.,
    'IDENTIFIER': 4.,
    'EXPRESSION': 5.,
    'LITERAL': 6.,
    'METHOD_REF': 7.,
    'LOCAL': 8.,
    'MODIFIER': 9.,
    'RETURN': 10.,
    'TYPE_REF': 11.,
    'UNKNOWN': 12.,
}


def _load_ast_attr_map():
    with open(ModelConfig.ast_attr_path, "r") as fp:
        s = fp.read()
        s = s.replace('\'', '\"')
        mapping = json.loads(json.dumps(eval(s)))
    mapping["Unknown"] = [0.] * 100
    return mapping


class NodesEmbedding:
    def __init__(self, nodes_dim: int, w2v_keyed_vectors: Word2VecKeyedVectors):
        """

        :param nodes_dim: max node length in a graph
        :param w2v_keyed_vectors: a map: token -> vector
        """
        self.w2v_keyed_vectors = w2v_keyed_vectors
        self.kv_size = w2v_keyed_vectors.vector_size
        self.nodes_dim = nodes_dim
        self.ast_attr_map = _load_ast_attr_map()

        assert self.nodes_dim >= 0

        # buffer for embeddings with padding
        self.target = torch.zeros([self.nodes_dim, self.kv_size + 100]).float()

    def __call__(self, nodes: List[Node], method: Method):
        nodes_embeddings = self.embed_nodes(nodes, method)
        nodes_tensor = torch.from_numpy(nodes_embeddings).float()

        # fix length
        self.target[:nodes_tensor.size(0), :] = nodes_tensor
        
        return self.target

    def get_vectors(self, tokens):
        vectors = []

        for token in tokens:
            if token in self.w2v_keyed_vectors:
                vectors.append(self.w2v_keyed_vectors[token])
            else:
                vectors.append(np.zeros([self.kv_size], dtype=np.float))

        return vectors

    def embed_nodes(self, nodes: List[Node], method: Method):
        embeddings = []

        for node in nodes:
            # get node's code
            node_code = node.code
            if not node_code:
                continue
            # tokenize
            tokens = extract_tokens(node_code, method)
            if not tokens:
                continue
            # get each token's learned embedding vector
            vector = np.array(self.get_vectors(tokens))
            # the node's source embedding is the average of it's token embeddings
            # TODO: utilize attention mechanism
            source_embedding = np.mean(vector, axis=0)
            # TODO: concatenate the node type with the source embeddings
            n_type = self.ast_attr_map[node.node_attr]
            embedding = np.concatenate((np.array(n_type), source_embedding), axis=0)
            embeddings.append(embedding)

        return np.array(embeddings)


class GraphsEmbedding:
    def __init__(self, edge_type, edge_map: Dict[Any, Any]):
        self.edge_type = edge_type
        self.edge_map = edge_map

    def __call__(self, nodes: List[Node]):
        if "AST" in self.edge_map:
            connections = self.nodes_connectivity_multi(nodes)
        else:
            connections = self.nodes_connectivity(nodes)

        return connections

    def nodes_connectivity(self, nodes: List[Node]):
        """
        encoding adjacency matrix
        """
        coo = [[], []]
        node_map = {node.id: idx for idx, node in enumerate(nodes)}
        node_id_set = {node.id for node in nodes}

        for et, em in self.edge_map.items():
            for edge in em:
                if edge[0] in node_id_set and edge[1] in node_id_set:
                    coo[0].append(node_map[edge[0]])
                    coo[1].append(node_map[edge[1]])

        return coo

    def nodes_connectivity_multi(self, nodes: List[Node]):
        """
        encoding adjacency matrix
        ## Multi edge types and embed in one adjacency matrix
        """
        coo_dict = {}
        node_map = {node.id: idx for idx, node in enumerate(nodes)}
        node_id_set = {node.id for node in nodes}

        for et, em in self.edge_map.items():
            coo = [[], []]
            for edge in em:
                if edge[0] in node_id_set and edge[1] in node_id_set:
                    coo[0].append(node_map[edge[0]])
                    coo[1].append(node_map[edge[1]])
            coo_dict[et] = coo

        return coo_dict
