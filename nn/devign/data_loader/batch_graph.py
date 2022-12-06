import torch
from dgl import DGLGraph
from utils.objects.dataset import GraphDataset


class BatchGraph:
    """
    Identity of batch graph
    """

    def __init__(self):
        self.graph = DGLGraph()
        self.node_number = 0
        self.graphid_to_nodeids = {}
        self.subgraph_number = 0

    def add_subgraph(self, _g):
        assert isinstance(_g, DGLGraph)
        num_new_nodes = _g.number_of_nodes()
