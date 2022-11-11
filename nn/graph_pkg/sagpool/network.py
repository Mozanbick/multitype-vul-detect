import torch
import torch.nn.functional as F
import dgl

from nn.graph_pkg.sagpool.layer import ConvPoolBlock, SAGPool
from dgl.nn.pytorch import AvgPooling, GraphConv, MaxPooling


class SAGModelHierarchical(torch.nn.Module):
    """
    The Self-Attention Graph Pooling Network with hierarchical readout

    Args:
        in_dim (int): The input node feature dimension.
        hid_dim (int): The hidden dimension for node feature.
        out_dim (int): The output dimension.
        num_convs (int, optional): The number of graph convolution layers. (default 3)
        pool_ratio (float, optional): The pool ratio which determines the amount of nodes
            remain after pooling. (default 0.5)
        dropout (float, optional): The dropout ratio for each layer. (default 0)
    """

    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 out_dim: int,
                 num_convs=3,
                 pool_ratio: float = 0.5,
                 dropout: float = 0.0):
        super(SAGModelHierarchical, self).__init__()
        self.dropout = dropout
        self.num_convpools = num_convs

        convpools = []
        for i in range(num_convs):
            _i_dim = in_dim if i == 0 else hid_dim
            _o_dim = hid_dim
            convpools.append(
                ConvPoolBlock(_i_dim, _o_dim, pool_ratio=pool_ratio)
            )
        self.convpools = torch.nn.ModuleList(convpools)

        self.lin1 = torch.nn.Linear(hid_dim * 2, hid_dim)
        self.lin2 = torch.nn.Linear(hid_dim, hid_dim // 2)
        self.lin3 = torch.nn.Linear(hid_dim // 2, out_dim)

    def forward(self,
                graph: dgl.DGLGraph):
        feat = graph.ndata['feat']
        final_readout = None

        for i in range(self.num_convpools):
            graph, feat, readout = self.convpools[i](graph, feat)
            if final_readout is None:
                final_readout = readout
            else:
                final_readout = final_readout + readout

        feat = F.relu(self.lin1(final_readout))
        feat = F.dropout(feat, p=self.dropout, training=self.training)
        feat = F.relu(self.lin2(feat))
        feat = F.log_softmax(self.lin3(feat), dim=-1)

        return feat


class SAGModelGlobal(torch.nn.Module):
    """
    The Self-Attention Graph Pooling Network with global readout

    Args:
        in_dim (int): The input node feature dimension.
        hid_dim (int): The hidden dimension for node feature.
        out_dim (int): The output dimension.
        num_convs (int, optional): The number of graph convolution layers. (default 3)
        pool_ratio (float, optional): The pool ratio which determines the amount of nodes
            remain after pooling. (default 0.5)
        dropout (float, optional): The dropout ratio for each layer. (default 0)
    """

    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 out_dim: int,
                 num_convs=3,
                 pool_ratio: float = 0.5,
                 dropout: float = 0.0):
        super(SAGModelGlobal, self).__init__()
        self.dropout = dropout
        self.num_convpools = num_convs

        convpools = []
        for i in range(self.num_convpools):
            _i_dim = in_dim if i == 0 else hid_dim
            _o_dim = hid_dim
            convpools.append(GraphConv(_i_dim, _o_dim))
        self.convpools = torch.nn.ModuleList(convpools)

        concat_dim = num_convs * hid_dim
        self.pool = SAGPool(concat_dim, ratio=pool_ratio)
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()

        self.lin1 = torch.nn.Linear(concat_dim * 2, hid_dim)
        self.lin2 = torch.nn.Linear(hid_dim, hid_dim // 2)
        self.lin3 = torch.nn.Linear(hid_dim // 2, out_dim)

    def forward(self,
                graph: dgl.DGLGraph):
        feat = graph.ndata['feat']
        conv_res = []

        for i in range(self.num_convpools):
            feat = self.convpools[i](graph, feat)
            conv_res.append(feat)

        conv_res = torch.cat(conv_res, dim=-1)
        graph, feat, _ = self.pool(graph, conv_res)
        feat = torch.cat(
            [self.avgpool(graph, feat), self.maxpool(graph, feat)], dim=-1
        )

        feat = F.relu(self.lin1(feat))
        feat = F.dropout(feat, p=self.dropout, training=self.training)
        feat = F.relu(self.lin2(feat))
        feat = F.log_softmax(self.lin3(feat), dim=-1)

        return feat


def get_sag_network(net_type: str = "hierarchical"):
    if net_type == "hierarchical":
        return SAGModelHierarchical
    elif net_type == "global":
        return SAGModelGlobal
    else:
        raise ValueError(
            "SAGNetwork type {} is not supported.".format(net_type)
        )
