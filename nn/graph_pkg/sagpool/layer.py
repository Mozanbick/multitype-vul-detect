import torch
import dgl
import torch.nn.functional as F

from dgl.nn.pytorch import AvgPooling, GraphConv, MaxPooling
from nn.graph_pkg.sagpool.utils import topk, get_batch_id


class SAGPool(torch.nn.Module):
    """
    the Self Attention Pooling layer

    Args:
        in_dim(int): The dimension of node feature
        ratio(float, optional): The pool ratio which determines the amount of nodes
            remain after pooling. (default: 0.5)
        conv_op(torch.nn.Module, optional): The graph convolution layer in dgl used to
            compute scale for each node. (default: dgl.nn.GraphConv)
        non_linearity(Callable, optional): The non-linearity function, a pytorch function.
            (default: torch.tanh)
    """

    def __init__(self,
                 in_dim: int,
                 ratio=0.5,
                 conv_op=GraphConv,
                 non_linearity=torch.tanh):
        super(SAGPool, self).__init__()
        self.in_dim = in_dim
        self.ratio = ratio
        # there is an importance score for each node
        # the importance score is then used to filter the top-k rank import nodes
        self.score_layer = conv_op(in_dim, 1)
        self.non_linearity = non_linearity

    def forward(self, graph: dgl.DGLGraph, feature: torch.Tensor):
        score = self.score_layer(graph, feature).squeeze()
        perm, next_batch_num_nodes = topk(
            score,
            self.ratio,
            get_batch_id(graph.batch_num_nodes()),
            graph.batch_num_nodes()
        )
        feature = feature[perm] * self.non_linearity(score[perm]).view(-1, 1)
        graph = dgl.node_subgraph(graph, perm)

        # node_subgraph currently does not support batch-graph,
        # the 'batch_num_nodes' of the result subgraph is None.
        # So we manually set the 'batch_num_nodes' here.
        # Since global pooling has nothing to do with 'batch_num_edges',
        # we can leave it to be None or unchanged.
        graph.set_batch_num_nodes(next_batch_num_nodes)

        return graph, feature, perm


class ConvPoolBlock(torch.nn.Module):
    """
    A combination of GCN layer and SAGPool layer,
    followed by a concatenated (mean||sum) readout operation.
    """

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 pool_ratio=0.8):
        super(ConvPoolBlock, self).__init__()
        self.conv = GraphConv(in_dim, out_dim)
        self.pool = SAGPool(out_dim, ratio=pool_ratio)
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()

    def forward(self,
                graph,
                feature):
        out = F.relu(self.conv(graph, feature))
        graph, out, _ = self.pool(graph, out)
        g_out = torch.cat(
            [self.avgpool(graph, out), self.maxpool(graph, out)], dim=-1
        )
        return graph, out, g_out
