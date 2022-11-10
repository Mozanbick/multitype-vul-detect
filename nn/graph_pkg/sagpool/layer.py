import torch
import dgl
import torch.nn.functional as F

from dgl.nn.pytorch import AvgPooling, GraphConv, MaxPooling


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
        pass
