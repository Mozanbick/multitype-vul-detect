import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import dgl.nn.functional as fn

from dgl.nn.pytorch import GINConv, RelGraphConv
from nn.graph_pkg.layers import RGINConv
from dgl.nn.pytorch.glob import SumPooling


class MLP(nn.Module):
    """
    Construct two-layer MLP-type aggregator for GIN model
    """

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        h = x
        h = F.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)


class RGINModel(nn.Module):
    """
    Relational GIN Model
    """

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 num_relations,
                 num_layers=5,
                 learn_eps=False,
                 dropout=0.5,
                 regularizer='basis',
                 num_bases=-1):
        super().__init__()
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(num_layers - 1):
            if layer == 0:
                mlp = MLP(input_dim, hidden_dim, hidden_dim)
                self.ginlayers.append(RGINConv(input_dim, hidden_dim, num_relations,
                                               regularizer=regularizer, num_bases=num_bases,
                                               apply_func=mlp, learn_eps=learn_eps))
            else:
                mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
                self.ginlayers.append(RGINConv(hidden_dim, hidden_dim, num_relations,
                                               regularizer=regularizer, num_bases=num_bases,
                                               apply_func=mlp, learn_eps=learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        # linear functions for graph sum pooling of output of each layer
        self.linear_prediction = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linear_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linear_prediction.append(nn.Linear(hidden_dim, output_dim))
        self.drop = nn.Dropout(dropout)
        self.pool = SumPooling()

    def forward(self, g, h):
        # list of hidden representation at each layer
        hidden_rep = [h]
        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h, g.edata['rel_type'])
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        score_over_layer = 0
        # perform graph sum pooling over all nodes in each layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linear_prediction[i](pooled_h))
        return score_over_layer
