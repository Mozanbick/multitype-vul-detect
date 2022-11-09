import torch
import torch.nn as nn
import torch.functional as F
import torchmetrics.functional as MF
import dgl

from dgl import apply_each
from dgl.nn.pytorch import GATConv, HeteroGraphConv


class RGATModel(nn.Module):

    def __init__(self,
                 etypes, in_feats, n_hidden, n_classes, n_heads=4,
                 num_hidden=2, feat_drop=0., attn_drop=0.):
        super().__init__()
        self.layers = nn.ModuleList()
        # build input layer
        self.layers.append(
            HeteroGraphConv({
                etype: GATConv(in_feats, n_hidden // n_heads, n_heads,
                               feat_drop=feat_drop, attn_drop=attn_drop)
                for etype in etypes
            })
        )
        # build hidden layers
        for _ in range(num_hidden):
            self.layers.append(
                HeteroGraphConv({
                    etype: GATConv(n_hidden, n_hidden // n_heads, n_heads,
                                   feat_drop=feat_drop, attn_drop=attn_drop)
                    for etype in etypes
                })
            )
        # build output layer
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(n_hidden, n_classes)

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            h = apply_each(h, lambda x: x.view(x.shape[0], x.shape[1] * x.shape[2]))
            if l != len(self.layers) - 1:
                h = apply_each(h, F.relu)
                h = apply_each(h, self.dropout)
        return self.linear(h['paper'])
