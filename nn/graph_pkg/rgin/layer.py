import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn

from dgl import function as fn
from dgl.utils import expand_as_pair
from dgl.nn.pytorch.linear import TypedLinear


class RGINConv(nn.Module):
    """
    Relational GIN Conv Layer
    """
    def __init__(self,
                 in_feat,
                 out_feat,
                 num_rels,
                 regularizer=None,
                 num_bases=None,
                 apply_func=None,
                 aggregator_type='sum',
                 init_eps=0,
                 learn_eps=False,
                 activation=None):
        super(RGINConv, self).__init__()
        if num_bases == -1:
            num_bases = num_rels
        self.linear_r = TypedLinear(in_feat, in_feat, num_rels, regularizer, num_bases)
        self.apply_func = apply_func
        self._aggregator_type = aggregator_type
        self.activation = activation
        if aggregator_type not in ('sum', 'max', 'mean'):
            raise KeyError(
                'Aggregator type {} not recognized.'.format(aggregator_type))
        # to specify whether eps is trainable or not.
        if learn_eps:
            self.eps = th.nn.Parameter(th.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', th.FloatTensor([init_eps]))

    def message(self, edges):
        """ Message function. """
        m = self.linear_r(edges.src['h'], edges.data['etype'], self.presorted)
        return {'m': m}

    def forward(self,
                graph,
                feat,
                etypes,
                presorted=False,
                edge_weight=None):
        _reducer = getattr(fn, self._aggregator_type)
        self.presorted = presorted
        with graph.local_scope():
            graph.edata['etype'] = etypes
            aggregate_fn = self.message
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')

            feat_src, feat_dst = expand_as_pair(feat, graph)
            graph.srcdata['h'] = feat_src
            # message passing
            graph.update_all(aggregate_fn, _reducer('m', 'neigh'))
            # graph.update_all(aggregate_fn, _reducer('m', 'neigh'))
            rst = (1 + self.eps) * feat_dst + graph.dstdata['neigh']
            if self.apply_func is not None:
                rst = self.apply_func(rst)
            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            return rst
