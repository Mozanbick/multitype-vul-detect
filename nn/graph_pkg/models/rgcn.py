import torch
import torch.nn as nn
import dgl
import torch.nn.functional as F
from functools import partial
from nn.graph_pkg.layers import RGCNLayer
from dgl.nn.pytorch import RelGraphConv, MaxPooling, GlobalAttentionPooling


class RGCNModel(nn.Module):

    def __init__(self,
                 num_nodes, h_dim, out_dim, num_relations,
                 num_bases=-1, num_hidden_layers=1, regularizer='basis',
                 dropout=0., self_loop=False, ns_mode=False):
        super(RGCNModel, self).__init__()

        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.regularizer = regularizer
        self.dropout = nn.Dropout(dropout)
        self.self_loop = self_loop
        self.ns_mode = ns_mode

        if self.num_bases == -1:
            self.num_bases = self.num_relations

        # create rgcn layers
        self.build_model()

        # create initial features
        self.features = self.create_features()

    def build_model(self):
        self.h_layers = nn.ModuleList()
        # input to hidden
        i2h = self.build_input_layer()
        self.in_layer = i2h
        # hidden to hidden
        for _ in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer()
            self.h_layers.append(h2h)
        # hidden to output
        h2o = self.build_output_layer()
        self.out_layer = h2o
        # MaxPooling
        # self.readout = MaxPooling()
        # GlobalAttentionPooling
        gate_nn = nn.Linear(self.out_dim, 1)
        self.readout = GlobalAttentionPooling(gate_nn)

    # initialize feature for each node
    def create_features(self):
        features = torch.arange(self.num_nodes)
        return features

    def build_input_layer(self):
        # return RGCNLayer(self.num_nodes, self.h_dim, self.num_relations, self.num_bases,
        #                  activation=F.relu, is_input_layer=True)
        return RelGraphConv(self.num_nodes, self.h_dim, self.num_relations, self.regularizer,
                            self.num_bases, self_loop=self.self_loop)

    def build_hidden_layer(self):
        # return RGCNLayer(self.h_dim, self.h_dim, self.num_relations, self.num_bases,
        #                  activation=F.relu)
        return RelGraphConv(self.h_dim, self.h_dim, self.num_relations, self.regularizer,
                            self.num_bases, self_loop=self.self_loop)

    def build_output_layer(self):
        # return RGCNLayer(self.h_dim, self.out_dim, self.num_relations, self.num_bases,
        #                  activation=partial(F.softmax, dim=1))
        return RelGraphConv(self.h_dim, self.out_dim, self.num_relations, self.regularizer,
                            self.num_bases, self_loop=self.self_loop)

    # def forward(self, g):
    #     if self.features is not None:
    #         g.ndata['id'] = self.features
    #     for layer in self.layers:
    #         layer(g)
    #     return g.ndata.pop('h')

    def forward(self, g):
        if self.ns_mode:
            # forward for neighbor sampling
            x = g[0].ndata['feat']
            # input layer
            h = self.in_layer(g[0], x, g[0].edata['rel_type'], g[0].edata['norm'])
            h = self.dropout(F.relu(h))
            # hidden layers
            for idx, layer in enumerate(self.h_layers):
                h = layer(g[idx+1], h, g[idx+1].edata['rel_type'], g[idx+1].edata['norm'])
                h = self.dropout(F.relu(h))
            # output layer
            idx = len(self.h_layers) + 1
            h = self.out_layer(g[idx], h, g[idx].edata['rel_type'], g[idx].edata['norm'])
            return h
        else:
            x = g.ndata['feat']
            # input layer
            h = self.in_layer(g, x, g.edata['rel_type'], g.edata['norm'])
            h = self.dropout(F.relu(h))
            # hidden layers
            for idx, layer in enumerate(self.h_layers):
                h = layer(g, h, g.edata['rel_type'], g.edata['norm'])
                h = self.dropout(F.relu(h))
            # output layer
            h = self.out_layer(g, h, g.edata['rel_type'], g.edata['norm'])
            # readout function
            h = self.readout(g, h)
            return h
