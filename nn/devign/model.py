import torch
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GatedGraphConv
from torch import nn


class DevignModel(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 edge_types,
                 num_steps=0):
        super(DevignModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_types = edge_types
        self.num_steps = num_steps

        # network construction
        self.ggnn = GatedGraphConv(
            in_feats=input_dim, out_feats=output_dim, n_steps=num_steps, n_etypes=edge_types
        )
        self.conv_l1 = torch.nn.Conv1d(output_dim, output_dim, (3,))
        self.maxpool1 = torch.nn.MaxPool1d(3, stride=2)
        self.conv_l2 = torch.nn.Conv1d(output_dim, output_dim, (1,))
        self.maxpool2 = torch.nn.MaxPool1d(2, stride=2)

        self.concat_dim = input_dim + output_dim
        self.conv_l1_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, (3,))
        self.maxpool1_for_concat = torch.nn.MaxPool1d(3, stride=2)
        self.conv_l2_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, (2,))
        self.maxpool2_for_concat = torch.nn.MaxPool1d(2, stride=2)

        self.mlp_z = nn.Linear(in_features=self.concat_dim, out_features=1)
        self.mlp_y = nn.Linear(in_features=output_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,
                graph: dgl.DGLGraph,
                edge_types):
        feature = graph.ndata['feat']
        outputs = self.ggnn(graph, feature, edge_types)
        ci = torch.cat((outputs, feature), dim=-1)
        Y_1 = self.maxpool1(
            F.relu(
                self.conv_l1(outputs.transpose(1, 2))
            )
        )
        Y_2 = self.maxpool2(
            F.relu(
                self.conv_l2(Y_1)
            )
        ).transpose(1, 2)
        Z_1 = self.maxpool1_for_concat(
            F.relu(
                self.conv_l1_for_concat(ci.transpose(1, 2))
            )
        )
        Z_2 = self.maxpool2_for_concat(
            F.relu(
                self.conv_l2_for_concat(Z_1)
            )
        ).transpose(1, 2)
        before_avg = torch.mul(self.mlp_y(Y_2), self.mlp_z(Z_2))
        avg = before_avg.mean(dim=1)
        result = self.sigmoid(avg).squeeze(dim=-1)
        return result


class GGNNSum(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 edge_types,
                 num_steps=8):
        super(GGNNSum, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_types = edge_types
        self.num_steps = num_steps
        self.ggnn = GatedGraphConv(
            in_feats=input_dim, out_feats=output_dim, n_steps=num_steps, n_etypes=edge_types
        )
        self.classifier = nn.Linear(in_features=output_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,
                graph: dgl.DGLGraph,
                edge_types):
        feature = graph.ndata['feat']
        outputs = self.ggnn(graph, feature, edge_types)
        ggnn_sum = self.classifier(outputs.sum(dim=1))
        result = self.sigmoid(ggnn_sum).squeeze(dim=-1)
        return result
