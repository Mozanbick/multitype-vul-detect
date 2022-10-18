import torch
import torch.nn.functional as F
from dgl.contrib.data import load_data
from dgl import DGLGraph
from nn.graph_pkg.models import RGCNModel
from dgl.nn.pytorch import TypedLinear


if __name__ == '__main__':
    x = torch.randn(100, 32)
    x_type = torch.randint(0, 5, (100,))
    m = TypedLinear(32, 64, 5, 'basis', 4)
    y = m(x, x_type)
    print(y.shape)
