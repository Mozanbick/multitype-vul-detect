import sys

from nn.graph_pkg.rgcn_cli import main as rgcn_main
from nn.graph_pkg.rgin_cli import main as rgin_main
from nn.graph_pkg.sagpool_cli import main as sagpool_main


if __name__ == '__main__':
    model = sys.argv[1]
    if model == "rgcn":
        rgcn_main()
    elif model == "rgin":
        rgin_main()
    elif model == "sagpool":
        sagpool_main()
