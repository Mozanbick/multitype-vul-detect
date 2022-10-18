class Edge:
    """
    A class for nodes' edges
    """

    def __init__(self, node_in, node_out, type):
        self._node_in = node_in
        self._node_out = node_out
        self._type = type

    @property
    def node_in(self):
        """
        Returns the in-node's id of the edge
        """
        return self._node_in

    @property
    def node_out(self):
        """
        Returns the out-node's id of the edge
        """
        return self._node_out

    @property
    def type(self):
        """
        Returns the type of the edge;
        Can be one of ["AST", "CFG", "CDG", "DDG"]
        """
        return self._type
