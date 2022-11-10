class ModelConfig:

    def __init__(self):
        self._embed_dim = 100
        self._nodes_dim = 205
        self._spgs_dir = "./joern/data/spgs"
        self._fpgs_dir = "./joern/data/fpgs"
        self._group = "group0"
        self._dataset = "fan"
        self._vul_ratio = 1
        self._ast_attr_path = "./joern/files/our_map_all.txt"
        self._differ_edges = False
        self._list_etypes = ['AST', 'CFG', 'CDG', 'DDG', 'CALL', 'RET']

    @property
    def embed_dim(self):
        return self._embed_dim

    def set_embed_dim(self, embed_dim: int):
        self._embed_dim = embed_dim

    @property
    def nodes_dim(self):
        return self._nodes_dim

    def set_nodes_dim(self, nodes_dim: int):
        self._nodes_dim = nodes_dim

    @property
    def spgs_dir(self):
        return self._spgs_dir

    def set_spgs_dir(self, spgs_dir: str):
        self._spgs_dir = spgs_dir

    @property
    def fpgs_dir(self):
        return self._fpgs_dir

    def set_fpgs_dir(self, fpgs_dir: str):
        self._fpgs_dir = fpgs_dir

    @property
    def group(self):
        return self._group

    def set_group(self, group: str):
        self._group = group

    @property
    def dataset(self):
        return self._dataset

    def set_dataset(self, dataset: str):
        self._dataset = dataset

    @property
    def vul_ratio(self):
        return self._vul_ratio

    def set_vul_ratio(self, vul_ratio: int):
        self._vul_ratio = vul_ratio

    @property
    def ast_attr_path(self):
        return self._ast_attr_path

    @property
    def differ_edges(self):
        return self._differ_edges

    def set_differ_edges(self, target: bool):
        self._differ_edges = target

    @property
    def list_etypes(self):
        return self._list_etypes


modelConfig = ModelConfig()
