import dgl
import os
import torch
import random
from os.path import join, exists
from os import listdir
from dgl.data import DGLDataset
from utils.objects import SPG
from dgl.data.utils import save_graphs, load_graphs
from configs import modelConfig as ModelConfig


class GraphDataset(DGLDataset):

    def __init__(self, dataset: str, data_path: str, save_size=10000, test=False):
        self.save_size = save_size
        self.save_idx = 0
        self.save_count = 0
        self.vul_count = 0
        self.non_vul_count = 0
        self.graph_set = []
        self.graph_list = []
        self.label_list = []
        self.test = test

        super(GraphDataset, self).__init__(dataset, raw_dir=data_path)

    def add_graph(self, graph: SPG):
        if not graph:
            return
        self.graph_set.append(graph)
        self.save_count += 1
        # if reaching the max batch size, save to disk
        if self.save_count >= self.save_size:
            self.save()

    def save(self):
        if self.test:
            self.save_test()
        else:
            self.save_train()

    def save_train(self):
        if not self.graph_set:
            return
        # data balance
        vul_idx = []
        non_vul_idx = []
        for idx, g in enumerate(self.graph_set):
            if g.label == 1:
                vul_idx.append(idx)
            else:
                non_vul_idx.append(idx)
        random.shuffle(vul_idx)
        random.shuffle(non_vul_idx)
        if len(non_vul_idx) > len(vul_idx):
            non_vul_idx = non_vul_idx[:len(vul_idx) * ModelConfig.vul_ratio]
        else:
            vul_idx = vul_idx[:len(non_vul_idx) * ModelConfig.vul_ratio]
        g_idx = vul_idx + non_vul_idx
        tmp_graph = []
        tmp_label = []
        for idx in g_idx:
            tmp_graph.append(self.graph_set[idx].g)
            tmp_label.append(self.graph_set[idx].label)
        # save to disk
        if not tmp_graph:
            return
        self.save_idx += 1
        save_path = join(self.save_path, f"batch_graphs_{ModelConfig.group}_{self.save_idx}.bin")
        label_dict = {'labels': torch.tensor(tmp_label)}
        save_graphs(save_path, tmp_graph, label_dict)
        self.graph_set = []
        self.save_count = 0

    def save_test(self):
        if not self.graph_set:
            return
        spg_dict = {}
        for spg in self.graph_set:
            t = spg.testID.strip('#').split('#')[0]
            if t not in spg_dict:
                spg_dict[t] = [[], []]
            spg_dict[t][0].append(spg.g)
            spg_dict[t][1].append(spg.label)
        for t in spg_dict:
            # data balance
            vul_idx = []
            non_vul_idx = []
            for idx, label in enumerate(spg_dict[t][1]):
                if label == 1:
                    vul_idx.append(idx)
                else:
                    non_vul_idx.append(idx)
            random.shuffle(vul_idx)
            random.shuffle(non_vul_idx)
            if len(non_vul_idx) > len(vul_idx):
                non_vul_idx = non_vul_idx[:len(vul_idx) * ModelConfig.vul_ratio]
            else:
                vul_idx = vul_idx[:len(non_vul_idx) * ModelConfig.vul_ratio]
            g_idx = vul_idx + non_vul_idx
            tmp_graph = []
            tmp_label = []
            for idx in g_idx:
                tmp_graph.append(spg_dict[t][0][idx])
                tmp_label.append(spg_dict[t][1][idx])
            if not tmp_graph:
                return 
            save_dir = join(self.save_path, t)
            if not exists(save_dir):
                os.makedirs(save_dir)
            save_path = join(save_dir, f"batch_graphs_{ModelConfig.group}_{self.save_idx}.bin")
            label_dict = {'labels': torch.tensor(tmp_label)}
            save_graphs(save_path, tmp_graph, label_dict)
        self.graph_set = []
        self.save_count = 0

    def load(self):
        for file in listdir(self.raw_path):
            graphs, label_dict = load_graphs(join(self.raw_path, file))
            self.graph_list += graphs
            self.label_list += label_dict['labels']

    def process(self):
        pass

    def __getitem__(self, idx):
        if self.graph_list:
            return self.graph_list[idx], self.label_list[idx]
        elif self.graph_set:
            return self.graph_set[idx].g, self.graph_set[idx].label
        else:
            return None, None

    def __len__(self):
        if self.graph_list:
            return len(self.graph_list)
        elif self.graph_set:
            return len(self.graph_set)
        else:
            return 0

    def statistics(self):
        return self.graph_list[0].ndata['feat'].shape[0], \
               self.graph_list[0].ndata['feat'].shape[1], \
               2,\
               self.save_size
