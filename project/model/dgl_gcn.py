'''this file include pl dgl_gcn model and dgl_dataset'''
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
import numpy as np
from collections import defaultdict
from scipy import spatial
import dgl
import dgl.nn.pytorch as dglnn
from dgl.data import DGLDataset
from dgl.data.utils import save_graphs, load_graphs, save_info, load_info

class MolecularGCN(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MolecularGNN")
        parser.add_argument("--learning_rate", type=float, default=1e-4)
        parser.add_argument("--N_atoms", type=int, default=8)
        return parent_parser
    
    def __init__(self, N_atoms, dim, **kwargs):
        super().__init__()
        self.lr = kwargs['learning_rate']
        self.save_hyperparameters()
        self.predictions = defaultdict(list)

        # layer parameter
        # conv layer
        dim_l = [dim, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256]
        self.conv_lt = [(dim_l[i], dim_l[i + 1]) for i in range(len(dim_l)-1)]
        self.dense_lt = [(256, 256), (256, 256), (256, 256), (256, 256), (256, 256), (256, 256), (256, 256), (256, 256)]
        dense_out = 256, 1
        # embeding layer
        self.embed_atom = nn.Embedding(N_atoms, dim)
        # torch.nn.init.xavier_uniform_(self.embed_atom.weight)
        self.gamma = nn.ModuleList([nn.Linear(1, 1)
                                    for _ in range(len(self.conv_lt))])
        for i in range(len(self.conv_lt)):
            ones = nn.Parameter(torch.ones((1, 1)))
            self.gamma[i].weight.data = ones
        # Conv layer
        class ConvBn(nn.Module):
            def __init__(self, in_dim, out_dim):
                super().__init__()
                self.conv = dglnn.GraphConv(in_dim, out_dim)
                self.bn = torch.nn.BatchNorm1d(out_dim)
            def forward(self, g, feats, edge_weight):
                return F.relu(self.bn(self.conv(g, feats, edge_weight=edge_weight)))
        self.conv = nn.ModuleList([ConvBn(*i) for i in self.conv_lt])
        # Dense layer
        self.dense = nn.ModuleList([nn.Linear(*i) for i in self.dense_lt])
        # self.dense4 = nn.Linear(*dense4)
        self.dense_out = nn.Linear(*dense_out)

    def forward(self, g, h):
        # emedding
        h = self.embed_atom(h)
        # 应用图卷积和激活函数
        for l in range(len(self.conv_lt)):
            h = self.conv[l](g, h, edge_weight=F.relu(self.gamma[l](g.edata['h'])))

        # h = F.relu(self.bn1(self.conv1(g, h, edge_weight=g.edata['h'])))
        with g.local_scope():
            g.ndata['h'] = h
            # 使用平均读出计算图表示
            hg = dgl.mean_nodes(g, 'h')
            for l in range(len(self.dense_lt)):
                hg = F.relu(self.dense[l](hg))
            predict_property = self.dense_out(hg)
            return torch.squeeze(predict_property, dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        g, label = train_batch
        feats = g.ndata['h']
        predicted_properties = self.forward(g, feats)
        # loss = F.smooth_l1_loss(predicted_properties, train_batch['label'])
        loss = F.mse_loss(predicted_properties, label)
        # self.log(batch_size=len(train_batch['id']))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        g, label = val_batch
        feats = g.ndata['h']
        predicted_properties = self.forward(g, feats)
        loss = F.mse_loss(predicted_properties, label)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        g, label = batch
        feats = g.ndata['h']
        y_hat = self.forward(g, feats)
        loss = F.mse_loss(y_hat, label.float())
        # loss = F.smooth_l1_loss(y_hat, batch['label'].float())
        # self.predictions['id'].extend(batch['id'])
        self.predictions['pred'].extend(y_hat.cpu().numpy())
        self.predictions['true'].extend(label.cpu().numpy())

        return loss


class LigandDataset_dgl(DGLDataset):
    """定义适用于DGL库的数据集"""
    def __init__(self, dataset_name, raw_dir=None, elements_dict=None, save_dir=None, force_reload=False, verbose=False, transform=None, **kwargs):
        self.elements_dict = elements_dict
        super().__init__(name=dataset_name, raw_dir=raw_dir, save_dir=save_dir, force_reload=force_reload, verbose=verbose, transform=transform)
        
    def process(self):
        #将原始数据处理为图、标签
        # read file
        data_path = self.raw_path + '.txt'
        with open(data_path, 'r') as f:
            # first line is property_type
            self._property_types = f.readline().strip().split()
            # split data
            self._data_original = f.read().strip().split('\n\n')
        self.graphs, self.label = self._load_data(self._data_original)
        return self.graphs, self.label
    def save(self):
        '''save the graph list and the labels'''
        graph_path = self.save_path + '_dgl_graph.bin'
        save_graphs(graph_path, self.graphs, {'labels': self.label})
        # 保存其它信息
        info_path = self.save_path + '_info.pkl'
        save_info(info_path, {'elements_dict': dict(self.elements_dict)})
    def load(self):
        # 读取图和标签
        graphs, label_dict = load_graphs(self.save_path + '_dgl_graph.bin')
        self.graphs = graphs
        self.label = label_dict['labels']
        # 读取额外信息
        info_path = self.save_path + '_info.pkl'
        self.elements_dict = load_info(info_path)['elements_dict']

    def has_cache(self):
        # 检查在‘self.save_path'里是否有处理过的数据文件
        graph_path = self.save_path + '_dgl_graph.bin'
        info_path = self.save_path + '_info.pkl'
        return os.path.exists(graph_path) and os.path.exists(info_path)
    
    def __getitem__(self, idx):
        '''Get graph and label by index'''
        return self.graphs[idx], self.label[idx]
    
    def __len__(self):
        '''Number of graphs in the dataset'''
        return len(self.graphs)
    
    def _load_data(self, data_list):
        # data list
        graphs = list()
        label = list()
        id_list = list()
        # make every data
        for data in data_list:
            # get every row
            data = data.strip().split('\n')
            # get data id
            id = data[0]
            # get property in last row
            property = float(data[-1].strip())
            # get atoms and its coordinate
            atoms, atom_coords = [], []
            for atom_xyz in data[1:-1]:
                atom, x, y, z = atom_xyz.split()
                atoms.append(atom)
                xyz = [float(v) for v in [x, y, z]]
                atom_coords.append(xyz)
            # transform symbols to numbers, such as:{'C':0, 'N':1, ...}
            atoms = [self.elements_dict[a] for a in atoms]
            # create distance matrix
            distance_matrix = spatial.distance_matrix(atom_coords, atom_coords)
            distance_matrix = np.where(distance_matrix == 0.0, 1e6, distance_matrix)
            n_s = list()
            for i in atoms: n_s += ([i] * len(atoms))
            g = dgl.graph((torch.tensor(n_s), torch.tensor(atoms*len(atoms))), idtype=torch.int32)
            g.edata['h'] = torch.tensor(distance_matrix.reshape(-1, 1), dtype=torch.float32)
            g.ndata['h'] = g.nodes().clone().detach()
            g = dgl.add_self_loop(g)
            id_list.append(id)
            graphs.append(g)
            label.append(property)
        return graphs, torch.tensor(label)

def collate_fn(batch):
    # 小批次是一个元组(graph, label)列表
    graphs = [e[0] for e in batch]
    g = dgl.batch(graphs)
    labels = [e[1] for e in batch]
    labels = torch.stack(labels, 0)
    return g, labels