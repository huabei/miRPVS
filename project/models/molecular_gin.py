# -*- coding: UTF-8 -*-
"""
@author:ZNDX
@file:molecular_gnn.py
@time:2022/10/14
"""
import torch
from torch.nn import Linear, Parameter, Embedding, ModuleList, Module
from torch_geometric.nn import MessagePassing, MLP
from torch_geometric.nn.models import GIN
from torch_geometric.utils import add_self_loops, degree
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_scatter import scatter_sum, scatter_mean


class MolecularGin(Module):
    """GIN模型"""

    def __init__(self, in_channels: int, hidden_channels: int, hidden_layers: int, out_layers: int, out_channels=None,
                 dropout=0, norm=None):
        """
        这个是自定义的模型
        :param in_channels: 元素个数
        :param hidden_channels: dim
        :param hidden_layers: 隐藏层个数
        :param out_layers: Dense层个数
        :param out_channels: 输出维度
        :param dropout:
        :param norm:
        """
        super().__init__()
        self.hidden_layers = hidden_layers
        self.out_layers = out_layers

        self.embd = Embedding(in_channels, hidden_channels, dtype=torch.float32)

        # self.m_gcn = ModuleList([MolecularGCN(hidden_channels, hidden_channels) for _ in range(hidden_layers)])
        self.m_gcn = GIN(hidden_channels, hidden_channels, hidden_layers, out_channels=hidden_channels)

        self.lin = ModuleList([Linear(hidden_channels, hidden_channels) for _ in range(out_layers)])
        self.w_property = Linear(hidden_channels, out_channels)
        # self.mlp = MLP(in_channels=hidden_channels, hidden_channels=hidden_channels,
        #                out_channels=out_channels, num_layers=out_layers)

    def forward(self, batch: Batch):
        x = self.embd(batch.x)
        # m_x = scatter_sum(x, batch.batch, dim=0)
        n_x = self.m_gcn(x, batch.edge_index)
        m_x = scatter_mean(n_x, batch.batch, dim=0)
        for i in range(self.out_layers):
            m_x = torch.relu(self.lin[i](m_x))
        properties = self.w_property(m_x)
        return torch.squeeze(properties, dim=1)

if __name__ == '__main__':
    pass
