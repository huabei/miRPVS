# -*- coding: UTF-8 -*-
"""
@author:ZNDX
@file:ResGateGraphConv_model.py
@time:2022/10/12
"""
import torch
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import GCN, MLP
from torch.nn import Embedding, Module
from torch_scatter import scatter_sum


class GcnModel(Module):

    def __init__(self, in_channels: int, hidden_channels: int, hidden_layers: int, out_layers: int, out_channels=None, dropout=0, norm=None):
        super().__init__()
        self.embd = Embedding(in_channels, hidden_channels, dtype=torch.float32)
        self.gcn = GCN(hidden_channels, hidden_channels, hidden_layers, hidden_channels, dropout, norm=norm)
        # self.lin = ModuleList([Linear(hidden_channels, hidden_channels) for _ in range(num_layers[1])])
        self.mlp = MLP(in_channels=hidden_channels, hidden_channels=hidden_channels,
                       out_channels=out_channels, num_layers=out_layers)

    def forward(self, batch):
        x = self.embd(batch.x)
        x = self.gcn(x, batch.edge_index)
        x = self.mlp(x)
        return scatter_sum(torch.squeeze(x, dim=1), batch.batch)


if __name__ == '__main__':
    pass
