# -*- coding: UTF-8 -*-
"""
@author:ZNDX
@file:molecular_gnn.py
@time:2022/10/14
"""
import torch
from torch.nn import Linear, Parameter, Embedding, ModuleList, Module
from torch_geometric.nn import MessagePassing, MLP
from torch_geometric.utils import add_self_loops, degree
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_scatter import scatter_sum, scatter_mean


class MolecularGCN(MessagePassing):
    def __init__(self, in_channels, out_channels):
        """
        :param in_channels: N_atoms
        :param out_channels: dim
        """
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.gamma = Embedding(in_channels, 1)
        self.w_atom = Linear(out_channels, out_channels)
        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.weight.data = Parameter(torch.ones(self.gamma.weight.data.shape))
        # self.w_atom.reset_parameters()

    def forward(self, x, x_ori, edge_index, edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # Step 1: Add self-loops to the adjacency matrix.
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        h_x = torch.relu(self.w_atom(x))
        gammas = torch.sigmoid(self.gamma(x_ori))[edge_index[1]]
        # 计算边的权重
        edge_weight = torch.exp(-gammas * (edge_attr**2))

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=h_x, edge_weight=edge_weight)

        return F.normalize(out + x, 2, 1)

    def message(self, x_j, edge_weight):
        # x_j has shape [E, out_channels]
        # x_j是j节点的特征，即edge_index的第二行的节点的特征
        # Step 4: Normalize node features.
        return edge_weight * x_j

    # def update(self, inputs: Tensor, x) -> Tensor:
    #     return F.normalize(inputs + x, p=2, dim=1)


class MolecularGnnAddxyz2(Module):
    """这是MolecularGNN那篇文献的pyg实现，加上了分子的位置信息。"""

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

        self.m_gcn = ModuleList([MolecularGCN(in_channels, hidden_channels) for _ in range(hidden_layers)])

        self.lin = ModuleList([Linear(hidden_channels+3, hidden_channels+3) for _ in range(out_layers)])
        self.w_property = Linear(hidden_channels+3, out_channels)
        # self.mlp = MLP(in_channels=hidden_channels, hidden_channels=hidden_channels,
        #                out_channels=out_channels, num_layers=out_layers)

    def forward(self, batch: Batch):
        x = self.embd(batch.x)
        for m in range(self.hidden_layers):
            x = self.m_gcn[m](x=x, x_ori=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr)
        # xyz = scatter_mean(batch.pos, batch.batch, dim=0)
        x = torch.cat((x, batch.pos), dim=1)
        for i in range(self.out_layers):
            x = torch.relu(self.lin[i](x))
        m_x = scatter_sum(x, batch.batch, dim=0)
        # m_x = torch.cat((m_x, xyz), dim=1)
        properties = self.w_property(m_x)
        return properties


if __name__ == '__main__':
    pass
