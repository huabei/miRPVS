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
from torch_scatter import scatter_sum, scatter
from e3nn import o3, nn
from e3nn.math import soft_one_hot_linspace


class Convolution(torch.nn.Module):
    def __init__(self, irreps_in, irreps_sh, irreps_out, num_basis) -> None:
        super().__init__()
        irreps_in = o3.Irreps(irreps_in)
        irreps_sh = o3.Irreps(irreps_sh)
        irreps_out = o3.Irreps(irreps_out)
        tp = o3.FullyConnectedTensorProduct(
            irreps_in1=irreps_in,
            irreps_in2=irreps_sh,
            irreps_out=irreps_out,
            internal_weights=False,
            shared_weights=False,
        )
        self.fc = nn.FullyConnectedNet([num_basis, 256, tp.weight_numel], torch.relu)
        self.tp = tp
        self.irreps_out = self.tp.irreps_out

    def forward(self, node_features, edge_src, edge_dst, edge_sh, edge_scalars) -> torch.Tensor:

        weight = self.fc(edge_scalars)
        edge_features = self.tp(node_features[edge_src], edge_sh, weight)
        node_features = scatter(edge_features, edge_dst, dim=0).div((len(edge_src)/node_features.shape[0])**0.5)
        return node_features


class MolecularGCN(MessagePassing):
    def __init__(self, in_channels, out_channels):
        """
        :param in_channels: N_atoms
        :param out_channels: dim
        """
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.gamma = Linear(out_channels, 1)
        # self.gamma = Embedding(in_channels, 1)
        self.w_atom = Linear(out_channels, out_channels)
        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        # self.gamma.weight.data = Parameter(torch.ones(self.gamma.weight.data.shape))
        self.w_atom.reset_parameters()

    def forward(self, x, x_ori, edge_index, edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # Step 1: Add self-loops to the adjacency matrix.
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        h_x = torch.relu(self.w_atom(x))
        gammas = torch.sigmoid(self.gamma(x))[edge_index[1]]
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


class MolecularE3nnEgcn(Module):
    """这是使用e3nn的GCN实现"""

    def __init__(self, in_channels: int, hidden_channels: int, hidden_layers: int, out_layers: int, out_channels=None,
                 dropout=0, norm=None, max_radius=2.0):
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
        self.max_radius = max_radius
        self.num_basis = 10
        # self.e3nn_conv1 = Convolution((hidden_channels, (1, 1)), [(1, (0, 1)), (1, (1, 1))], [(hidden_channels, (0, 1))],
        #                               num_basis=self.num_basis)
        # self.e3nn_conv2 = Convolution((hidden_channels, (1, 1)), [(1, (0, 1), (1, (1, 1)))], [(12, (1, 1))],
        #                               num_basis=self.num_basis)
        self.hidden_layers = hidden_layers
        self.out_layers = out_layers

        self.embd = Embedding(in_channels, hidden_channels, dtype=torch.float32)

        # self.m_gcn = ModuleList([MolecularGCN(in_channels, hidden_channels) for _ in range(hidden_layers)])
        self.m_gcn = ModuleList([Convolution([(hidden_channels, (0, 1))], [(1, (0, 1)), (1, (1, 1))],
                                             [(hidden_channels, (0, 1))],
                                             num_basis=self.num_basis) for _ in range(hidden_layers)])
        self.lin = ModuleList([Linear(hidden_channels, hidden_channels) for _ in range(out_layers)])
        self.w_property = Linear(hidden_channels, out_channels)
        # self.mlp = MLP(in_channels=hidden_channels, hidden_channels=hidden_channels,
        #                out_channels=out_channels, num_layers=out_layers)

    def forward(self, batch: Batch):
        x = self.embd(batch.x)
        # test
        # m_x = scatter_sum(x, batch.batch, dim=0)
        # properties = self.w_property(m_x)
        # return properties
        # e3nn 层
        num_basis = self.num_basis
        edge_length_embedding = soft_one_hot_linspace(
            batch.edge_attr.norm(dim=1),
            start=0.0,
            end=self.max_radius,
            number=num_basis,
            basis='smooth_finite',
            cutoff=True,
        )
        edge_length_embedding = edge_length_embedding.mul(num_basis ** 0.5)
        irreps_sh = o3.Irreps("0e + 1o")
        # x = self.e3nn_conv1(x, batch.edge_index[0], batch.edge_index[1], batch.edge_attr, edge_length_embedding)
        # e = self.e3nn_conv2(x, batch.edge_index[0], batch.edge_index[1], batch.edge_attr, edge_length_embedding)
        sh = o3.spherical_harmonics(irreps_sh, batch.edge_attr, normalize=True, normalization='component')
        for m in range(self.hidden_layers):
            x = self.m_gcn[m](x, batch.edge_index[0], batch.edge_index[1], sh, edge_length_embedding)
        for i in range(self.out_layers):
            x = torch.relu(self.lin[i](x))
        m_x = scatter_sum(x, batch.batch, dim=0)
        properties = self.w_property(m_x)
        return properties


if __name__ == '__main__':
    pass
