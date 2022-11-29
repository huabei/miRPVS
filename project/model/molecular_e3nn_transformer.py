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
from e3nn.math import soft_one_hot_linspace, soft_unit_step


class Transformer(torch.nn.Module):
    def __init__(self, irreps_input, irreps_query, irreps_key, irreps_output, number_of_basis=10, max_radius=2.0) -> None:
        super().__init__()
        # Just define arbitrary irreps
        # irreps_input = o3.Irreps([(hidden_channels, (0, 0))])
        # irreps_query = o3.Irreps([(hidden_channels, (0, 0))])
        # irreps_key = o3.Irreps([(hidden_channels, (0, 0))])
        # irreps_output = o3.Irreps([(hidden_channels, (0, 0))])  # also irreps of the values
        # self.hidden_layers = hidden_layers
        # self.out_layers = out_layers
        self.max_radius = max_radius
        # 1. Define the query
        self.h_q = o3.Linear(irreps_input, irreps_query)
        # 2. Define the edge
        self.irreps_sh = o3.Irreps.spherical_harmonics(2)

        # 3. Define the key
        self.tp_k = o3.FullyConnectedTensorProduct(irreps_input, self.irreps_sh, irreps_key, shared_weights=False)
        self.fc_k = nn.FullyConnectedNet([number_of_basis, 16, self.tp_k.weight_numel], act=torch.nn.functional.silu)
        # 4. Define the value
        self.tp_v = o3.FullyConnectedTensorProduct(irreps_input, self.irreps_sh, irreps_output, shared_weights=False)
        self.fc_v = nn.FullyConnectedNet([number_of_basis, 16, self.tp_v.weight_numel], act=torch.nn.functional.silu)
        # 5. Define the dot
        self.dot = o3.FullyConnectedTensorProduct(irreps_query, irreps_key, "0e")

    def forward(self, node_features, edge_src, edge_dst, edge_sh, edge_weight_cutoff, edge_scalars) -> torch.Tensor:

        f = node_features
        # compute the queries (per node), keys (per edge) and values (per edge)
        q = self.h_q(f)
        k = self.tp_k(f[edge_src], edge_sh, self.fc_k(edge_scalars))
        v = self.tp_v(f[edge_src], edge_sh, self.fc_v(edge_scalars))

        # compute the softmax (per edge)
        exp = edge_weight_cutoff[:, None] * self.dot(q[edge_dst], k).exp()  # compute the numerator
        z = scatter(exp, edge_dst, dim=0, dim_size=len(f))  # compute the denominator (per nodes)
        z[z == 0] = 1  # to avoid 0/0 when all the neighbors are exactly at the cutoff
        alpha = exp / z[edge_dst]
        # compute the outputs (per node)
        f_out = scatter(alpha.relu().sqrt() * v, edge_dst, dim=0, dim_size=len(f))
        return F.normalize(f_out, 2, 1)


class MolecularE3nnTransformer(Module):
    """这是使用e3nn的GCN实现"""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, hidden_layers: int, out_layers: int,
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
        self.number_of_basis = 10
        # self.e3nn_conv1 = Convolution((hidden_channels, (1, 1)), [(1, (0, 1)), (1, (1, 1))], [(hidden_channels, (0, 1))],
        #                               num_basis=self.num_basis)
        # self.e3nn_conv2 = Convolution((hidden_channels, (1, 1)), [(1, (0, 1), (1, (1, 1)))], [(12, (1, 1))],
        #                               num_basis=self.num_basis)
        self.hidden_layers = hidden_layers
        self.out_layers = out_layers
        # 2. Define the edge
        self.irreps_sh = o3.Irreps.spherical_harmonics(2)

        self.embd = Embedding(in_channels, hidden_channels, dtype=torch.float32)

        # self.m_gcn = ModuleList([MolecularGCN(in_channels, hidden_channels) for _ in range(hidden_layers)])
        self.m_gcn = ModuleList([Transformer([(hidden_channels, (0, 1))], [(hidden_channels, (0, 1))],
                                             [(hidden_channels, (0, 1))], [(hidden_channels, (0, 1))],
                                             number_of_basis=self.number_of_basis) for _ in range(hidden_layers)])
        self.lin = ModuleList([Linear(hidden_channels, hidden_channels) for _ in range(out_layers)])
        self.w_property = Linear(hidden_channels, out_channels)
        # self.mlp = MLP(in_channels=hidden_channels, hidden_channels=hidden_channels,
        #                out_channels=out_channels, num_layers=out_layers)

    def forward(self, batch: Batch):
        x = self.embd(batch.x)
        edge_length = batch.edge_attr.norm(dim=1)
        # test
        # m_x = scatter_sum(x, batch.batch, dim=0)
        # properties = self.w_property(m_x)
        # return properties
        # e3nn 层
        # num_basis = self.number_of_basis
        edge_length_embedded = soft_one_hot_linspace(
            edge_length,  # edge length
            start=0.0,
            end=self.max_radius,
            number=self.number_of_basis,
            basis='smooth_finite',
            cutoff=True  # goes (smoothly) to zero at `start` and `end`
        )
        edge_length_embedded = edge_length_embedded.mul(self.number_of_basis**0.5)
        edge_weight_cutoff = soft_unit_step(10 * (1 - edge_length / self.max_radius))
        edge_sh = o3.spherical_harmonics(self.irreps_sh, batch.edge_attr, True, normalization='component')
        # irreps_sh = o3.Irreps("0e")
        # # x = self.e3nn_conv1(x, batch.edge_index[0], batch.edge_index[1], batch.edge_attr, edge_length_embedding)
        # # e = self.e3nn_conv2(x, batch.edge_index[0], batch.edge_index[1], batch.edge_attr, edge_length_embedding)
        # edge_sh = o3.spherical_harmonics(irreps_sh, batch.edge_attr, normalize=True, normalization='component')
        for m in range(self.hidden_layers):
            h_x = self.m_gcn[m](x, batch.edge_index[0], batch.edge_index[1], edge_sh, edge_weight_cutoff, edge_length_embedded)
            x = F.relu(h_x) + x
        for i in range(self.out_layers):
            x = torch.relu(self.lin[i](x))
        m_x = scatter_sum(x, batch.batch, dim=0)
        properties = self.w_property(m_x)
        return properties


if __name__ == '__main__':
    pass
