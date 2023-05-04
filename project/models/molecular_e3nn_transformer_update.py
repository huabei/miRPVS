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
from torch_scatter import scatter_mean, scatter
from e3nn import o3, nn
from e3nn.math import soft_one_hot_linspace, soft_unit_step




class TransformerUpdate(torch.nn.Module):
    def __init__(self, irreps_input, irreps_query, irreps_key, irreps_output, num_heads=1) -> None:
        super().__init__()
        # Just define arbitrary irreps
        self.num_heads = num_heads
        # 1. Define the query
        self.h_q = o3.Linear(irreps_input, irreps_query)
        # 2. Define the key and value
        self.h_k = o3.Linear(irreps_input, irreps_key)
        self.h_v = o3.Linear(irreps_input, irreps_output)
        # 激活函数Normalization
        self.na = nn.NormActivation(irreps_output, torch.sigmoid, epsilon=1e-5)
        # 2. Define the edge
        # self.irreps_sh = o3.Irreps.spherical_harmonics(2)
        # 计算注意力分数
        self.dot = o3.FullyConnectedTensorProduct(irreps_query, irreps_key, f"{num_heads}x0e", irrep_normalization='component')

    def forward(self, node_features, edge_dst, edge_src) -> torch.Tensor:
        """node fature: [N, F] F: ele_dim + 1 + 3 + 5
        """

        f = node_features
        # compute the queries (per node), keys (per edge) and values (per edge)
        q = self.na(self.h_q(f))
        k = self.na(self.h_k(f[edge_src]))
        v = self.na(self.h_v(f[edge_src]))
        # compute the softmax (per edge)
        exp = self.dot(q[edge_dst], k).exp()  # compute the numerator
        exp = torch.mean(exp, dim=1, keepdim=True)  # compute the denominator
        z = scatter(exp, edge_dst, dim=0, dim_size=len(f))  # compute the denominator (per nodes)
        z[z == 0] = 1  # to avoid 0/0 when all the neighbors are exactly at the cutoff
        alpha = exp / z[edge_dst]
        # compute the outputs (per node)
        f_out = scatter(alpha.relu().sqrt() * v, edge_dst, dim=0, dim_size=len(f))
        return f_out + v


class MolecularE3nnTransformerUpdate(Module):
    """这是使用e3nn的GCN实现"""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, hidden_layers: int, out_layers: int,
                 dropout=0, norm=None, num_heads=1):
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
        # self.number_of_basis = 10
        self.hidden_layers = hidden_layers
        self.out_layers = out_layers
        self.embd = Linear(10, 10)
        # 2. Define the edge
        self.irreps_sh = o3.Irreps.spherical_harmonics(2)
        irreps_input = [(11, (0, 1)), (1, (1, 1)), (1, (2, 1))]
        irreps_query = [(11, (0, 1)), (in_channels+1, (1, 1)), (in_channels+1, (2, 1))]
        irreps_key = [(11, (0, 1)), (in_channels+1, (1, 1)), (in_channels+1, (2, 1))]
        irreps_output = [(11, (0, 1)), (in_channels+1, (1, 1)), (in_channels+1, (2, 1))]
        self.et = TransformerUpdate(irreps_input, irreps_query, irreps_key, irreps_output, num_heads)
        self.m_et = ModuleList([TransformerUpdate(irreps_output, irreps_query, irreps_key, irreps_output, num_heads)
                                for _ in range(hidden_layers)])
        # O3Linear
        self.ol = o3.Linear(irreps_output, [(hidden_channels, (0, 1))])
        # self.na = nn.NormActivation(irreps_output, [torch.relu] * 3)
        self.lin = ModuleList([Linear(hidden_channels, hidden_channels) for _ in range(out_layers)])
        self.out = Linear(hidden_channels, 9)
        # self.mlp = MLP(in_channels=hidden_channels, hidden_channels=hidden_channels,
        #                out_channels=out_channels, num_layers=out_layers)

    def forward(self, batch: Batch):
        x = batch.x
        x = self.embd(batch.x)

        edge_sh = o3.spherical_harmonics(self.irreps_sh, batch.edge_attr, True, normalization='component')
        # print(batch.x.shape, edge_sh.shape)
        x = torch.concat((x, edge_sh), dim=1)
        x = self.et(x, batch.edge_index[1], batch.edge_index[0])
        for m in range(self.hidden_layers):
            h_x = self.m_et[m](x, batch.edge_index[1], batch.edge_index[0])
            x = h_x + x
        x = F.normalize(self.ol(x), 2, 1)
        # x = F.relu(x)
        for i in range(self.out_layers):
            x = torch.relu(self.lin[i](x))
        m_x = scatter_mean(x[batch.edge_index[1]], batch.batch, dim=0)
        out = F.softmax(self.out(m_x), dim=1)
        # print(out)
        # raise Exception
        return out


if __name__ == '__main__':
    from project.data.zinc_complex3a6p_data_e3nn_subgraph import ZincComplex3a6pDataE3nnSubgraph
    from torch_geometric.data import DataLoader
    dataset = ZincComplex3a6pDataE3nnSubgraph(data_dir='../data/3a6p/zinc_drug_like_100k/3a6p_pocket5_202020')
    dataloder = DataLoader(dataset, batch_size=512, shuffle=True)
    for batch in dataloder:
        model = MolecularE3nnTransformerUpdate(64, 256, 10, 4, 2)
        out = model(batch)
        print(out)
        break
    pass
