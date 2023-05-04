from torch_geometric.nn import TransformerConv, GAT, LayerNorm, GraphNorm
import torch
from torch.nn import Linear, Parameter, Embedding, ModuleList, Module
from torch_geometric.nn import MessagePassing, MLP
from torch_geometric.utils import add_self_loops, degree
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_scatter import scatter_sum


class MolecularGraphTransformer(Module):

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, hidden_layers: int, out_layers: int,
                 heads: int, dropout=0, norm=None):
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

        self.m_gcn = ModuleList([TransformerConv(hidden_channels, hidden_channels, heads=6, concat=False)
                                for _ in range(hidden_layers)])
        # self.m_gcn = ModuleList([GATv2Conv(hidden_channels, hidden_channels, edge_dim=1, heads=heads, concat=False)
        #                         for _ in range(hidden_layers)])

        self.g_norm = ModuleList([LayerNorm(hidden_channels) for _ in range(hidden_layers)])
        # self.g_norm = ModuleList([GraphNorm(hidden_channels) for _ in range(hidden_layers)])
        self.lin = ModuleList([Linear(hidden_channels, hidden_channels) for _ in range(out_layers)])
        self.w_property = Linear(hidden_channels, out_channels)
        # self.mlp = MLP(in_channels=hidden_channels, hidden_channels=hidden_channels,
        #                out_channels=out_channels, num_layers=out_layers)

    def forward(self, batch: Batch):
        x = self.embd(batch.x)
        for m in range(self.hidden_layers):
            h_x = self.m_gcn[m](x=x, edge_index=batch.edge_index)
            # x = F.normalize(h_x+x, 2, 1)
            x = torch.relu(self.g_norm[m](h_x+x))
        # m_x = scatter_sum(x, batch.batch, dim=0)
        # 获取全局原子的特征
        m_x = x[torch.where(batch.x == 0)]
        for i in range(self.out_layers):
            m_x = torch.relu(self.lin[i](m_x))
            # m_x = F.normalize(m_x, 2, 1)
        properties = self.w_property(m_x)
        return torch.squeeze(properties, dim=1)
