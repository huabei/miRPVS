from torch import nn
import torch
from torch_scatter import scatter_mean
from .model_baseclass import E_GCL, PLBaseModel

# class E_GCL(nn.Module):
#     """
#     E(n) Equivariant Convolutional Layer
#     re
#     """

#     def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False):
#         super(E_GCL, self).__init__()
#         input_edge = input_nf * 2 # source + target dim
#         self.residual = residual
#         self.attention = attention
#         self.normalize = normalize
#         self.coords_agg = coords_agg
#         self.tanh = tanh
#         self.epsilon = 1e-8
#         edge_coords_nf = 1 # edge length

#         self.edge_mlp = nn.Sequential(
#             nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
#             act_fn,
#             nn.Linear(hidden_nf, hidden_nf),
#             act_fn)

#         # 读出层
#         self.node_mlp = nn.Sequential(
#             nn.Linear(hidden_nf + input_nf, hidden_nf), # edge massage + node feature
#             act_fn,
#             nn.Linear(hidden_nf, output_nf))

#         # 产生coord的权重
#         layer = nn.Linear(hidden_nf, 1, bias=False)
#         torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        
#         coord_mlp = []
#         coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
#         coord_mlp.append(act_fn)
#         coord_mlp.append(layer)
#         if self.tanh:
#             coord_mlp.append(nn.Tanh())
#         self.coord_mlp = nn.Sequential(*coord_mlp)

#         if self.attention:
#             self.att_mlp = nn.Sequential(
#                 nn.Linear(hidden_nf, 1),
#                 nn.Sigmoid())

#     def edge_model(self, source, target, radial, edge_attr):
#         # 产生edge的特征
#         if edge_attr is None:  # Unused.
#             out = torch.cat([source, target, radial], dim=1)
#         else:
#             out = torch.cat([source, target, radial, edge_attr], dim=1)
#         out = self.edge_mlp(out)
#         if self.attention:
#             att_val = self.att_mlp(out)
#             out = out * att_val
#         return out

#     def node_model(self, x, edge_index, edge_attr, node_attr):
#         row, col = edge_index
#         agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
#         if node_attr is not None:
#             agg = torch.cat([x, agg, node_attr], dim=1)
#         else:
#             agg = torch.cat([x, agg], dim=1)
#         out = self.node_mlp(agg)
#         if self.residual:
#             out = x + out
#         return out, agg

#     def coord_model(self, coord, edge_index, coord_diff, edge_feat):
#         row, col = edge_index
#         trans = coord_diff * self.coord_mlp(edge_feat)
#         if self.coords_agg == 'sum':
#             agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
#         elif self.coords_agg == 'mean':
#             agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
#         else:
#             raise Exception('Wrong coords_agg parameter' % self.coords_agg)
#         coord = coord + agg
#         return coord # (N, 3) 更新后的坐标

#     def coord2radial(self, edge_index, coord):
#         row, col = edge_index
#         coord_diff = coord[row] - coord[col]
#         radial = torch.sum(coord_diff**2, 1).unsqueeze(1) # 半径

#         if self.normalize:
#             norm = torch.sqrt(radial).detach() + self.epsilon
#             coord_diff = coord_diff / norm

#         return radial, coord_diff # 半径和相对位置

#     def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
#         row, col = edge_index
#         radial, coord_diff = self.coord2radial(edge_index, coord) # 计算半径和相对位置

#         edge_feat = self.edge_model(h[row], h[col], radial, edge_attr) # 产生edge的特征:(E, hidden_nf)
#         coord = self.coord_model(coord, edge_index, coord_diff, edge_feat) # 更新坐标:(N, 3)
#         h, agg = self.node_model(h, edge_index, edge_feat, node_attr) # 产生node的特征:(N, hidden_nf)

#         return h, coord, edge_attr # 新的node特征和坐标


class EgnnMultilable(PLBaseModel):
    def __init__(self,
                 in_node_nf: int=11,
                 hidden_nf: int=64,
                 out_node_nf: int=5,
                 batch_size: int=64,
                 loss: str='smooth_l1',
                 lr: float=5e-4,
                 lr_decay_min_lr: float=0.0,
                 lr_scheduler: str='cosine',
                 lr_t_0: int=2,
                 lr_t_mul: int=2,
                 in_edge_nf=0,
                 act_fn: str='silu',
                 n_layers=4,
                 residual=True,
                 attention=False,
                 normalize=False,
                 tanh=False,
                 weight_decay=0.0,
                 **kwargs):
        '''
        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param in_edge_nf: Number of features for the edge features
        :param loss: Loss function
        :param lr: Learning rate
        :param lr_scheduler: Learning rate scheduler
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        '''

        super().__init__(loss=loss,
                         lr=lr,
                         lr_decay_min_lr=lr_decay_min_lr,
                         lr_scheduler=lr_scheduler,
                         batch_size=batch_size,
                         lr_t_0=lr_t_0,
                         lr_t_mul=lr_t_mul,
                         weight_decay=weight_decay)
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        act_fn = nn.SiLU() if act_fn == 'silu' else nn.ReLU()
        self.embedding_in = nn.Embedding(in_node_nf, self.hidden_nf)
        self.out_linear = nn.Sequential(nn.Linear(hidden_nf, hidden_nf), act_fn, nn.Linear(hidden_nf, hidden_nf))
        self.embedding_out = nn.Sequential(nn.Linear(hidden_nf, hidden_nf), act_fn, nn.Linear(hidden_nf, out_node_nf))
        # self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                act_fn=act_fn, residual=residual, attention=attention,
                                                normalize=normalize, tanh=tanh))
        # self.to(self.device)

    def forward_(self, batch):
        
        h = batch.x
        x = batch.pos
        edges = batch.edge_index
        edge_attr = None
        h = self.embedding_in(h)

        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr)
        h = self.out_linear(h)
        m_x = scatter_mean(h, batch.batch, dim=0)
        p = self.embedding_out(m_x)
        # h 为node的特征
        
        return p, x
    
    def forward(self, batch):
        p, _ = self.forward_(batch)
        return p


if __name__ == "__main__":
    # Dummy parameters
    batch_size = 8
    n_nodes = 4
    n_feat = 1
    x_dim = 3

    # Dummy variables h, x and fully connected edges
    h = torch.ones(batch_size *  n_nodes, n_feat)
    x = torch.ones(batch_size * n_nodes, x_dim)
    edges, edge_attr = get_edges_batch(n_nodes, batch_size)

    # Initialize EGNN
    egnn = Egnn(in_node_nf=n_feat, hidden_nf=32, out_node_nf=1, in_edge_nf=1)

    # Run EGNN
    h, x = egnn(h, x, edges, edge_attr)
