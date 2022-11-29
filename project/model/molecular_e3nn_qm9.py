# -*- coding: UTF-8 -*-
"""
@author:ZNDX
@file:molecular_gnn.py
@time:2022/10/14
"""
from math import pi

import torch
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.o3 import TensorProduct, FullyConnectedTensorProduct
from e3nn.nn import Gate, FullyConnectedNet
from torch_geometric.nn import radius_graph
from torch_scatter import scatter
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def print_std(name, x):
    pass
    # print(f"{name}{list(x.shape)}: {x.mean(0).abs().mean():.2f} +- {x.var(0).mean().sqrt():.1f}")


class MolecularE3nnQm9(torch.nn.Module):
    def __init__(
            self,
            muls=(256, 16, 0),
            lmax=2,
            num_layers=3,
            cutoff=10.0,
            rad_gaussians=50,
            rad_hs=(128, 128),
            num_neighbors=20,
            num_atoms=20,
            mean=None,
            std=None,
            scale=None,
            atomref=None
    ):
        super().__init__()

        self.cutoff = cutoff
        self.mean = mean
        self.std = std
        self.scale = scale
        self.num_neighbors = num_neighbors
        self.num_atoms = num_atoms
        self.rad_gaussians = rad_gaussians
        self.cutoff = cutoff

        # self.radial = FullyConnectedNet((rad_gaussians, ) + rad_hs, swish, variance_in=1 / rad_gaussians, out_act=True)
        self.radial = FullyConnectedNet((self.rad_gaussians, )+ rad_hs, act=torch.nn.functional.silu)
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)  # spherical harmonics representation
        # self.irreps_edge = o3.Irreps([(25, l, (-1)**l) for l in range(lmax + 1)])
        self.irreps_edge = self.irreps_sh

        irreps = o3.Irreps([(muls[0], (0, 1)), (muls[1], (1, -1)), (muls[2], (2, 1))])
        self.mul_node = FullyConnectedTensorProduct([(10, "0e")], self.irreps_sh, irreps)

        modules = []
        for _ in range(num_layers):
            act = make_gated_block(irreps, muls, self.irreps_sh)
            conv = Conv(irreps, act.irreps_in, self.irreps_edge, rad_hs[-1])
            irreps = act.irreps_out.simplify()

            modules += [torch.nn.ModuleList([conv, act])]

        self.layers = torch.nn.ModuleList(modules)

        self.irreps_out = o3.Irreps("0e + 0o")
        self.layers.append(Conv(irreps, self.irreps_out, self.irreps_edge, rad_hs[-1]))

        self.register_buffer('atomref', atomref)

    def forward(self,batch):
        z = batch.x
        pos = batch.pos
        batch = batch.batch
        assert z.dim() == 1 and z.dtype == torch.long
        assert pos.dim() == 2 and pos.shape[1] == 3
        batch = torch.zeros_like(z) if batch is None else batch

        edge_src, edge_dst = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=1000)
        edge_vec = pos[edge_src] - pos[edge_dst]
        edge_sh = o3.spherical_harmonics(self.irreps_sh, edge_vec, True, 'component')
        edge_len = edge_vec.norm(dim=1)
        edge_len_emb = self.radial(soft_one_hot_linspace(edge_len, 0.0, self.cutoff, self.rad_gaussians,
                                                         cutoff=True,
                                                         basis='smooth_finite'))

        edge_c = (pi * edge_len / self.cutoff).cos().add(1).div(2)
        edge_sh = edge_c[:, None] * edge_sh / self.num_neighbors**0.5

        # z : [1, 6, 7, 8, 9] -> [0, 1, 2, 3, 4]
        # node_z = z.new_tensor([-1, 0, -1, -1, -1, -1, 1, 2, 3, 4])[z]
        # edge_zz = 5 * node_z[edge_src] + node_z[edge_dst]

        node_z = torch.nn.functional.one_hot(z, 10).mul(10**0.5)
        # edge_zz = torch.nn.functional.one_hot(edge_zz, 25).mul(5.0)

        # edge_attr = self.mul(edge_zz, edge_sh)
        edge_attr = edge_sh

        h = scatter(edge_sh, edge_src, dim=0, dim_size=len(pos))
        h[:, 0] = 1
        h = self.mul_node(node_z, h)

        print_std('h', h)

        for conv, act in self.layers[:-1]:
            # with torch.autograd.profiler.record_function("Layer"):
            h = conv(h, node_z, edge_src, edge_dst, edge_len_emb, edge_attr)  # convolution
            print_std('post conv', h)
            h = act(h)  # gate non linearity
            print_std('post gate', h)

        # with torch.autograd.profiler.record_function("Layer"):
        h = self.layers[-1](h, node_z, edge_src, edge_dst, edge_len_emb, edge_attr)

        print_std('h out', h)

        s = 0
        for i, (mul, (l, p)) in enumerate(self.irreps_out):
            assert mul == 1 and l == 0
            if p == 1:
                s += h[:, i]
            if p == -1:
                s += h[:, i].pow(2).mul(0.5)  # odd^2 = even
        h = s.view(-1, 1)

        print_std('h out+', h)

        # for the scatter we normalize
        h = h / self.num_atoms**0.5

        if self.mean is not None and self.std is not None:
            h = h * self.std + self.mean

        if self.atomref is not None:
            h = h + self.atomref[z]
        # for target=7, MAE of 75eV

        out = scatter(h, batch, dim=0)

        if self.scale is not None:
            out = self.scale * out

        return out


def make_gated_block(irreps_in, muls, irreps_sh):
    """
    Make a Gate assuming many things
    """
    irreps_available = [
        (l, p_in * p_sh)
        for _, (l_in, p_in) in irreps_in.simplify()
        for _, (l_sh, p_sh) in irreps_sh
        for l in range(abs(l_in - l_sh), l_in + l_sh + 1)
    ]

    scalars = o3.Irreps([(muls[0], (0, p)) for p in (1, -1) if (0, p) in irreps_available])
    # act_scalars = [swish if p == 1 else torch.tanh for _, (_, p) in scalars]
    act_scalars = [torch.tanh for _, (_, p) in scalars]
    nonscalars = o3.Irreps([(muls[l], (l, p*(-1)**l)) for l in range(1, len(muls)) for p in (1, -1) if (l, p*(-1)**l) in irreps_available])
    if (0, +1) in irreps_available:
        gates = o3.Irreps([(nonscalars.num_irreps, (0, +1))])
        act_gates = [torch.sigmoid]
    else:
        gates = o3.Irreps([(nonscalars.num_irreps, (0, -1))])
        act_gates = [torch.tanh]

    return Gate(scalars, act_scalars, gates, act_gates, nonscalars)


class Conv(torch.nn.Module):
    def __init__(self, irreps_in, irreps_out, irreps_sh, dim_key):
        super().__init__()
        self.irreps_in = irreps_in.simplify()
        self.irreps_out = irreps_out.simplify()
        self.irreps_sh = irreps_sh.simplify()

        # self.si = Linear(self.irreps_in, self.irreps_out, internal_weights=True, shared_weights=True)
        self.si = FullyConnectedTensorProduct(self.irreps_in, o3.Irreps("10x0e"), self.irreps_out)

        # self.lin1 = Linear(self.irreps_in, self.irreps_in, internal_weights=True, shared_weights=True)
        self.lin1 = FullyConnectedTensorProduct(self.irreps_in, o3.Irreps("10x0e"), self.irreps_in)

        instr = []
        irreps = []
        for i_1, (mul_1, (l_1, p_1)) in enumerate(self.irreps_in):
            for i_2, (_, (l_2, p_2)) in enumerate(self.irreps_sh):
                for l_out in range(abs(l_1 - l_2), l_1 + l_2 + 1):
                    p_out = p_1 * p_2
                    if (l_out, p_out) in [(l, p) for _, (l, p) in self.irreps_out]:
                        r = (mul_1, (l_out, p_out))
                        if r in irreps:
                            i_out = irreps.index(r)
                        else:
                            i_out = len(irreps)
                            irreps.append(r)
                        instr += [(i_1, i_2, i_out, 'uvu', True)]
        irreps = o3.Irreps(irreps)
        self.tp = TensorProduct(self.irreps_in, self.irreps_sh, irreps, instr, internal_weights=False, shared_weights=False)

        self.tp_weight = torch.nn.Parameter(torch.randn(dim_key, self.tp.weight_numel))

        # self.lin2 = Linear(irreps, self.irreps_out, internal_weights=True, shared_weights=True)
        self.lin2 = FullyConnectedTensorProduct(irreps, o3.Irreps("10x0e"), self.irreps_out)

    def forward(self, x, z, edge_src, edge_dst, edge_len_emb, edge_attr):
        # with torch.autograd.profiler.record_function("Conv"):
            # x = [num_atoms, dim(irreps_in)]
        s = self.si(x, z)

        x = self.lin1(x, z)

        weight = edge_len_emb @ self.tp_weight

        # edge_attr are divided by sqrt(num_neighbors)
        edge_x = self.tp(x[edge_src], edge_attr, weight)
        x = scatter(edge_x, edge_dst, dim=0, dim_size=len(x))

        x = self.lin2(x, z)

        print_std('self', s)
        print_std('+x', x)
        return s + x / 10

if __name__ == '__main__':
    from project.data.zinc_complex3a6p_data import ZincComplex3a6pData
    from torch_geometric.data import DataLoader
    dataset = ZincComplex3a6pData(data_dir='../data/3a6p/zinc_drug_like_100k/3a6p_pocket5_202020')
    dataloder = DataLoader(dataset, batch_size=512, shuffle=True)
    model = MolecularE3nnQm9()
    for batch in dataloder:
        out = model(batch.x, batch.pos, batch.batch)
        print(out)
        break
    pass
