# -*- coding: UTF-8 -*-
"""
@author:ZNDX
@file:test.py
@time:2022/10/15
"""
from data.zinc_complex3a6p_data_e3nn import ZincComplex3a6pDataE3nn
from torch_geometric.loader import DataLoader
from model.molecular_e3nn_egcn import MolecularE3nnEgcn
from data.data_interface import DInterface
from torch_scatter import scatter_sum
import pytorch_lightning as pl
from tqdm import tqdm
from model.molecular_graph_conv import MolecularGraphConv

if __name__ == '__main__':
    pl.seed_everything(1234)
    # dataset = ZincComplex3a6pDataE3nn(
    #     data_dir='data/3a6p/zinc_drug_like_100k/3a6p_pocket5_202020')
    data_dir = 'data/3a6p/zinc_drug_like_100k/3a6p_pocket5_202020'
    dataset = 'zinc_complex3a6p_data_graph_conv'
    d = DInterface(dataset=dataset, data_dir=data_dir, batch_size=128)
    d.setup(stage='fit')
    model = MolecularGraphConv(10, 128, 1, 2, 6)
    # dataloader = DataLoader(d, batch_size=128, shuffle=True)
    for epoch in range(100):
        print(epoch)
        for batch_data in tqdm(d.train_dataloader()):
            results = model(batch_data)
            # x = scatter_sum(batch_data.x[batch_data.edge_index[0]], batch_data.edge_index[1], dim=0)
            # scatter_sum(x, batch_data.batch, dim=0)
            pass
        # break
    # model = MolecularE3nnEgcn(10, hidden_channels=128, hidden_layers=3, out_layers=8, out_channels=1)
    # model(batch_data)
