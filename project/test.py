# -*- coding: UTF-8 -*-
"""
@author:ZNDX
@file:test.py
@time:2022/10/15
"""
from data.zinc_complex3a6p_data import ZincComplex3a6pData
from torch_geometric.loader import DataLoader
from model.molecular_gnn import MolecularGnn
if __name__ == '__main__':
    ligand_dataset = ZincComplex3a6pData(
        '/mnt/e/Python_Project/SMTarRNA/project/data/3a6p/zinc_drug_like_100k/exhaus_96')
    dataloder = DataLoader(ligand_dataset, batch_size=32, shuffle=True)
    for batch_data in dataloder:
        # results = model(batch_data.x, batch_data.edge_index, batch_data.edge_weight)
        break
    model = MolecularGnn(10, hidden_channels=256, hidden_layers=16, out_layers=8, out_channels=1)
    model(batch_data)
