# -*- coding: UTF-8 -*-
"""
@author:ZNDX
@file:zinc_complex3a6p_data.py
@time:2022/10/13
"""
from torch_geometric.data import InMemoryDataset, Data
import torch
import numpy as np
from scipy import spatial
from torch.utils.data import random_split
from tqdm import tqdm


class ZincComplex3a6pDataY2(InMemoryDataset):

    def __init__(self, data_dir, transform=None, pre_transform=None, pre_filter=None, **kwargs):
        self.elements_dict = {'C': 0, 'H': 1, 'N': 2, 'O': 3, 'F': 4, 'S': 5, 'CL': 6, 'BR': 7, 'I': 8, 'P': 9}
        super().__init__(data_dir, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['raw_data.txt']

    @property
    def processed_file_names(self):
        return ['Ligands_Graph_Data_y2.pt']

    def download(self):
        # Download to `self.raw_dir`
        raise IOError(f'There are No raw data in {self.raw_dir}!')

    def process(self):
        # Read data into huge `Data` list.
        # raw_data = self.raw_dir
        # read file
        with open(self.raw_paths[0], 'r') as f:
            # first line is property_type
            property_types = f.readline().strip().split()
            # split data
            data_original = f.read().strip().split('\n\n')
        # get data number
        # num_examples = len(data_original)
        # data list:[(atom, distance_matrix, label), ...]
        # items = list()
        # make every data graph
        total_ligands_graph = list()
        for data in tqdm(data_original):
            # get every row
            data = data.strip().split('\n')
            # get data id
            id = data[0]
            # get property in last row
            property = torch.tensor(float(data[-1].strip()), dtype=torch.float32)
            # get atoms and its coordinate
            atoms, atom_coords = [], []
            for atom_xyz in data[1:-1]:
                atom, *xyz = atom_xyz.split()
                atoms.append(atom)
                xyz = list(map(float, xyz))
                atom_coords.append(xyz)
                # print(xyz)
            # transform symbols to numbers, such as:{'C':0, 'H':1, ...}
            atoms = np.array([self.elements_dict[a] for a in atoms])
            # create distance matrix
            distance_matrix = spatial.distance_matrix(atom_coords, atom_coords)
            distance_matrix = np.where(distance_matrix == 0.0, 1e6, distance_matrix)
            # 构建全连接图的edge_index
            row, col = [], []
            # 总共有len(atoms)个节点
            for i in range(len(atoms)):
                row.extend([i] * len(atoms))
                col.extend(range(len(atoms)))
            edge_index = torch.tensor((col, row), dtype=torch.long)
            edge_attr = torch.tensor(distance_matrix.reshape(-1, 1), dtype=torch.float32)
            d = Data(x=torch.tensor(atoms, dtype=torch.long), edge_index=edge_index, edge_attr=edge_attr,
                     y=torch.pow(property, 2),
                     pos=torch.tensor(atom_coords), id=id)
            # return d
            total_ligands_graph.append(d)

        if self.pre_filter is not None:
            total_ligands_graph = [data for data in total_ligands_graph if self.pre_filter(data)]

        if self.pre_transform is not None:
            total_ligands_graph = [self.pre_transform(data) for data in total_ligands_graph]
        data, slices = self.collate(total_ligands_graph)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == '__main__':
    data = ZincComplex3a6pData('3a6p/zinc_drug_like_100k/3a6p_pocket5_202020')
    pass
