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
# from torch.utils.data import random_split
from torch_cluster import radius_graph


class ZincComplex3a6pDataE3nn(InMemoryDataset):

    def __init__(self, data_dir, transform=None, pre_transform=None, pre_filter=None, **kwargs):
        self.elements_dict = {'C': 0, 'H': 1, 'N': 2, 'O': 3, 'F': 4, 'S': 5, 'CL': 6, 'BR': 7, 'I': 8, 'P': 9}
        super().__init__(data_dir, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['raw_data.txt']

    @property
    def processed_file_names(self):
        return ['Ligands_Graph_Data_e3nn.pt']

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
        for data in data_original:
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
            pos = torch.tensor(atom_coords, dtype=torch.float32)
            # create edges
            max_radius = 2.0
            edge_index = radius_graph(pos, r=max_radius, loop=True, max_num_neighbors=pos.shape[0]-1)
            # create edge_vec
            edge_vec = pos[edge_index[0]] - pos[edge_index[1]]

            d = Data(x=torch.tensor(atoms, dtype=torch.long), edge_index=edge_index, edge_attr=edge_vec,
                     y=property,
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
    pass