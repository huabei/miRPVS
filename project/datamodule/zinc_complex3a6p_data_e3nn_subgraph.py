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
from torch_cluster import knn_graph
from torch_geometric.utils.subgraph import k_hop_subgraph
from tqdm import tqdm


class ZincComplex3a6pDataE3nnSubgraph(InMemoryDataset):

    def __init__(self, data_dir, transform=None, pre_transform=None, pre_filter=None, **kwargs):
        self.elements_dict = {'C': 0, 'P': 1, 'N': 2, 'O': 3, 'F': 4, 'S': 5, 'CL': 6, 'BR': 7, 'I': 8}
        super().__init__(data_dir, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['raw_data.txt']

    @property
    def processed_file_names(self):
        return ['Ligands_Graph_Data_e3nn_subgraph_constant_unknown.pt']

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
            # get atoms and its coordinate
            atoms, atom_coords = [], []
            for atom_xyz in data[1:-1]:
                atom, *xyz = atom_xyz.split()
                # 去除H原子
                atom: str
                if atom.strip().upper() == 'H':
                    continue
                atoms.append(atom)
                xyz = list(map(float, xyz))
                atom_coords.append(xyz)
                # print(xyz)
            # transform symbols to numbers, such as:{'C':0, 'P':1, ...}
            atoms = np.array([self.elements_dict[a] for a in atoms])
            # one_hot = np.zeros((atoms.size, len(self.elements_dict)))
            # one_hot[np.arange(atoms.size), atoms] = 1

            pos = torch.tensor(atom_coords, dtype=torch.float32)
            # create edges
            # 因为P能连5个原子，所以这里设置为5
            edge_index = knn_graph(pos, k=6, loop=True)
            edge_attr = pos[edge_index[0]] - pos[edge_index[1]]
            # create subgraph
            # 遍历节点，找到每个节点的子图
            num_nodes = len(atoms)
            for c_node in range(num_nodes):
                # 获取每个原子的子图
                # subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(c_node, num_hops=1,
                #                                                             edge_index=edge_index,
                #                                                             relabel_nodes=True,
                #                                                             num_nodes=len(atoms))
                # 获取每个原子的子图
                label = torch.tensor(atoms[c_node], dtype=torch.float32)
                edge_mask = edge_index[1] == c_node
                sub_edge_index_ = edge_index.T[edge_mask].T
                subset = sub_edge_index_[0]
                # 边的特征
                edge_attr_ = edge_attr[edge_mask]
                # 重新标记节点
                node_idx = label.new_full((num_nodes, ), -1)
                node_idx[subset] = torch.arange(subset.size(0))
                sub_edge_index_ = node_idx[sub_edge_index_]
                # 节点特征
                x = torch.tensor(atoms[:][subset], dtype=torch.float32)
                # 随机将中心原子设为一个新值
                # x[sub_edge_index_[1][0]] = torch.randint(0, len(self.elements_dict), (1,))
                x[sub_edge_index_[1][0]] = torch.tensor([len(self.elements_dict)], dtype=torch.long)
                x_one_hot = torch.zeros((x.size(0), len(self.elements_dict)+1), dtype=torch.long)
                x_one_hot[torch.arange(len(subset)), x] = torch.tensor([1], dtype=torch.long)
                x_one_hot = x_one_hot.to(torch.float32)
                # edge_attr_ = edge_attr[edge_mask]
                total_ligands_graph.append(Data(x=x_one_hot, edge_index=sub_edge_index_, edge_attr=edge_attr_, y=label))

        if self.pre_filter is not None:
            total_ligands_graph = [data for data in total_ligands_graph if self.pre_filter(data)]

        if self.pre_transform is not None:
            total_ligands_graph = [self.pre_transform(data) for data in total_ligands_graph]
        data_, slices = self.collate(total_ligands_graph)
        torch.save((data_, slices), self.processed_paths[0])


if __name__ == '__main__':
    data = ZincComplex3a6pDataE3nnSubgraph('3a6p/zinc_drug_like_100k/3a6p_pocket5_202020')
    pass
