"""
@author:ZNDX
@file:zinc_complex3a6p_data.py
@time:2022/10/13
"""

import logging
import random

import torch
from scipy import spatial
from torch_geometric.data import Data
from tqdm import tqdm

from .zinc_complex_base import ZincComplexBase


class ZincComplexDataMoleculeGnn(ZincComplexBase):
    def __init__(self, data_dir, cmpx='3a6p', train=True, transform=None, pre_transform=None, pre_filter=None):
        self._cmpx = cmpx
        super().__init__(data_dir, train, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        """返回原始数据文件名."""
        return [f"{self._cmpx}_1m.h5"]

    @property
    def processed_file_names(self):
        """返回处理后的数据文件名，应该具有意义，方便辨识."""
        return [
            "Ligands_Graph_Data_Multi_Label_molecule_gnn.pt",
            "Ligands_Graph_Data_Multi_Label_Test_100k_molecule_gnn.pt",
        ]

    def process(self):
        # Read data into huge `Data` list.
        # raw_data = self.raw_dir
        # read file
        coor, label = self.load_data()

        # 使用label中的信息构建图
        total_ligands_graph = []
        for zinc_id, r in tqdm(label.iterrows()):
            # 索引相关数据
            if self.index_data(coor, zinc_id, r) is not None:
                id, pos, x, y = self.index_data(coor, zinc_id, r)
            else:
                logging.warning(f"skip {zinc_id}")
                continue
            # 构建全连接图的edge_index
            edge_index = [[], []]
            for i in range(len(pos)):
                edge_index[0].extend([i] * len(pos))
                edge_index[1].extend(list(range(len(pos))))
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            # 计算distance matrix
            distance_matrix = spatial.distance_matrix(pos, pos)
            # 计算edge_attr
            edge_attr = torch.tensor(distance_matrix, dtype=torch.float32).view(-1, 1)
            d = Data(
                x=torch.tensor(x.values, dtype=torch.long),
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor(y.values.reshape(1, 5), dtype=torch.float),
                id=torch.tensor(id, dtype=torch.long),
            )
            total_ligands_graph.append(d)
        # 随机打乱数据
        random.shuffle(total_ligands_graph)
        # 保存训练集数据
        self.save_data(total_ligands_graph[:-100000], self.processed_paths[0])
        # 保存测试集数据
        self.save_data(total_ligands_graph[-100000:], self.processed_paths[1])
