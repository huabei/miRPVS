"""
@author:ZNDX
@file:zinc_complex3a6p_data.py
@time:2022/10/13
"""

import logging

import torch
from torch_geometric.data import Data
from tqdm import tqdm

from .zinc_complex_base import ZincComplexBase


class ZincComplex3a6pDataSingleLabelTest(ZincComplexBase):
    def __init__(self, data_dir, train=True, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(data_dir, train, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ["3a6p_1m.h5"]

    @property
    def processed_file_names(self):
        return ["Ligands_Graph_Data_Single_Label_Test.pt"]

    def download(self):
        # Download to `self.raw_dir`
        raise OSError(f"There are No raw data in {self.raw_dir}!")

    def process(self):
        # Read data into huge `Data` list.
        # raw_data = self.raw_dir
        # read file
        coor, label = self.load_data()
        # 利用label分割图
        total_ligands_graph = []
        t = 0
        for zinc_id, r in tqdm(label.iterrows()):
            # 索引相关数据
            if self.index_data(coor, zinc_id, r) is not None:
                id, pos, x, y = self.index_data(coor, zinc_id, r)
            else:
                logging.warning(f"skip {zinc_id}")
                continue
            y = y[["total"]]
            # 构建全连接图的edge_index
            edge_index = [[], []]
            for i in range(len(pos)):
                edge_index[0].extend([i] * len(pos))
                edge_index[1].extend(list(range(len(pos))))
            edge_index = torch.tensor(edge_index, dtype=torch.long)

            d = Data(
                x=torch.tensor(x.values, dtype=torch.long),
                edge_index=edge_index,
                y=torch.tensor(y.values.reshape(1, 1), dtype=torch.float),
                pos=torch.tensor(pos.values, dtype=torch.float),
                id=torch.tensor(id, dtype=torch.long),
            )
            # return d
            total_ligands_graph.append(d)
            t += 1
            if t > 10000:
                break
        self.save_data(total_ligands_graph, self.processed_paths[0])
