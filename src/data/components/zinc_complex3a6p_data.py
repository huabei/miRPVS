"""
@author:ZNDX
@file:zinc_complex3a6p_data.py
@time:2022/10/13
"""
import numpy as np
import pandas as pd
import torch
from scipy import spatial
from torch.utils.data import random_split
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm


class ZincComplex3a6pData(InMemoryDataset):
    def __init__(self, data_dir, transform=None, pre_transform=None, pre_filter=None):
        self.elements_dict = dict(C=0, N=1, O=2, H=3, F=4, S=5, CL=6, BR=7, I=8, SI=9, P=10)
        super().__init__(data_dir, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["raw_data.h5"]

    @property
    def processed_file_names(self):
        return ["Ligands_Graph_Data.pt"]

    def download(self):
        # Download to `self.raw_dir`
        raise OSError(f"There are No raw data in {self.raw_dir}!")

    def process(self):
        # Read data into huge `Data` list.
        # raw_data = self.raw_dir
        # read file
        ele_df = pd.DataFrame.from_dict(self.elements_dict, orient="index", columns=["element_id"])

        with pd.HDFStore(self.raw_paths[0], "r") as store:
            coor = store["pos"]
            label = store["label"]
            # 将atom转换为数字
            coor["atom_id"] = coor["atom"].map(ele_df["element_id"])
            print(coor[coor["atom_id"].isnull()])

            assert (
                len(coor) == label["end"].max()
            ), f"coor length is {len(coor)}, label length is {label['end'].max()}"
            coor: pd.DataFrame
            label: pd.DataFrame
        # 利用label分割图
        total_ligands_graph = []
        # t = 0
        for zinc_id, r in tqdm(label.iterrows()):
            id = int(zinc_id[4:])
            # if id in [562412253, 584535530, 342391465]:  # 有问题的数据
            #     print(zinc_id)
            #     continue
            r: pd.Series
            # 获取pos
            # print(r['start'], r['end'], type(r['start']))
            # raise Exception
            start = int(r["start"])
            end = int(r["end"])
            # 跳过含有none的数据
            if coor.iloc[start:end]["atom_id"].isnull().any():
                print(f"skip {zinc_id}")
                continue
            pos = coor.iloc[start:end][["x", "y", "z"]]
            x = coor.iloc[start:end]["atom_id"]
            y = r[["total", "inter", "intra", "torsions", "intra best pose"]]
            # 构建全连接图的edge_index
            edge_index = [[], []]
            for i in range(len(pos)):
                edge_index[0].extend([i] * len(pos))
                edge_index[1].extend(list(range(len(pos))))
            edge_index = torch.tensor(edge_index, dtype=torch.long)

            d = Data(
                x=torch.tensor(x.values, dtype=torch.long),
                edge_index=edge_index,
                y=torch.tensor(y.values.reshape(1, 5), dtype=torch.float),
                pos=torch.tensor(pos.values, dtype=torch.float),
                id=torch.tensor(id, dtype=torch.long),
            )
            # return d
            total_ligands_graph.append(d)
            # t += 1
            # if t > 10000:
            #     break

        if self.pre_filter is not None:
            total_ligands_graph = [data for data in total_ligands_graph if self.pre_filter(data)]

        if self.pre_transform is not None:
            total_ligands_graph = [self.pre_transform(data) for data in total_ligands_graph]
        data, slices = self.collate(total_ligands_graph)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    data = ZincComplex3a6pData("data/dataset/dataset/3a6p_100w")
    pass
