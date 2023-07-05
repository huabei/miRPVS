# 此文件用于定义数据集的基类

import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset


class ZincComplexBase(InMemoryDataset):
    """此类用于定义数据集的基类，包含了一些共有的方法，如数据集的读取、处理等."""

    def __init__(self, data_dir, train=True, transform=None, pre_transform=None, pre_filter=None):
        self.elements_dict = dict(C=0, N=1, O=2, H=3, F=4, S=5, CL=6, BR=7, I=8, SI=9, P=10)
        super().__init__(data_dir, transform, pre_transform, pre_filter)
        if train:
            # 载入训练集
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            # 载入测试集
            self.data, self.slices = torch.load(self.processed_paths[1])

    def download(self):
        # Download to `self.raw_dir`
        raise OSError(f"There are No raw data in {self.raw_dir}!")

    def load_data(self):
        """载入h5文件中的数据，返回coor和label两个DataFrame，分别存储了原子坐标和标签信息."""
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
        return coor, label

    def index_data(self, coor, zinc_id, r):
        """依据r中的起始终止点从coor当中索引相关数据."""
        # 获取id和起始点与终止点
        id = int(zinc_id[4:])
        r: pd.Series

        # 根据起始点和终止点获取pos
        start = int(r["start"])
        end = int(r["end"])
        # 跳过含有none的数据
        if coor.iloc[start:end]["atom_id"].isnull().any():
            print(f"skip {zinc_id}")
            return None
        pos = coor.iloc[start:end][["x", "y", "z"]]
        x = coor.iloc[start:end]["atom_id"]

        # label
        y = r[["total", "inter", "intra", "torsions", "intra best pose"]]
        return id, pos, x, y

    def save_data(self, ligands_graph: list, file_path: str):
        """转换并保存数据."""

        if self.pre_filter is not None:
            ligands_graph = [data for data in ligands_graph if self.pre_filter(data)]

        if self.pre_transform is not None:
            ligands_graph = [self.pre_transform(data) for data in ligands_graph]
        data, slices = self.collate(ligands_graph)
        torch.save((data, slices), file_path)
