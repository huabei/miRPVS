"""
@author:ZNDX
@file:zinc_complex3a6p_data.py
@time:2022/10/13
此文件用来作为数据处理的封装，对接dataloader
"""
import logging
import os
import os.path as osp
import shutil
import sys
import time

import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

TIME = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
# 保存日志到文件
logging.basicConfig(
    level=logging.INFO,
    filename="zinc_infer_complete.log",
    filemode="a",
    format="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s",
)


class ZincInferDataset(InMemoryDataset):
    """前向推理使用的数据集."""

    def __init__(self, data_dir, dataset, transform=None, pre_transform=None, pre_filter=None):
        # 为了方便分配任务，这里的data_dir是一个文件夹，里面存放了多个h5文件，使用data来指定使用哪个h5文件
        super().__init__(
            None, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter
        )
        self.elements_dict = dict(C=0, N=1, O=2, H=3, F=4, S=5, CL=6, BR=7, I=8, SI=9, P=10)
        ele_df = pd.DataFrame.from_dict(self.elements_dict, orient="index", columns=["element_id"])
        self.data_dir = data_dir
        # index文件数据用以分割coor文件数据
        self.index_data_path = osp.join(self.data_dir, f"{dataset}_index.h5")
        # coor文件数据用以构建图
        self.coor_data_path = osp.join(self.data_dir, f"{dataset}_coor.h5")

        if not osp.exists(self.index_data_path):
            raise FileNotFoundError(f"{self.index_data_path} not found")
        if not osp.exists(self.coor_data_path):
            raise FileNotFoundError(f"{self.coor_data_path} not found")

        logging.info(f"start processing {dataset}")
        # 读取文件数据,coor数据经过了处理，atom列已经转换为了数字，但是保留了Nan
        coor_hf, label_hf = self.load_data()
        ns = 0
        # 如果已经存在了，就不用再处理了,继续检查之后的文件的是否存在,从第ns个开始
        while osp.exists(f"./outputs/{dataset}{ns}_infer.pt.zip"):
            ns += 1
        # 使用label中的信息构建图
        self.total_ligands_graph = []
        self.total_ligands_graph_num = 0
        n = 0
        for k in tqdm(label_hf.keys(), desc=dataset):
            label = label_hf[k]
            coor = coor_hf[k]
            coor: pd.DataFrame
            label: pd.DataFrame
            # 将atom转换为数字
            coor["atom_id"] = coor["atom"].map(ele_df["element_id"])
            # print(coor[coor["atom_id"].isnull()])
            # print(label.head())
            # print(coor.head())
            assert (
                len(coor) == label["end"].max()
            ), f"coor length is {len(coor)}, label length is {label['end'].max()}"
            if n < ns:  # 跳过已经处理过的文件
                self.total_ligands_graph_num += self.cont_graph(label, coor)
                if self.total_ligands_graph_num > 1_000_000:
                    logging.info(f"skip {dataset}{n}")
                    self.total_ligands_graph_num = 0
                    n += 1
                continue
            self.total_ligands_graph.extend(self.create_graph(label, coor))
            # 超过100万个图就保存一次,防止内存溢出与速度变慢
            if len(self.total_ligands_graph) > 1_000_000:
                self.save_graph(dataset + str(n))
                del self.total_ligands_graph
                self.total_ligands_graph = []
                n += 1
        coor_hf.close()
        label_hf.close()
        # 保存最后一批数据
        self.save_graph(dataset + str(n))
        logging.info(f"finish processing {dataset}")

    def save_graph(self, dataset_name):
        """保存图数据."""
        logging.info(f"save {dataset_name}_infer.pt")
        data, slices = self.collate(self.total_ligands_graph)
        torch.save((data, slices), f"./outputs/{dataset_name}_infer.pt")
        logging.info(f"zip {dataset_name}_infer.pt to {dataset_name}_infer.pt.zip")
        # 压缩文件
        os.system(f"zip ./outputs/{dataset_name}_infer.pt.zip ./outputs/{dataset_name}_infer.pt")
        # shutil.make_archive(f'./outputs/{dataset_name}_infer.pt', 'zip', './outputs', f'{dataset_name}_infer.pt')
        # 删除pt数据
        logging.info(f"remove {dataset_name}_infer.pt")
        os.remove(f"./outputs/{dataset_name}_infer.pt")

    def cont_graph(self, label, coor):
        """计算有效分子图的数量."""
        total_ligands_graph_num = 0
        for zinc_id, r in tqdm(label.iterrows()):
            # 索引相关数据
            if self.index_data(coor, zinc_id, r) is not None:
                total_ligands_graph_num += 1
            else:
                logging.warning(f"skip {zinc_id}")
                continue
        return total_ligands_graph_num

    def create_graph(self, label, coor):
        total_ligands_graph = []
        for zinc_id, r in tqdm(label.iterrows()):
            # 索引相关数据
            if self.index_data(coor, zinc_id, r) is not None:
                id, pos, x = self.index_data(coor, zinc_id, r)
            else:
                logging.warning(f"skip {zinc_id}")
                continue
            # 构建全连接图的edge_index
            edge_index = [[], []]
            for i in range(len(pos)):
                edge_index[0].extend([i] * len(pos))
                edge_index[1].extend(list(range(len(pos))))
            edge_index = torch.tensor(edge_index, dtype=torch.long)

            d = Data(
                x=torch.tensor(x.values, dtype=torch.long),
                edge_index=edge_index,
                pos=torch.tensor(pos.values, dtype=torch.float),
                id=torch.tensor(id, dtype=torch.long),
            )
            total_ligands_graph.append(d)
        return total_ligands_graph

    def load_data(self):
        """载入h5文件中的数据，返回coor和label两个DataFrame，分别存储了原子坐标和标签信息."""

        coor = pd.HDFStore(self.coor_data_path, mode="r")
        label = pd.HDFStore(self.index_data_path)
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
        return id, pos, x


if __name__ == "__main__":
    # 生成数据
    # python zinc_infer.py dir/ 3a6p/
    _, data_dir, dataset = sys.argv
    if dataset.endswith("/"):
        dataset = dataset[:-1]
    print(data_dir, dataset)
    if osp.exists(f"./outputs/{dataset}_infer.pt.zip"):
        logging.info(f"{dataset} already exists")
        exit(0)
    ZincInferDataset(data_dir=data_dir, dataset=dataset)
    del ZincInferDataset
