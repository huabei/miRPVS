# 设置工作目录
import logging
import os
import sys

import numpy as np
import ray
import requests
import torch
from tqdm import tqdm

ray.init(num_cpus=7)
import time

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
os.chdir("/mnt/f/SMTarRNA_total_results/total_dataset_results/total_dataset_results")

# hpc
# os.chdir("/public/home/hpc192311018/Huabei/project/SMTARRNA-sync/SMTarRNA/data/total_dataset_results")

# 设置python工作目录


# sys.path.append("/home/huabei/projects/SMTarRNA")
# os.listdir()
def create_zinc_id(id: int):
    return "ZINC" + str(int(id + 1e12))[1:]


@ray.remote(num_cpus=0.2)
def download_smiles(url: str):
    smiles = requests.get(url)
    smiles = str(smiles.content, encoding="utf-8")
    return smiles


def create_smiles_url(zinc_id: str):
    return "https://zinc20.docking.org/substances/" + zinc_id + ".smi"


# 统计每个文件夹下的结果
folders = ["3a6p", "4z4c", "4z4d", "6cbd"]
# folder = '3a6p'
# folder = '4z4d'
if not os.path.exists("total_zinc_id_set.pt"):
    data_list = []
    for folder in folders:
        logging.info(f"processing {folder}")
        # 把第一列的zinc_id取出来
        data_list.append(set(torch.load(folder + "_top_data.pt")[:, 0]))
    # 进行并集操作
    total_zinc_id = set()
    for data in data_list:
        total_zinc_id = total_zinc_id | data
    # 保存并集
    torch.save(total_zinc_id, "total_zinc_id_set.pt")
else:
    total_zinc_id = torch.load("total_zinc_id_set.pt")


# 将smiles写入单个文件
with open("total_smiles.smi", "w") as f:
    smiles_list = []
    refer_list = []
    for i in tqdm(list(total_zinc_id)[100:200]):
        zinc_id = create_zinc_id(i)
        smiles_url = create_smiles_url(zinc_id)
        if len(refer_list) >= 20:
            smiles_refers, refer_list = ray.wait(refer_list, num_returns=1)
            # smiles_list.append(ray.get(smiles_refers[0]))
            f.write(ray.get(smiles_refers[0]))
        time.sleep(0.05)
        smiles_refer = download_smiles.remote(smiles_url)
        # print(smiles.split(' ')[0])
        refer_list.append(smiles_refer)
        # break
    for smiles in ray.get(refer_list):
        # smiles_list.append(ray.get(smiles))
        f.write(smiles)
# with open('total_smiles_100000_200000.smi', 'w') as f:
#     f.writelines(smiles_list)
