# 设置工作目录
import logging
import os
import sys
import time
from concurrent import futures
from functools import partial

import numpy as np
import requests
import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="get_smiles.log",
)
# os.chdir("/mnt/f/SMTarRNA_total_results/total_dataset_results/total_dataset_results")
MAX_WORKERS = 20
# hpc
# os.chdir("/public/home/hpc192311018/Huabei/project/SMTARRNA-sync/SMTarRNA/data/total_dataset_results")
_, start, end = sys.argv
# 设置python工作目录


# sys.path.append("/home/huabei/projects/SMTarRNA")
# os.listdir()
def create_zinc_id(id: int):
    """将id转换为zinc_id."""
    return "ZINC" + str(int(id + 1e12))[1:]


def download_smiles(url: str):
    """下载smiles."""
    smiles = requests.get(url, timeout=10)
    smiles = str(smiles.content, encoding="utf-8")
    return smiles


def create_smiles_url(zinc_id: str):
    """创建smiles的url."""
    return "https://zinc20.docking.org/substances/" + zinc_id + ".smi"


# 统计每个文件夹下的结果
folders = ["3a6p", "4z4c", "4z4d", "6cbd"]
# folder = '3a6p'
# folder = '4z4d'
if not os.path.exists("total_zinc_id_set.pt"):
    data_list = []
    for folder in folders:
        logging.info(f"processing {folder}")
        data_list.append(set(torch.load(folder + "_top_data.pt")[:, 0]))
    # 进行并集操作
    total_zinc_id = set()
    for data in data_list:
        total_zinc_id = total_zinc_id | data
    # 保存并集
    torch.save(total_zinc_id, "total_zinc_id_set.pt")
else:
    total_zinc_id = torch.load("total_zinc_id_set.pt")


def download_one(id, f):
    """下载一个smiles."""
    zinc_id = create_zinc_id(id)
    smiles_url = create_smiles_url(zinc_id)
    try:
        smiles = download_smiles(smiles_url)
    except Exception as e:
        logging.info(f"error {zinc_id}")
        return id
    f.write(smiles)
    return id


def download_many(id_list, f):
    """多线程下载smiles."""
    _download_one = partial(download_one, f=f)
    workers = min(MAX_WORKERS, len(id_list))
    with futures.ThreadPoolExecutor(max_workers=workers) as executor:
        executor.map(_download_one, id_list)


time_start = time.time()

start = int(start)
end = len(total_zinc_id) if end == "-1" else int(end)

# start = 0
# end = 500_000
# end = 100
# start = 500_000
# end = len(total_zinc_id)
# 10万个smiles保存一个文件
logging.info(f"start:{start}, end: {end}")
for i in range(start, end, 10_000):
    # print(i, i+100_000)
    if i + 10_000 > end:
        e = end
    else:
        e = i + 10_000
    if os.path.exists(f"total_smiles_{i}_{e}.smi"):
        logging.info(f"file total_smiles_{i}_{e}.smi exists")
        continue
    logging.info(f"processing file total_smiles_{i}_{e}.smi")
    f = open(f"total_smiles_{i}_{e}.smi", "w")
    download_many(list(total_zinc_id)[i:e], f)
    logging.info(f"file total_smiles_{i}_{e}.smi finished")
    f.close()
# f.close()
time_end = time.time()

logging.info(f"time cost: {time_end-time_start} s")

# # 将smiles写入单个文件
# with open('total_smiles.smi', 'w') as f:
#     smiles_list = []
#     refer_list = []
#     for i in tqdm(list(total_zinc_id)[100:200]):
#         zinc_id = create_zinc_id(i)
#         smiles_url = create_smiles_url(zinc_id)
#         if len(refer_list) >= 20:
#             smiles_refers, refer_list = ray.wait(refer_list, num_returns=1)
#             # smiles_list.append(ray.get(smiles_refers[0]))
#             f.write(ray.get(smiles_refers[0]))
#         time.sleep(0.05)
#         smiles_refer = download_smiles.remote(smiles_url)
#         # print(smiles.split(' ')[0])
#         refer_list.append(smiles_refer)
#         # break
#     for smiles in ray.get(refer_list):
#         # smiles_list.append(ray.get(smiles))
#         f.write(smiles)
# with open('total_smiles_100000_200000.smi', 'w') as f:
#     f.writelines(smiles_list)
