# 设置工作目录
import os
from tqdm import tqdm
import sys
import torch
import numpy as np
import requests
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
os.chdir("/mnt/f/SMTarRNA_total_results/total_dataset_results/total_dataset_results")

# hpc
# os.chdir("/public/home/hpc192311018/Huabei/project/SMTARRNA-sync/SMTarRNA/data/total_dataset_results")

# 设置python工作目录

sys.path.append("/home/huabei/projects/SMTarRNA")
# os.listdir()

# 将data_(n, 5)的第一列和data_(1, )的第一列拼接起来
def get_data(folder, file_lists):
    data = []
    for file in tqdm(file_lists):
        data_ = torch.load(os.path.join(folder, file))
        data_ = np.concatenate([data_[1].reshape(-1, 1), data_[0][:, 0].reshape(-1, 1)], axis=1)
        data.append(data_)
    data = np.concatenate(data, axis=0)
    return data
def create_zinc_id(id: int):
    return 'ZINC'+str(int(id + 1e12))[1:]

# 依据total_data第二列获取top千分之一的数据
def get_top_data(total_data, top=0.001):
    top_data = total_data[total_data[:, 1].argsort()][:int(total_data.shape[0]*top)]
    return top_data


def download_smiles(url: str):
    smiles = requests.get(url)
    smiles = str(smiles.content, encoding='utf-8')
    return smiles
def create_smiles_url(zinc_id: str):
    return 'https://zinc20.docking.org/substances/'+zinc_id+'.smi'

# 统计每个文件夹下的结果
folders = ['3a6p', '4z4c', '4z4d', '6cbd']
# folder = '3a6p'
# folder = '4z4d'

for folder in folders:
    logging.info(f'processing {folder}')
    file_lists = os.listdir(folder)[:10]
    total_data = get_data(folder, file_lists)
    top_data = get_top_data(total_data)
    logging.info('saving data')
    # torch.save(top_data, folder+'_top_data.pt')
    np.savetxt(folder+'_top_data.csv', top_data, delimiter=',')

    # 将smiles写入单个文件
    # with open(folder+'_top_data.smi', 'w') as f:
    #     for i in tqdm(range(top_data.shape[0])):
    #         zinc_id = create_zinc_id(top_data[i, 0])
    #         smiles_url = create_smiles_url(zinc_id)
    #         smiles = download_smiles(smiles_url)
    #         f.write(smiles)


