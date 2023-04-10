"""
analyze_zinc_pdbqt_gz: 此函数用于分析ZINC的pdbqt.gz文件，将其转换成3维坐标数据和原子在3维坐标数据中的起始和终止位置
create_total_dataset_hdf5: 将ZINC所有的分子3维数据，合并成一个hdf5文件
"""

import pandas as pd
import gzip
from pandas import HDFStore
import numpy as np
import os
from tqdm import tqdm
from typing import Union
import logging
import time


def analyze_zinc_pdbqt_gz(pdbqt_gz_path: str) -> Union[pd.DataFrame, pd.DataFrame]:
    """此函数用于分析ZINC的pdbqt.gz文件，将其转换成3维坐标数据和原子在3维坐标数据中的起始和终止位置
    input: pdbqt_gz_path: str, pdbqt.gz文件的路径
    output: coor: pd.DataFrame, 3维坐标数据
            index: pd.DataFrame, 每个分子中的原子在coor中的起始和终止位置
    """
    coor = []
    index = []
    # 读取pdbqt.gz文件
    with gzip.open(pdbqt_gz_path, 'rb') as f:
        t_start = 0 # 记录当前分子的原子起始位置
        t_end = 0 # 记录当前分子的原子终止位置
        for line in f:
            if line.startswith(b'ATOM'):
                coor.append([str(line[12:14].strip(), 'utf-8'), float(line[30:38]), float(line[38:46]), float(line[46:54])])
                t_end += 1 # 记录已存入原子的个数
            if line.startswith(b'REMARK  Name = '): # 一个分子的起始位置
                if t_end == 0:
                    # 记录第一个分子的id
                    zinc_id = str(line[15:].strip(), 'utf-8')
                    continue
                index.append([zinc_id, t_start, t_end])  # 存储上一个分子的信息
                zinc_id = str(line[15:].strip(), 'utf-8')  # 记录当前分子的id
                t_start = t_end # 记录当前分子的原子起始位置
        index.append([zinc_id, t_start, t_end])
    return pd.DataFrame(coor, columns=['atom', 'x', 'y', 'z']), pd.DataFrame(index, columns=['zinc_id', 'start', 'end']).set_index('zinc_id', drop=True)


def create_total_dataset_hdf5(data_folder: str) -> None:
    """将ZINC所有的分子3维数据，合并成一个hdf5文件
    input: data_folder: str, 存储ZINC分子3维数据的文件夹
           output_path: str, 输出的hdf5文件路径
    """
    if data_folder.endswith('/'):
        data_folder = data_folder[:-1]
    output_path_coor = data_folder + '_coor.h5'
    output_path_index = data_folder + '_index.h5'
    with HDFStore(output_path_coor) as store_coor:
        with HDFStore(output_path_index) as store_index:
            for path, sub_dir, filename in tqdm(os.walk(data_folder)):
                for file in filename:
                    if file.endswith('.pdbqt.gz'):
                        # 构造hdf5的key，去掉ZINC_DATA_PATH的前缀，去掉后缀
                        d = os.path.join(path.replace(data_folder, os.path.basename(data_folder)), '_'.join(file.split('.')[:2])).replace('-','_')
                        # 分析pdbqt.gz文件,加入try except是因为有些文件可能有问题
                        try:
                            logging.info(f'processing {os.path.join(path, file)}')
                            store_coor[d], store_index[d] = analyze_zinc_pdbqt_gz(os.path.join(path, file))
                        except:
                            logging.warning(f'error in {os.path.join(path, file)}')
                            continue
