# -*- coding: UTF-8 -*-
"""
@author:ZNDX
@file:test.py
@time:2022/10/30
"""
import os
import pandas as pd
from collections import defaultdict
import numpy as np
import copy
from tqdm import tqdm
from utils import ligand_pocket_position_statistics, ZincPdbqt, generate_coor, get_pocket_info, map_and_conjunction, zinc_pdbqt_transform_decorator
from functools import partial


@zinc_pdbqt_transform_decorator
def transform(pdbqt_model):
    return generate_coor(pdbqt_model)


@zinc_pdbqt_transform_decorator
def transform2(atom_list, pocket_alpha: list):
    return ligand_pocket_position_statistics(pocket_alpha, atom_list)


def main(dock_out_folder, fpocket_out_folder, pocket_index):
    # 提取pocket的信息
    pocket_dict = get_pocket_info(fpocket_out_folder)
    # 提取对接输出目录
    dock_conformation_sm = [os.path.join(dock_out_folder, file_name) for file_name in os.listdir(dock_out_folder) if file_name.endswith('.gz')]

    # 以每个分子中所有原子与最近的alpha球距离的平均值作为分子与口袋的距离
    statis_results = [ZincPdbqt(file, transform=[transform, partial(transform2, pocket_alpha=pocket_dict[pocket_index])]) for file in dock_conformation_sm]
    # results 的长度是分子的个数
    return map_and_conjunction(list, statis_results)


if __name__ == '__main__':
    # IO file
    dock_out_folder = r'/mnt/e/Python_Project/SMTarRNA/project/data/3a6p/10k/'
    fpocket_out_folder = r'/mnt/e/Research/SM_miRNA/Data/Dock/complex/fpocket_results/3a6p_out/pockets'
    pocket = [('3a6p', 5),
              ('4z4c', 1),
              ('4z4d', 7),
              ('5zal', 7),
              ('5zam', 5),
              ('6cbd', 44),
              ('6lxd', 90),
              ('6v5b', 19)]
    results = main(dock_out_folder, fpocket_out_folder, 5)
