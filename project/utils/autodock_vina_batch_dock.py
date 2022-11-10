# -*- coding: UTF-8 -*-
"""
@author:ZNDX
@file:autodock_vina_batch_dock.py
@time:2022/10/04
"""
from vina import Vina
from utils import ZincPdbqt, gz_writer, ele_filter
import os
from tqdm import tqdm
import pickle
from functools import partial
import sys


def main(params):
    v = Vina(sf_name='vina')
    # 设置受体文件
    receptor_name = params['receptor_name']
    v.set_receptor(rigid_pdbqt_filename=receptor_name+'.pdbqt')
    # 计算受体力场
    # 3a6p:center=[11.944, 77.055, 34.959], box_size=[40, 40, 40]
    v.compute_vina_maps(center=[11.944, 77.055, 34.959], box_size=[20, 20, 20], spacing=0.375)
    # v.set_ligand_from_file("test_ligand.pdbqt")
    elements_list = ['C', 'H', 'O', 'N', 'S', 'P', 'BR', 'CL', 'F', 'I']
    ligands_file_name = 'zinc/zinc_drug_like_3d_100k_to_10k_rand.pdbqt.gz'
    # ligands_file_name = 'test_ligand.pdbqt.gz'
    filter_ = partial(ele_filter, elements_list=elements_list)
    ligands = ZincPdbqt(ligands_file_name, filter_=[filter_])
    # print(len(ligands))
    ligands = ligands[params['slicer']]
    index_area = '{start}-{stop}'.format(start=params['slicer'].start, stop=params['slicer'].stop)
    print(f'ready to dock ligands index {index_area}')
    dock_results_file_name = ligands_file_name.replace('.pdbqt.gz', f'_{receptor_name}_dock_results_{index_area}.pdbqt.gz')
    dock_energy_dict_file_name = ligands_file_name.replace('.pdbqt.gz', f'_{receptor_name}_dock_energy_{index_area}.pkl')
    dock_energy = dict()
    # dock_results_file_name_list = [ligands_file_name.replace('.pdbqt.gz', f'_dock_results_{i}.pdbqt.gz') for i in
                                   # range((len(ligands) // 1000) + 1)]

    # print(dock_results_file_name_list)
    writer = gz_writer(dock_results_file_name)
    i = 0
    # f = open(f'docked_100k_zinc_id{index_area}.txt', 'w')
    docked_ligand_id_tmp = list()
    # 进行对接
    for ligand_id, ligand_pdbqt in tqdm(ligands, desc='docking'):
        i += 1
        # 传入配体pdbqt字符串
        v.set_ligand_from_string(ligand_pdbqt)
        # 对接
        v.dock(exhaustiveness=96)
        # 将最后一个能量结果存入文件
        energy = v.energies()[0, 0]
        # 存入字典
        dock_energy[ligand_id] = energy
        # 写入最优构象
        writer.writelines(v.poses(n_poses=1))
        # 写入缓存
        if i % 100 == 0:
            writer.buffer.flush()
        # 将zinc id 存入列表
        # docked_ligand_id_tmp.append(ligand_id + '\n')
        # i+=1
    # 写入zinc id
    # f.writelines(docked_ligand_id_tmp)
    # f.close()
    writer.close()
    # 输出zinc id和对接能量字典
    with open(dock_energy_dict_file_name, 'wb', buffering=0) as handle:
        pickle.dump(dock_energy, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    _, start, stop = sys.argv
    params = dict(slicer=slice(eval(start), eval(stop)), receptor_name='3a6p_new')
    main(params)
    # pass