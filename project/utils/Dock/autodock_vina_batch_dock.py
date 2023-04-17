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
import time
TIME = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
OUTPUT_DIR = 'outputs'
EXHAUSTIVENESS = 64

def main(params):
    receptor_dict = {'3a6p': 'receptor/3a6p_dOthers_apH', '4z4c': 'receptor/4z4c_dOthers_apH',
                     '4z4d': 'receptor/4z4d_dOthers_apH', '6cbd': 'receptor/6cbd_dOthers_apH'}
    receptor_config = {'3a6p': dict(center=[11.944, 77.055, 34.959], box_size=[20, 20, 20]),
                       '4z4c': dict(center=[65.072, -9.582, 17.018], box_size=[34, 50, 32]),
                       '4z4d': dict(center=[60.345, -15.032, 23.241], box_size=[22, 30, 20]),
                       '6cbd': dict(center=[56.706, -12.643, 11.284], box_size=[18, 18, 16])}
    ligands_file_name = params['ligand_file']
    v = Vina(sf_name='vina')
    # 设置受体文件
    receptor_name = receptor_dict[params['receptor_name']]
    v.set_receptor(rigid_pdbqt_filename=receptor_name+'.pdbqt')
    # 计算受体力场
    # 3a6p
    v.compute_vina_maps(**receptor_config[params['receptor_name']])
    # 4z4c
    # v.compute_vina_maps(spacing=0.375, **receptor_config[params['receptor_name']])
    # 4z4d
    # v.compute_vina_maps(center=[60.345, -15.032, 23.241], box_size=[22, 30, 20], spacing=0.375)
    # 6cbd
    # v.compute_vina_maps(center=[56.706, -12.643, 11.284], box_size=[18, 18, 16], spacing=0.375)
    # v.set_ligand_from_file("test_ligand.pdbqt")
    # elements_list = ['C', 'H', 'O', 'N', 'S', 'P', 'BR', 'CL', 'F', 'I']
    # ligands_file_name = 'zinc/zinc_drug_like_3d_100k_to_10k_rand.pdbqt.gz'
    # ligands_file_name = 'test_ligand.pdbqt.gz'
    # filter_ = partial(ele_filter, elements_list=elements_list)
    filter_ = lambda x: True # 不过滤
    ligands = ZincPdbqt(ligands_file_name, filter_=[filter_])
    # print(len(ligands))
    ligands = ligands[params['slicer']]
    index_area = '{start}_{stop}'.format(start=params['slicer'].start, stop=params['slicer'].stop)
    print(f'ready to dock ligands index {index_area}')
    dock_results_file_name = ligands_file_name.replace('.pdbqt.gz', f'_{receptor_name}_dock_results_{index_area}_{EXHAUSTIVENESS}_{TIME}.pdbqt.gz')
    dock_energy_dict_file_name = ligands_file_name.replace('.pdbqt.gz', f'_{receptor_name}_dock_energy_{index_area}_{EXHAUSTIVENESS}_{TIME}.pkl')
    dock_energy = dict()

    # print(dock_results_file_name_list)
    writer = gz_writer(os.path.join(OUTPUT_DIR, os.path.basename(dock_results_file_name)))
    i = 0
    # 进行对接
    for ligand_id, ligand_pdbqt in tqdm(ligands, desc='docking'):
        i += 1
        try:
            # 传入配体pdbqt字符串
            v.set_ligand_from_string(ligand_pdbqt)
            # 对接
            v.dock(exhaustiveness=EXHAUSTIVENESS) # 参数设置
        except:
            print(f'ligand {ligand_id} dock error!')
            continue
        # 将能量结果存入文件
        energy = v.energies()
        # 存入字典
        dock_energy[ligand_id] = energy
        # 写入最优构象
        writer.writelines(v.poses())
        # 写入缓存
        if i % 100 == 0:
            writer.buffer.flush()
    writer.close()
    # 输出zinc id和对接能量字典
    with open(os.path.join(OUTPUT_DIR, os.path.basename(dock_energy_dict_file_name)), 'wb', buffering=0) as handle:
        pickle.dump(dock_energy, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'dock ligands index {index_area} done!')


if __name__ == '__main__':
    _, receptor, ligand_file, start, stop = sys.argv
    params = dict(slicer=slice(eval(start), eval(stop)), receptor_name=receptor, ligand_file=ligand_file)
    start_time = time.time()
    main(params)
    end_time = time.time()
    print(f'cost time: {end_time - start_time}s')
    # pass
