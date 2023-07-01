"""
@author:ZNDX
@file:autodock_vina_batch_dock.py
@time:2022/10/04
"""
import ast
import logging
import os
import pickle
import sys
import time

from tqdm import tqdm
from vina import Vina

from utils import ZincPdbqt, gz_writer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TIME = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))

OUTPUT_DIR = "outputs"
EXHAUSTIVENESS = 32


# 受体文件路径字典
RECEPTOR_FILE = {
    "3a6p": "receptor/3a6p_dOthers_apH.pdbqt",
    "4z4c": "receptor/4z4c_dOthers_apH.pdbqt",
    "4z4d": "receptor/4z4d_dOthers_apH.pdbqt",
    "6cbd": "receptor/6cbd_dOthers_apH.pdbqt",
}
# 受体配置
RECEPTOR_CONFIG = {
    "3a6p": dict(center=[11.944, 77.055, 34.959], box_size=[20, 20, 20]),
    "4z4c": dict(center=[65.714, -8.540, 16.711], box_size=[32, 40, 30]),
    "4z4d": dict(center=[59.285, -14.357, 23.829], box_size=[20, 28, 20]),
    "6cbd": dict(center=[55.639, -12.584, 9.444], box_size=[20, 20, 20]),
}


def main(params):
    # 配体文件路径
    ligands_file_name = params["ligand_file"]
    logging.info(f"ligands file name: {ligands_file_name}")

    # 实例化vina对接对象
    v = Vina(sf_name="vina")

    # 设置受体文件
    v.set_receptor(rigid_pdbqt_filename=RECEPTOR_FILE[params["receptor_name"]])

    # 计算受体力场
    v.compute_vina_maps(**RECEPTOR_CONFIG[params["receptor_name"]])

    # 过滤器
    # filter_ = lambda x: True # 不过滤

    ligands = ZincPdbqt(ligands_file_name, filter_=None)
    logging.info(f"total ligands: {len(ligands)}")

    ligands = ligands[params["slicer"]]
    logging.info(f'docking ligands index: {params["slicer"].start} - {params["slicer"].stop}')

    # 构建输出文件名：配体文件名_受体名_索引区间_穷举度_时间戳
    index_area = "{start}-{stop}".format(start=params["slicer"].start, stop=params["slicer"].stop)
    dock_structure_output_file = ligands_file_name.replace(
        ".pdbqt.gz",
        f'_{params["receptor_name"]}_dock_results_{index_area}_{EXHAUSTIVENESS}_{TIME}.pdbqt.gz',
    )
    dock_energy_dict_pkl_file = ligands_file_name.replace(
        ".pdbqt.gz",
        f'_{params["receptor_name"]}_dock_energy_{index_area}_{EXHAUSTIVENESS}_{TIME}.pkl',
    )

    dock_energy = dict()

    # 输出文件
    logging.info(f"dock structure output file: {dock_structure_output_file}")
    logging.info(f"dock energy dict pkl file: {dock_energy_dict_pkl_file}")

    writer = gz_writer(os.path.join(OUTPUT_DIR, os.path.basename(dock_structure_output_file)))

    i = 0  # 对接计数
    wrong_ligands = []  # 对接失败的配体
    # 进行对接
    for ligand_id, ligand_pdbqt in tqdm(ligands, desc="docking"):
        i += 1
        try:
            # 传入配体pdbqt字符串
            v.set_ligand_from_string(ligand_pdbqt)
            # 对接
            v.dock(exhaustiveness=EXHAUSTIVENESS)  # 参数设置
        except Exception as e:
            wrong_ligands.append(ligand_id)
            print(f"ligand {ligand_id} dock error!")
            continue
        # 获取对接能量
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
    with open(
        os.path.join(OUTPUT_DIR, os.path.basename(dock_energy_dict_pkl_file)), "wb", buffering=0
    ) as handle:
        pickle.dump(dock_energy, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"dock ligands index {index_area} done!")
    print(f"wrong ligands: {wrong_ligands}")


if __name__ == "__main__":
    _, receptor, ligand_file, start, stop = sys.argv
    assert start.isdigit() or start == "None"
    assert stop.isdigit() or stop == "None"
    params = dict(
        slicer=slice(ast.literal_eval(start), ast.literal_eval(stop)),
        receptor_name=receptor,
        ligand_file=ligand_file,
    )
    start_time = time.time()
    main(params)
    end_time = time.time()
    print(f"cost time: {end_time - start_time}s")
    # pass
