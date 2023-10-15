# 通过zinc id在整个数据库中找到对应的pdbqt文件

from ..dock_utils.utils import ZincPdbqt, write_pdbqt_to_gz
import logging
from tqdm import tqdm
# 读取需要处理的数据
file_name = 'four_complex_zinc_id_index.txt'

data = dict()
with open(file_name, 'r') as f:
    its = f.read().strip().split('\n\n')
    for i in its:
        t = i.split('\n')
        data[t[0]] = t[1:]

root_folder = '/public/home/hpc192311018/Huabei/data/ZINC-DrugLike-3D-20230407'

results = list()
for key, value in tqdm(data.items()):
    # logging.info(f'processing {key}')
    file_path = root_folder + key.replace('_', '.') + '.pdbqt.gz'
    zinc_pdbqt = ZincPdbqt(file_path)
    for p in zinc_pdbqt:
        if p[0] in value:
            results.append(p)
write_pdbqt_to_gz(results, 'four_complex_zinc_top_data_pdbqt.pdbqt.gz')