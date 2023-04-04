"""此scripts用于将ZINC所有的分子3维数据，合并成一个hdf5文件，方便后续的预测"""

import pandas as pd
import gzip
from pandas import HDFStore
import os
from tqdm import tqdm

def analyze_zinc_pdbqt_gz(pdbqt_gz_path: str):
    coor = []
    index = []
    with gzip.open(pdbqt_gz_path, 'rb') as f:
        t_start = 0
        t_end = 0
        for line in f:
            if line.startswith(b'ATOM'):
                t_end += 1
                coor.append([str(line[13:16].strip(), 'utf-8'), float(line[30:38]), float(line[38:46]), float(line[46:54])])
            if line.startswith(b'REMARK  Name = '):
                if t_end == 0:
                    zinc_id = str(line[15:].strip(), 'utf-8')
                    continue
                index.append([zinc_id, t_start, t_end])  # 存储上一个分子的信息
                zinc_id = str(line[15:].strip(), 'utf-8')  # 当前分子的id
                t_start = t_end
        index.append([zinc_id, t_start, t_end])
    return pd.DataFrame(coor, columns=['atom', 'x', 'y', 'z']), pd.DataFrame(index, columns=['zinc_id', 'start', 'end']).set_index('zinc_id', drop=True)

if __name__ == '__main__':
    data_folder = 'BJ'
    with HDFStore('zinc_drug_like_data.h5') as store:
        for path, sub_dir, filename in tqdm(os.walk(data_folder)):
            for file in filename:
                if file.endswith('.pdbqt.gz'):
                    # print(file)
                    d = os.path.join(path, file[:-13])
                    # store.put(os.path.join(path, file[:-13]), analyze_zinc_pdbqt_gz(os.path.join(path, file)))
                    store[d+'_coor'], store[d+'_index'] = analyze_zinc_pdbqt_gz(os.path.join(path, file))
