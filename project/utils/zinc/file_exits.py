#!python3
# this file to verify file exit
# date: 20220628
# editor Huabei

import os
import gzip

from sympy import re, root
import pickle


def check_exists(index_file, root_dir):
    # this function to check download file exists
    not_exists_file = list()
    with open(index_file, 'r') as f:
        for f_path in f:
            f_path = os.path.join(root_dir, f_path.strip())
            if os.path.exists(f_path):
                # print('ok')
                continue
            elif os.path.exists(f_path):
                not_exists_file.append(f_path)
    return not_exists_file

def zinc_id_find(f_path):
    # zf = gzip.open(f_path, mode='rb')
    # content = zf.read().decode()
    return re.findall('Name = (.*?)\n', gzip.open(f_path, mode='rb').read().decode())


def get_zinc_id_index(index_file, root_dir):
    # this function to get zinc_id from every pdbqt file
    zinc_index = dict()
    with open(index_file, 'r') as f:
        for f_path in f:
            f_path = os.path.join(root_dir, f_path.strip())
            zinc_id = zinc_id_find(f_path)
            f_name = f_path.split('/')[-1]
            zinc_index[f_name] = zinc_id
    return zinc_index



if __name__ == "__main__":
    index_file = 'ZINC-downloader-3D-pdbqt.gz.database_index'
    # f_path = next(f)
    root_dir = os.getcwd()
    zinc_pdbqt_index = get_zinc_id_index(index_file, root_dir)
    pkl_name = 'zinc_pdbqt_3d_index.pkl'
    with open(pkl_name, 'wb') as handle:
        pickle.dump(zinc_pdbqt_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('--ok---'*5)