#!python3
# this file to verify file exit
# date: 20220628
# editor Huabei

import os
import gzip

import re
import pickle
from tqdm import tqdm
import random


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


def pdbqt_split(f_path) -> list:
    """generate a list of pdbqt from a gz file"""
    return re.findall('(MODEL.*?ENDMDL\n)', gzip.open(f_path, mode='rb').read().decode(), re.S)


def get_zinc_id_index(index_file, root_dir):
    # this function to get zinc_id from every pdbqt file
    zinc_index = dict()
    with open(index_file, 'r') as f:
        for f_path in tqdm(f):
            f_path = os.path.join(root_dir, f_path.strip())
            zinc_id = zinc_id_find(f_path)
            f_name = f_path.split('/')[-1]
            zinc_index[f_name] = zinc_id
    return zinc_index


def random_accept(ratio):
    "generate a random True with ratio "
    if random.random() < ratio:
        return True
    else:
        return False


class ZincPdbqt():
    def __init__(self, pdbqt_file):
        self.f_str = gzip.open(pdbqt_file, mode='rb').read().decode()
        self.zinc_id = re.findall('Name = (.*?)\n', self.f_str)
        self.molecules = re.findall('MODEL.*?\n(.*?)ENDMDL\n', self.f_str, re.S)
        self.model = zip(self.zinc_id, self.molecules)

    def __iter__(self):
        return iter(self.model)

    def __len__(self):
        return len(self.zinc_id)


if __name__ == "__main__":
    zinc_pdbqt_file = 'zinc_drug_like_3d_10k.pdbqt.gz'
    ligand_pdbqt = ZincPdbqt(zinc_pdbqt_file)
    for i in ligand_pdbqt:
        print(i)
        break
