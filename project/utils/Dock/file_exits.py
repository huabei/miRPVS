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
import io
from collections import defaultdict

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
        # print(self.f_str[:1000])
        self.zinc_id = re.findall('Name = (.*?)\n', self.f_str)
        # print(len(self.zinc_id))
        self.molecules = re.findall('MODEL.*?\n(.*?)ENDMDL\n', self.f_str, re.S)

    def __iter__(self):
        return iter(zip(self.zinc_id, self.molecules))

    def __len__(self):
        return len(self.zinc_id)

    @property
    def data(self):
        return dict(zip(self.zinc_id, self.molecules))

    @property
    def scores(self):
        # 此函数用于给出auto dock vina结果的对接分数
        score = re.findall('REMARK VINA RESULT:(.*?) 0.000      0.000\n', self.f_str)
        return dict(zip(self.zinc_id, score))

    @property
    def elements(self):
        # 以字典的形式给出每种元素原子的个数
        lines = self.f_str.strip().split('\n')
        total_elements = defaultdict(lambda: 0)
        for line in lines:
            if line.startswith(('ATOM', 'HETATM')):
                # ele = line[12:16].strip()
                # 去除元素符号中的非字母字符
                ele = ''.join(filter(str.isalpha, line[12:16]))
                # 在元素符号字母数大于2的时候，保留元素的数目类型
                if len(ele) != 1 and ele.upper() not in ['BR', 'CL', 'SI']:
                    ele = ele[0]
                # 计数
                total_elements[ele] += 1
        return total_elements


def gz_writer(file_name):
    output = gzip.open(file_name, 'wb')
    ecn = io.TextIOWrapper(output, encoding='utf-8')
    return ecn


def generate_coor(pdbqt_model: str, elements_list: list):
    lines = pdbqt_model.strip().split('\n')
    pos = []
    for line in lines:
        if line.startswith(('ATOM', 'HETATM')):
            elements = line[12:16].strip()
            if len(elements) != 1 and elements not in ['BR', 'CL']:
                elements = elements[0]
            if elements not in elements_list:
                return False
            pos.append([elements, line[30:38].strip(), line[38:46].strip(), line[46:54].strip()])
    return pos


def write_data(f, data_dict):
    '''data_dict['id'] is a str;'pos' is a list of xyz tuple;['score'] is dock score'''
    f.write('\n')
    name = data_dict['id']
    pos = data_dict['pos']
    score = data_dict['score']
    f.write(name)
    f.write('\n')
    for line in pos:
        line = map(str, line)
        f.write(' '.join(line))
        f.write('\n')
    f.write(str(score))
    f.write('\n')
    return f


if __name__ == "__main__":
    zinc_pdbqt_file = 'zinc_drug_like_3d_10k.pdbqt.gz'
    ligand_pdbqt = ZincPdbqt(zinc_pdbqt_file)
    for i in ligand_pdbqt:
        print(i)
        break
