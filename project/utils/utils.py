import collections
import os
import gzip
import re
from tqdm import tqdm
import random
import io
from collections import defaultdict
import pickle
import pandas as pd
import numpy as np
import copy
import functools
from rdkit.Chem import Descriptors

def zinc_id_find(f_path):
    return re.findall('Name = (.*?)\n', gzip.open(f_path, mode='rb').read().decode())


def pdbqt_split(f_path) -> list:
    """generate a list of pdbqt from a gz file"""
    return re.findall('(MODEL.*?ENDMDL\n)', gzip.open(f_path, mode='rb').read().decode(), re.S)


def get_zinc_id_index(index_file, root_dir) -> dict:
    """
    this function to get zinc_id from every pdbqt.gz file list in index_file
    :param index_file: down from zinc which include every single files relative path
    :param root_dir: the download path, which include files list in index_file
    :return: dict: key=file_name, value=list:zinc_id
    """
    zinc_index = dict()
    with open(index_file, 'r') as f:
        for f_path in tqdm(f):
            f_path = os.path.join(root_dir, f_path.strip())
            zinc_id = zinc_id_find(f_path)
            f_name = f_path.split('/')[-1]
            zinc_index[f_name] = zinc_id
    return zinc_index


def random_accept(ratio: float) -> bool:
    """
    generate a random True with ratio
    :param ratio: probability of True to return, such as 1/100
    :return: True with probability ratio
    """
    if random.random() < ratio:
        return True
    else:
        return False


class ZincPdbqt():
    """
    A class for pdbqt or pdbqt.gz file, this class could transfer str dict to some friendly format.
    """

    def __init__(self, pdbqt_file, filter_=None, transform=None):
        # 读取.pdbqt.gz文件，转换为str
        self.f_str = gzip.open(pdbqt_file, mode='rb').read().decode()
        # 读取.pdbqt.gz文件中的zinc_id
        self.zinc_id = re.findall('Name = (.*?)\n', self.f_str)
        # 读取.pdbqt.gz文件中的分子结构
        self.molecules = re.findall('MODEL.*?\n(.*?)ENDMDL\n', self.f_str, re.S)
        # 生成一个list，包含zinc_id和分子结构
        self.data = list(zip(self.zinc_id, self.molecules))
        # 过滤和转换
        if filter_ is not None:
            assert type(filter_) in (list, tuple), 'filter type must be list or tuple'
            for fil in filter_:
                self.data = list(filter(fil, self.data))
        if transform is not None:
            assert type(transform) in (list, tuple), 'transform type must be list or tuple'
            for trans in transform:
                self.data = list(map(trans, self.data))

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @property
    def data_dict(self):
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

    def random_sample(self, ratio: float) -> list:
        num = int(len(self.data) * ratio)
        return random.sample(self.data, num)


def zinc_pdbqt_transform_decorator(f):
    """for transform func, to split (zinc_id, data), and return (zinc_id, data)"""

    def wrapper(*args, **kwargs):
        zinc_id, data = args[0]
        return zinc_id, f(data, *args[1:], **kwargs)

    return wrapper


def gz_writer(file_name: str) -> io.TextIOWrapper:
    """ get a file name, return a gz file api with wb mode"""
    output = gzip.open(file_name, 'wb')
    ecn = io.TextIOWrapper(output, encoding='utf-8')
    return ecn

def write_pdbqt_to_gz(pdbqt_list, gz_file):
    """write a list of pdbqt to gz file"""
    with gz_writer(gz_file) as f:
        for pdbqt in tqdm(pdbqt_list, desc='write to gz'):
            f.writelines('MODEL\n'+pdbqt[1]+'ENDMDL\n')

def generate_coor(pdbqt_model: str):
    """
    analyze pdbqt format str with \n, return all atom in a given elements_list molecular
    :param pdbqt_model: pdbqt format str
    :return: list:[element, x, y, z].str or bool
    """
    lines = pdbqt_model.strip().split('\n')
    pos = []
    for line in lines:
        if line.startswith(('ATOM', 'HETATM')):
            atom_info = get_atom_inline(line)
            # elements = line[12:16].strip()
            elements = atom_info['atom_name']
            if len(elements) != 1 and elements not in ['BR', 'CL']:
                elements = elements[0]
            pos.append([elements, atom_info['x'], atom_info['y'], atom_info['z']])
    return pos


def ele_filter(zinc_pdbqt_item, elements_list=None):
    """
    if pdbqt item have element that not in elements_list, return False, else return True.
    Use in filter() function.
    :param zinc_pdbqt_item: [..., pdbqt_str]
    :param elements_list: ['H', 'C', 'O']
    :return: True or False.
    """
    assert elements_list is not None, 'elements_list is None'
    lines = zinc_pdbqt_item[1].strip().split('\n')
    elements_list = [i.upper() for i in elements_list]
    for line in lines:
        if line.startswith(('ATOM', 'HETATM')):
            # ele = line[12:16].strip()
            # 去除元素符号中的非字母字符
            ele = ''.join(filter(str.isalpha, line[12:16]))
            if ele.upper() in elements_list:
                continue
            else:
                return False
        else:
            continue
    return True


def write_data(f, data_dict: dict):
    """
    a dataset text writer.
    :param f: file object
    :param data_dict: key=['id', 'pos', 'score']
    :return: f
    """
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


def read_dock_score(origin_data_path: str) -> dict:
    """
    load .pkl file generated from dock process
    :param origin_data_path: where pkl file located
    :return: all dock energy in format dict(zinc_id=energy)
    """
    file_list = [os.path.join(origin_data_path, i) for i in os.listdir(origin_data_path) if i.endswith('.pkl')]
    score_dict = dict()
    for file in file_list:
        f = open(file, 'rb')
        score_dict_tmp = pickle.load(f)
        f.close()
        score_dict.update(score_dict_tmp)
    return score_dict


def statistic_pocket_interaction(pocket_atom):
    """statistic 17-20 letters by length in every line start with ATOM in file pocket_atom
    output like:
                {'len_three': 12,
                'len_one': 15}"""
    pocket_atom = pocket_atom.strip().split('\n')
    pocket_atom = [i for i in pocket_atom if i.startswith('ATOM')]
    pocket_atom = [i[17:20].strip() for i in pocket_atom]
    pocket_atom = [len(i) for i in pocket_atom]
    pocket_atom = collections.Counter(pocket_atom)
    return pocket_atom


def ligand_pocket_position_statistics(pocket_alpha: list, atom_list):
    """
    计算对接后的分子中m个原子距离pocket中n个alpha球的距离,return nxm
    :param pocket_alpha: [[atom_name, x, y, z], [atom_name, x, y, z], ...]
    :param atom_list: [[element, x, y, z], [element, x, y, z], ...]
    :return: 1xm, 分子中每个原子最近的alpha球的距离
    """
    atom_xyz = get_xyz(atom_list)
    alpha_sphere = get_xyz(pocket_alpha)
    # print(alpha_sphere)
    # 计算分子中每个原子据所有alpha球的距离（n, 3, m)， n为alpha球的个数
    vector_matrix = atom_xyz.T[np.newaxis, :] - alpha_sphere[:, :, np.newaxis]
    # 利用爱因斯坦求和简记法对中间一个维度求和->(n, m)
    distance_matrix = np.einsum('ijk, ijk->ik', vector_matrix, vector_matrix)
    # 返回所有原子最近alpha球的距离(1xm)
    return np.mean(np.sqrt(np.min(distance_matrix, axis=0)))


def get_pocket_info(pocket_folder):
    pocket_num = int(len(os.listdir(pocket_folder)) / 2)
    # print(pocket_num)
    # 获取pocket中alpha球的位置
    pocket_dict = dict()
    for i in range(pocket_num):
        i += 1
        pocket_dict[i] = get_pdb_atom_info(os.path.join(pocket_folder, f'pocket{i}_vert.pqr'))['ATOM']
    return pocket_dict


def get_atom_inline(line):
    """
    get atom info from line
    :param line: line start with ATOM
    :return: dict
    """
    atom_info = dict()
    atom_info['atom_name'] = line[12:16].strip()
    atom_info['residue_name'] = line[17:20].strip()
    atom_info['chain_id'] = line[21:22].strip()
    atom_info['residue_number'] = line[22:26].strip()
    atom_info['x'] = float(line[30:38].strip())
    atom_info['y'] = float(line[38:46].strip())
    atom_info['z'] = float(line[46:54].strip())
    return atom_info


def get_xyz(atom_list):
    df = pd.DataFrame(atom_list, columns=['atom', 'x', 'y', 'z'])
    # print(df['atom'])
    return df[['x', 'y', 'z']].to_numpy()


def get_pdb_atom_info(file_path):
    """get [[atom_type, x, y, z], ...]"""
    f = open(file_path, 'r')
    # 分别将HETATM和ATOM类型原子的信息进行统计
    atom_info_dict = defaultdict(list)
    for row in f:
        if row[:6] in ['HETATM', 'ATOM  '] :
            atom_info = get_atom_inline(row)
            atom_info_dict[row[:6].strip()].append((atom_info['atom_name'],
                                                    atom_info['x'],
                                                    atom_info['y'],
                                                    atom_info['z']))
    f.close()
    return atom_info_dict


def map_and_conjunction(func, iterables):
    """
    map and conjunction
    :param func: function
    :param iterables: iterables
    :return: conjunction result
    """
    assert len(iterables) > 1, 'iterables must be more than 1'
    return functools.reduce(lambda x, y: x+y, list(map(func, iterables)))

