# Copyright 2021 Zhongyang Zhang
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib2 import Path
import gzip
import re
from tqdm import tqdm
import random
import io
from collections import defaultdict


def load_model_path(root=None, version=None, v_num=None, best=False):
    """ When best = True, return the best model's path in a directory 
        by selecting the best model with the largest epoch. If not, return
        the last model saved. You must provide at least one of the 
        first three args.
    Args: 
        root: The root directory of checkpoints. It can also be a
            model ckpt file. Then the function will return it.
        version: The name of the version you are going to load.
        v_num: The version's number that you are going to load.
        best: Whether return the best model.
    """

    def sort_by_epoch(path):
        name = path.stem
        epoch = int(name.split('-')[1].split('=')[1])
        return epoch

    def generate_root():
        if root is not None:
            return root
        elif version is not None:
            return str(Path('lightning_logs', version, 'checkpoints'))
        else:
            return str(Path('lightning_logs', f'version_{v_num}', 'checkpoints'))

    if root == version == v_num == None:
        return None

    root = generate_root()
    if Path(root).is_file():
        return root
    if best:
        files = [i for i in list(Path(root).iterdir()) if i.stem.startswith('best')]
        files.sort(key=sort_by_epoch, reverse=True)
        res = str(files[0])
    else:
        res = str(Path(root) / 'last.ckpt')
    return res


def load_model_path_by_args(args):
    return load_model_path(root=args.load_dir, version=args.load_ver, v_num=args.load_v_num)


def check_exists(index_file: str, root_dir: str) -> list:
    """
    this function to check download file exists
    :param index_file: a file which include line of file path
    :param root_dir: the folder need to check
    :return: not exist file list
    """
    not_exists_file = list()
    with open(index_file, 'r') as f:
        for f_path in f:
            f_path = os.path.join(root_dir, f_path.strip())
            if os.path.exists(f_path):
                continue
            elif os.path.exists(f_path):
                not_exists_file.append(f_path)
    return not_exists_file


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
    A class for pdbqt.gz file, this class could transfer str dict to some friendly format.
    """

    def __init__(self, pdbqt_file, transform=None):
        self.f_str = gzip.open(pdbqt_file, mode='rb').read().decode()
        # print(self.f_str[:1000])
        self.zinc_id = re.findall('Name = (.*?)\n', self.f_str)
        # print(len(self.zinc_id))
        self.molecules = re.findall('MODEL.*?\n(.*?)ENDMDL\n', self.f_str, re.S)
        self.data = list(zip(self.zinc_id, self.molecules))
        if transform is not None:
            self.data = list(filter(transform, self.data))
            # self.data = list(zip(self.zinc_id, self.molecules))

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
        return random.shuffle(self.data)[:num]


def gz_writer(file_name: str) -> io.TextIOWrapper:
    """ get a file name, return a gz file api with wb mode"""
    output = gzip.open(file_name, 'wb')
    ecn = io.TextIOWrapper(output, encoding='utf-8')
    return ecn


def generate_coor(pdbqt_model: str, elements_list: list) -> list:
    """
    analyze pdbqt format str with \n, return all atom in a given elements_list molecular
    :param pdbqt_model: pdbqt format str
    :param elements_list: all elements molecular could include
    :return: list:[element, x, y, z].str
    """
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


def ele_transform(zinc_pdbqt_item, elements_list):
    """
    if pdbqt item have element that not in elements_list, return False, else return True.
    Use in filter() function.
    :param zinc_pdbqt_item: [..., pdbqt_str]
    :param elements_list: ['H', 'C', 'O']
    :return: True or False.
    """
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
