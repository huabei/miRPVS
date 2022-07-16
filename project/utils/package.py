
import numpy as np
import matplotlib.pyplot as plt
import requests
from collections import defaultdict
from rdkit import Chem

def plot_fit_confidence_bond(x, y, r2, annot=True):
    # fit a linear curve an estimate its y-values and their error.
    a, b = np.polyfit(x, y, deg=1)
    y_est = a * x + b
    # y_err = x.std() * np.sqrt(1 / len(x) +
    #                           (x - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))

    fig, ax = plt.subplots()
    ax.plot([-20, 0], [-20, 0], '-')
    ax.plot(x, y_est, '-')
    # ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)
    ax.plot(x, y, 'o', color='tab:brown')
    if annot:
        num = 0
        for x_i, y_i in zip(x, y):
            ax.annotate(str(num), (x_i, y_i))
            # if y_i > -3:
            #     print(num)
            num += 1
    ax.set_xlabel('True Energy(Kcal/mol)')
    ax.set_ylabel('Predict Energy(Kcal/mol)')
    # ax.text(0.1, 0.5, 'r2:  ' + str(r2))
    ax.text(0.4, 0.9,
            'r2:  ' + str(r2), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
            fontsize=12)
    return fig


def send_to_wechat(message):
    key = 'SCT67936Tpp9RtEM5SnSNxczhMTKaMzW1'
    url = f'https://sctapi.ftqq.com/{key}.send'
    return requests.post(url=url, data=message)


def get_pdb_atom_info(file_path):
    """get [[atom_type, x, y, z], ...]"""
    f = open(file_path, 'r')
    # 分别将HETATM和ATOM类型原子的信息进行统计
    atom_info_dict = defaultdict(list)
    for row in f:
        if row[:6] in ['HETATM', 'ATOM  '] :
            x = float(row[30:38])
            y = float(row[38:46])
            z = float(row[46:54])
            atom_type = row[76:78]
            atom_info_dict[row[:6].strip()].append((atom_type, x, y, z))
    f.close()
    return atom_info_dict


def get_pdbqt_info(file_path: str, idx=0):
    # get file path, return [(atom_type, x, y, z), ...] of modules[cont]
    # if modules[idx] is [], return modules[0]
    def read_atom_xyz(lines:list):
        data = [[line[13:15].strip(), float(line[30:38]), float(line[38:46]), float(line[46:54])] for line in lines if line[:4] == 'ATOM']
        # for i, atom in enumerate(data):
        #     if atom[0] != 'Br':
        #         data[i][0] = atom[0][0]
        return data
    f = open(file_path, 'r')
    modules = f.read().strip().split('ENDMDL')
    out_data = list()
    for module in modules:
        lines = module.split('\n')
        atom_xyz = read_atom_xyz(lines=lines)
        out_data.append(atom_xyz)
    # 如果索引idx超过构象数目，则返回第一个构象
    if idx > len(out_data) -1:
        idx = 0
    return out_data[idx]


def get_autodock_info(file_path):
    '''return [[energy(kcal/mol), lb(rmsd), ub(rmsd)], ...]'''
    if file_path[-6:] == '.pdbqt':
        f = open(file_path, 'r')
        dock_data = list()
        for line in f:
            if line[:18] == 'REMARK VINA RESULT':
                x = line.split(' ')
                x = [i for i in x if i != ''][3:]
                x = [float(i) for i in x]
                dock_data.append(x)
        f.close()
    else:
        print(' Nonsupport file {}'.format(file_path[-6:]))
    return dock_data


def get_autodock_best_score(file_path):
    # 获取Autodock输出文件中第一个构象的对接分数
    f = open(file_path, 'r')
    for line in f:
        if line[:18] == 'REMARK VINA RESULT':
            score = float(line[24:30])
            break
    f.close()
    return score


def get_sdf_info(inf):
    # 按照sdf文件中的分子顺序，输出分子
    data_set = dict()
    with Chem.ForwardSDMolSupplier(inf) as fsuppl:
        mol_count = 0
        for mol in fsuppl:
            mol_count += 1
            if mol is None:
                print(f'wrong mol: {mol_count}')
                prop_dict = None
                # continue
            else:
                prop_dict = mol.GetPropsAsDict()
                conformer = mol.GetConformer().GetPositions()
                atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
                pos = [[atoms[i]] + conformer.tolist()[i] for i in range(len(atoms))]
                # for i, atom in enumerate(pos):
                #     if atom[0] != 'Br':
                #         pos[i][0] = atom[0][0]
                prop_dict['pos'] = pos
            data_set[str(mol_count)] = prop_dict
    return data_set


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


