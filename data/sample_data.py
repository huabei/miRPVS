
import os
import sys
import random
import pickle
import re
import gzip
from tqdm import tqdm
from pandas import HDFStore
import io


class ZincPdbqt:
    """
    A class for pdbqt or pdbqt.gz file, this class could transfer str dict to some friendly format.
    """

    def __init__(self, pdbqt_file):
        # 读取.pdbqt.gz文件，转换为str
        self.f_str = gzip.open(pdbqt_file, mode="rb").read().decode()
        # 读取.pdbqt.gz文件中的zinc_id
        self.zinc_id = re.findall("Name = (.*?)\n", self.f_str)
        # 读取.pdbqt.gz文件中的分子结构
        if self.f_str.startswith("MODEL"):
            self.molecules = re.findall("MODEL.*?\n(.*?)ENDMDL\n", self.f_str, re.S)
        else:
            self.molecules = [self.f_str]
        # 生成一个list，包含zinc_id和分子结构
        self.data = list(zip(self.zinc_id, self.molecules))

    @property
    def data_dict(self):
        return dict(zip(self.zinc_id, self.molecules))

def gz_writer(file_name: str) -> io.TextIOWrapper:
    """get a file name, return a gz file api with wb mode"""
    output = gzip.open(file_name, "wb")
    ecn = io.TextIOWrapper(output, encoding="utf-8")
    return ecn


def write_pdbqt_to_gz(pdbqt_list, gz_file):
    """write a list of pdbqt to gz file"""
    with gz_writer(gz_file) as f:
        for pdbqt in tqdm(pdbqt_list, desc="write to gz"):
            f.writelines("MODEL\n" + pdbqt[1] + "ENDMDL\n")


if __name__ == '__main__':
    args = sys.argv
    if len(args) != 2:
        print("Usage: python sample_data.py <data_path>")
        sys.exit(1)
    ZINC_DATA_PATH = args[1]
    hdf_index_file = [
        os.path.join(ZINC_DATA_PATH, i) for i in os.listdir(ZINC_DATA_PATH) if i.endswith("_index.h5")
    ]
    hdf_coor_file = [
        os.path.join(ZINC_DATA_PATH, i) for i in os.listdir(ZINC_DATA_PATH) if i.endswith("_coor.h5")
    ]
    hdf_index_file.sort()
    hdf_coor_file.sort()
    all_hdf_file = list(zip(hdf_index_file, hdf_coor_file))

    # Random sample 1/600 molecule from all molecule
    random_sample_molecule = dict()

    ratio = 1 / 600 # choose 1/600 molecule from all molecule

    for index_file, _ in tqdm(all_hdf_file):
        index_store = HDFStore(index_file, mode="r")
        for path, sub_group, datasetes in tqdm(index_store.walk()):
            for dataset in datasetes:
                d = os.path.join(path, dataset)
                zinc_id_tmp = index_store.get(d).index.to_list()
                random_sample_molecule[d] = [zi for zi in zinc_id_tmp if random.random() < ratio]
        index_store.close()

    with open("zinc20_druglike_random_sample_molecule_1f600.pkl", "wb") as f:
        pickle.dump(random_sample_molecule, f)
    
    print("Random sample 1/600 molecule done!")

    # Use index to get molecule
    pdbqt_list = []
    for k, v in tqdm(random_sample_molecule.items(), desc="read pdbqt.gz"):
        # skip empty list
        if not v:
            continue
        file = ZINC_DATA_PATH + k.replace("_", ".") + ".pdbqt.gz"
        zinc_pdbqt = ZincPdbqt(file).data_dict
        try:
            for zinc_id in v:
                pdbqt_list.append((zinc_id, zinc_pdbqt[zinc_id]))
        except KeyError:
            print(f"{zinc_id} not in {file}")
            continue
    write_pdbqt_to_gz(pdbqt_list, "zinc20_druglike_random_sample_molecule_1f600.pdbqt.gz")
    print("Write pdbqt to zinc20_druglike_random_sample_molecule_1f600.pdbqt.gz done!")
    pdbqt_list_10k = random.sample(pdbqt_list, 10000)
    write_pdbqt_to_gz(
        pdbqt_list_10k, "zinc20_druglike_random_sample_molecule_1f600_10k.pdbqt.gz"
    )