
# set the working directory
import sys
import os
sys.path.append("..")

import gzip
import pickle

import pandas as pd

def analyze_zinc_pdbqt_gz(pdbqt_gz_path: str):
    """此函数用于分析ZINC的pdbqt.gz文件，将其转换成3维坐标数据和原子在3维坐标数据中的起始和终止位置
    input: pdbqt_gz_path: str, pdbqt.gz文件的路径
    output: coor: pd.DataFrame, 3维坐标数据
            index: pd.DataFrame, 每个分子中的原子在coor中的起始和终止位置
    """
    coor = []
    index = []
    # 读取pdbqt.gz文件
    with gzip.open(pdbqt_gz_path, "rb") as f:
        t_start = 0  # 记录当前分子的原子起始位置
        t_end = 0  # 记录当前分子的原子终止位置
        for line in f:
            if line.startswith(b"ATOM"):
                coor.append(
                    [
                        str(line[12:14].strip(), "utf-8"),
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ]
                )
                t_end += 1  # 记录已存入原子的个数
            if line.startswith(b"REMARK  Name = "):  # 一个分子的起始位置
                if t_end == 0:
                    # 记录第一个分子的id
                    zinc_id = str(line[15:].strip(), "utf-8")
                    continue
                index.append([zinc_id, t_start, t_end])  # 存储上一个分子的信息
                zinc_id = str(line[15:].strip(), "utf-8")  # 记录当前分子的id
                t_start = t_end  # 记录当前分子的原子起始位置
        index.append([zinc_id, t_start, t_end])
    return pd.DataFrame(coor, columns=["atom", "x", "y", "z"]), pd.DataFrame(
        index, columns=["zinc_id", "start", "end"]
    ).set_index("zinc_id", drop=True)

args = sys.argv
assert len(args) == 2, "Usage: python construct_train_dataset_from_dock_output.py <data_dir>"
data_dir = args[1]
data_files = os.listdir(data_dir)
pkl_files = [f for f in data_files if f.endswith(".pkl")]
pdbqt_gz_files = [f for f in data_files if f.endswith(".pdbqt.gz")]

# get dock energy data
total_data = dict()
for f in pkl_files:
    with open(os.path.join(data_dir, f), "rb") as f:
        data = pickle.load(f)
        total_data.update(data)
print(f"total data: {len(total_data)}")
total_data_best = {k: v[0] for k, v in total_data.items()}
total_data_best_df = pd.DataFrame.from_dict(
    total_data_best,
    columns=["total", "inter", "intra", "torsions", "intra best pose"],
    orient="index",
)

# get 3d coordinate data
structure_file = 'zinc20_druglike_random_sample_molecule_1f600.pdbqt.gz'
coor_df, index_df = analyze_zinc_pdbqt_gz(structure_file)

# merge data
total_data_best_df.index.name = "zinc_id"
total_data_best_df = index_df.join(total_data_best_df, how="left")

# store data
os.mkdir("datasets/raw")
store = pd.HDFStore("datasets/raw/raw.h5")
store["label"] = total_data_best_df
store["pos"] = coor_df
store.close()
print("store done!")