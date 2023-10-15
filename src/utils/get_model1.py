# 此文件用于处理对接输出的pdbqt/pdb文件，将其中的model1即最优的构象提取出来，包括多个配体的情况

import os
import sys
import re

def get_model1(pdb_file):
    f = open(pdb_file, 'r')
    f_str = f.read()
    f.close()
    if f_str.startswith("MODEL"):
        models = re.findall("(MODEL 1.*?\n.*?ENDMDL\n)", f_str, re.S)
    else:
        raise Exception("There is no model1 in the file")
    return models

if __name__ == "__main__":
    pdb_file = sys.argv[1]
    models = get_model1(pdb_file)
    if pdb_file.endswith('.pdbqt'):
        output_file = pdb_file[:-6] +".model1.pdbqt"
    elif pdb_file.endswith('.pdb'):
        output_file = pdb_file[:-4] +".model1.pdb"
    else:
        raise Exception("The file is not pdbqt or pdb file")
    with open(pdb_file.replace('.pdb' ,".model1.pdb"), 'w') as f:
        for model in models:
            f.write(model)
