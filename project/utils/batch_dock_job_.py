
"""用于批量的提交对接任务，主要用于将大量的分子对接任务进行分割。"""
import os
import sys
import time
from utils import ZincPdbqt

# num 为单个节点的对接数目
_, receptor, ligand_dataset, node_num = sys.argv
node_num = int(node_num)
job_num = len(ZincPdbqt(ligand_dataset))
job_per_node = int(job_num/node_num)
job_ = list()

for i in range(node_num):
    print(f'start job {i}')
    if i == (node_num-1):
        # os.system('sbatch batch_dock.sh {} {} {}'.format(receptor, job_per_node * i, None))
        commd = 'sbatch batch_dock.sh {} {} {} {}'.format(receptor, ligand_dataset, job_per_node * i, None)
    else:
        commd = 'sbatch batch_dock.sh {} {} {} {}'.format(receptor, ligand_dataset, job_per_node * i, job_per_node * (i+1))
    print(commd)
    os.system(commd)
    time.sleep(10)

