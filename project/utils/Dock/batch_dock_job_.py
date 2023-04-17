
"""用于批量的提交对接任务，主要用于将大量的分子对接任务进行分割。"""
import os
import sys
import time
from utils import ZincPdbqt

# 指定node_num,自动分配任务
_, receptor, ligand_dataset, node_num = sys.argv

node_num = int(node_num)

job_num = len(ZincPdbqt(ligand_dataset)) # 总的任务数

job_per_node = int(job_num/node_num) # 每个node上的任务数
job_ = list()

for i in range(node_num):
    print(f'start job {i}')
    if i == (node_num-1): # 最后一个node， 将剩余的任务全部分配给它
        # os.system('sbatch batch_dock.sh {} {} {}'.format(receptor, job_per_node * i, None))
        commd = 'sbatch batch_dock_2_jobs.sh {} {} {} {}'.format(receptor, ligand_dataset, job_per_node * i, job_num+1) # 由于python的切片是左闭右开，所以这里需要加1
    else:
        commd = 'sbatch batch_dock_2_jobs.sh {} {} {} {}'.format(receptor, ligand_dataset, job_per_node * i, job_per_node * (i+1))
    print(commd)
    os.system(commd)
    time.sleep(10)

