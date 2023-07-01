#!/usr/bin/bash
rsync -av -e 'ssh -p 8220' hpc192311018@hpclogin1.csu.edu.cn:/public/home/hpc192311018/Huabei/project/SMTARRNA-sync/SMTarRNA/logs /home/huabei/projects/SMTarRNA/logs --exclude=test
