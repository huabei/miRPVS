#!/usr/bin/bash
rsync -av -e 'ssh -p 8220' /home/huabei/projects/SMTarRNA hpc192311018@hpclogin1.csu.edu.cn:/public/home/hpc192311018/Huabei/project/SMTARRNA-sync --exclude=logs  --exclude=log --exclude=.env --exclude=dataset --exclude=local
