#!/usr/bin/bash
rsync -av -e 'ssh -p 22' /home/huabei/projects/SMTarRNA huabei@192.168.0.254:/home/huabei/Project/ --exclude=logs  --exclude=log --exclude=.env --exclude=dataset --exclude=local
