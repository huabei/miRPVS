# Equivalent Graph Neural Network-based Virtual Screening of Ultra-large chemical libraries  Targeting miRNA-protein complex

![](img/workflow.png)

## Contents

- [Overview](#overview)
- [Software Requirements](#software-requirements)
- [Installation Guide](#installation-guide)
- [Demo: Ligand docking](#demo1--ligand-docking-on-pdbbind-core-set)
- [Demo: Virtual screening](#demo2--virtual-screening-on-dekois-20)

## Overview

MicroRNAs (miRNAs) are pivotal post-transcriptional regulators within cells and hold immense therapeutic potential for a wide range of diseases. Despite this promise, the development of small-molecule drugs targeting miRNAs remains relatively scarce. There is a need to establish efficient screening methodologies for identifying small molecules that modulate miRNAs, thereby assessing the feasibility of miRNAs as druggable targets.
In this study, we initiated a rapid screening approach for small-molecule drugs that influence miRNA biogenesis. This approach combines structure-based virtual screening with an equivariant graph neural network. Four miRNA protein complexes associated with miRNA migration or preprocessing were selected as targets and lead compounds were screened in the druglike 3D small molecule dataset in the Zinc20 database which has 600 million compounds, respectively. Remarkably, the screening process for all four complexes was completed within 48 hours, utilizing minimal computational resources.

If you find it useful, please cite:

**Equivalent Graph Neural Network-based Virtual Screening of Ultra-large chemical libraries  Targeting miRNA-protein complex**
Huabei Wang; Zhimin Zhang; Guangyang Zhang, Ming Wen\* and Hongmei Lu\*.
*Will Published in:*
*DOI:* [](<>)

## Software Requirements

### OS Requirements

The package development version is tested on *Linux: Ubuntu 22.04* operating systems.

### Python Dependencies

Dependencies for SMTarRNA:

```
pytorch
pyg
rdkit=2022.09.1
```

## Installation Guide

### download this repo

```
git clone https://github.com/huabei/SMTarRNA.git
```

### install env

you can install the env via yaml file

```
cd SMTarRNA
conda env create -f requirements.yaml
```

this project use [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) as the base project.

## ligand docking

## train model

### config

> This template is suitable for multi-platform operation, please note that the config/local is configured specifically for different platforms.

You just need to configure your own hyperparameters in config/experiment and then run：

```shell
python src/train.py experiment=exp_name
```

The configuration used for this job is also stored in the config/experiment directory and can be used directly.

## eval

The config/eval.yaml file needs to be configured with your data locations, model parameter paths, etc. And run:

```shell
python src/eval.py
```

## predict

The config/predict.yaml file needs to be configured with your data locations, model parameter paths, etc. And run:

```shell
python src/predict.py
```
