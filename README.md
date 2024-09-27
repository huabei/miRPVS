# Equivalent Graph Neural Network-based Accurate and Ultra-fast Virtual Screening of Small Molecules Targeting miRNA-Protein Complex

![](img/workflow.png)

## Contents

- [Equivalent Graph Neural Network-based Accurate and Ultra-fast Virtual Screening of Small Molecules Targeting miRNA-Protein Complex](#equivalent-graph-neural-network-based-accurate-and-ultra-fast-virtual-screening-of-small-molecules-targeting-mirna-protein-complex)
  - [Contents](#contents)
  - [Software Requirements](#software-requirements)
    - [OS Requirements](#os-requirements)
    - [Python Dependencies](#python-dependencies)
  - [Installation Guide](#installation-guide)
    - [download this repo](#download-this-repo)
    - [install env](#install-env)
  - [Dataset Download and Processing](#dataset-download-and-processing)
  - [Ligand Docking](#ligand-docking)
  - [Train model](#train-model)
  - [Model Tuning](#model-tuning)
  - [eval](#eval)
  - [predict](#predict)

If you find it useful, please cite:

**Equivalent Graph Neural Network-based Virtual Screening of Ultra-large chemical libraries  Targeting miRNA-protein complex**
Huabei Wang; Zhimin Zhang; Guangyang Zhang, Ming Wen\* and Hongmei Lu\*.
*Will Published in:*
*DOI:* [](<>)

## Software Requirements

autodock vina

python

### OS Requirements

The package development version is tested on *Linux: Ubuntu 22.04* operating systems.

### Python Dependencies

Dependencies for miRPVS:

```
pytorch
pyg
rdkit=2022.09.1
```

## Installation Guide

### download this repo

```
git clone https://github.com/huabei/miRPVS.git
```

### install env

you can install the env via yaml file

```
cd miRPVS
conda env create -f requirements.yaml
conda activate miRPVS
```

this project use [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) as the base project.

## Dataset Download and Processing

The entire docking dataset can be downloaded from the [ZINC20 Tranches](https://zinc20.docking.org/tranches/home/#), and we chose a subset of the *drug-like* data containing *3D* structures as the docking dataset.

The molecules were downloaded directly into *pdbqt* format, which can be used directly for autodock vina docking.

Due to the large amount of data, we can construct an index file for the entire dataset in order to facilitate statistics and sampling of the dataset.
By running the following command, you can generate an index file for each subfolder, as well as a structural information file for the molecules.

```bash
# assert the zinc data is placed in zinc_drug-like_3d folder
cd data
mkdir zinc20_drug-like_3d
cd zinc20_drug-like_3d
# run zinc download file here.
# after download complete, get all molecule index.
cd ..
ls zinc20_drug-like_3d | xargs -I {} python create_zinc20_hdf5.py {}
```

## Sample Ligand and Docking

### Sample
Use the following command to extract 1/600 of all molecules to train the model and randomly sample 10k molecules out of the extracted molecules to optimize the docking parameters.
```bash
cd data
python sample_data.py zinc20_drug-like_3d
```

### Docking
Docking a large number of molecules is recommended to be done using multiple compute nodes, if you are using a slurm cluster, you can refer to the file `data/submit_batch_dock_job.py` to assign the docking task.

## Constructing the training dataset
After getting the docking results for all molecules, assumed that all the docking energies are saved in the dock_results folder. Run the following command to construct the training dataset.
```bash
cd data
python construct_train_dataset_from _dock_output.py dock_results
```
It will create *dataset* folder to save the raw data.

## Train model

> `config` folder 

You just need to configure your own hyperparameters in config/experiment and then run：

```bash
python src/train.py experiment=egnn
```

The configuration used for this job is also stored in the config/experiment directory and can be used directly.The config folder contains the hyperparameter configuration files for the training model, which can be changed as appropriate.

## Model Tuning

Similarly, modify the configuration file `config/experiment/egnn_tune.yaml` and run the following command to perform a hyperparameter search of the model.
```bash
python src/train.py experiment=egnn_tune
```

## eval

The `config/eval.yaml` file needs to be configured with your data locations, model parameter paths, etc. And run:

```shell
python src/eval.py
```


## Screen The Whole Dataset

The `config/predict.yaml` file needs to be configured with your data locations, model parameter paths, etc. And run:

```shell
python src/predict.py
```
