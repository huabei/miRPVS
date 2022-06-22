# %%
from argparse import ArgumentParser
from model.origgnn import MolecularGNN
from utils.package import plot_fit_confidence_bond
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch
import numpy as np
import time
from sklearn.metrics import r2_score
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import random
import math

# %%
# %%wandb
def main(hparams):
    
    model_name = f'3dgnn-dim-{hparams.dim}-hlayer-{hparams.layer_hidden}-olayer-{hparams.layer_output}-' + time.strftime("%Y%m%d_%H%M%S", time.localtime())
    dict_args = vars(hparams)
    model = MolecularGNN(**dict_args)
    # logger
    wandb_logger = pl.loggers.TensorBoardLogger(save_dir='log/origgnn', name=model_name)
    # callbacks
    # early stopping
    early_stopping = pl.callbacks.early_stopping.EarlyStopping(monitor='val_loss', patience=20, mode='min')
    # checkpoint
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss', mode='min', save_last=True,
                                                         dirpath='checkpoints', filename=model_name)
    if hparams.checkpoint == None:
        trainer = Trainer.from_argparse_args(hparams, logger=wandb_logger, auto_lr_find=True, callbacks=[early_stopping, checkpoint_callback])
    else:
        trainer = Trainer(resume_from_checkpoint=hparams.checkpoint, callbacks=[early_stopping])
    # trainer.tune(model)
 
    # Train
    trainer.fit(model)
    # trainer.save_checkpoint(time.strftime("%Y%m%d_%H%M%S", time.localtime()) + ".ckpt")
    trainer.test(model, dataloaders=model.val_dataloader(), verbose=False)
    tensorboard = model.logger.experiment
    x = np.array(model.predictions['true'])
    y = np.array(model.predictions['pred'])
    val_r2 = r2_score(x, y)
    val_fig = plot_fit_confidence_bond(x, y, val_r2, annot=False)
    
    model.predictions = defaultdict(list)
    trainer.test(model, dataloaders=model.train_dataloader(), verbose=False)
    x = np.array(model.predictions['true'])
    y = np.array(model.predictions['pred'])
    train_r2 = r2_score(x, y)
    train_fig = plot_fit_confidence_bond(x, y, train_r2, annot=False)
    if True:
        tensorboard.add_figure('train_res', train_fig)
        tensorboard.add_figure('val_res', val_fig)
        model.log({'val_r2': val_r2, 'train_r2':train_r2})

def add_args():
    parser = ArgumentParser()
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--layer_hidden",type=int, default=24)
    parser.add_argument("--layer_output",type=int, default=8)
    parser.add_argument("--batch_size",type=int, default=128)
    parser.add_argument("--data_path",type=str, default=None)
    parser.add_argument("--checkpoint",type=str, default=None)
    return parser


# %%

if __name__ == "__main__":
    
    dataset_path = '/public/home/hpc192311018/Huabei/data/in-man-orig-conformation/exhaus_96/in_man_exhaustiveness_96_orig_conformation.txt'
    checkpoint = '20220618_211314.ckpt'
    # add model args
    parser = add_args()
    parser = MolecularGNN.add_model_specific_args(parent_parser=parser)
    # add Trainer args
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args(['--data_path', dataset_path, '--learning_rate', '0.0001', '--gpus=1', '--max_epochs', '1000'])

    main(args)

# %%
from multiprocessing import Pool
from argparse import ArgumentParser
class ConfigGenerater():
    def __init__(self, sweep_config: dict) -> None:
        self.name = sweep_config['name']
        self.method = sweep_config['method']
        self.param_grid = sweep_config['parameters']

    def get_random_value(self, conditions: dict):
        if 'distribution' in conditions.keys():
            min, max = conditions['min'], conditions['max']
            if conditions['distribution'] == 'log_uniform_values':
                value = random.uniform(math.log(min), math.log(max))
                return math.exp(value)
            elif conditions['distribution'] == 'int_uniform':
                value = random.randint(min, max)
                return value
            elif conditions['distribution'] == 'uniform':
                value = random.uniform(min, max)
                return value
            else:
                print(f"not surport distribution: {conditions['distribution']}")
        elif 'value' in conditions.keys():
            value = conditions['value']
            return value
        elif 'values' in conditions.keys():
            return random.sample(conditions['values'], 1)[0]
        elif 'min' in conditions.keys() and 'max' in conditions.keys():
            if type(conditions['min']) == int and type(conditions['max']) == int:
                value = random.randint(conditions['min'], conditions['max'])
                return value
            else:
                assert type(conditions['min']) == float
                assert type(conditions['max']) == float
                value = random.uniform(conditions['min'], conditions['max'])
                return value
        else:
            raise ValueError('Unknow Conditions')

    @property
    def get_random_config(self):
        random_params = {k: self.get_random_value(c) for k, c in self.param_grid.items()}
        return random_params

def arg_for_sweep(config: dict):
    # prepare args
    parser = add_args()
    dataset_path = '/public/home/hpc192311018/Huabei/data/in-man-orig-conformation/exhaus_96/in_man_exhaustiveness_96_orig_conformation.txt'
    checkpoint = '20220618_211314.ckpt'
    # add model args
    parser = MolecularGNN.add_model_specific_args(parent_parser=parser)
    # add Trainer args
    parser = Trainer.add_argparse_args(parser)
    hyperparameter_list = ['--data_path', dataset_path, '--gpus=1']
    for key, value in config.items():
        # print(key, value)
        hyperparameter_list.extend(['--' + key, str(value)])
    # print(hyperparameter_list)
    args = parser.parse_args(hyperparameter_list)
    # print('here is right')
    return args
def train(sweep_config: dict, cont: int, process=2):
    """process is multiprocess"""
    config_generater = ConfigGenerater(sweep_config=sweep_config)
    process_pool = Pool(3)
    for i in range(cont):
        print(f'start process{i}')
        config = config_generater.get_random_config
        print(config)
        args = arg_for_sweep(config=config)
        # main(args)
        process_pool.apply_async(main, (args, ))
    process_pool.close()
    process_pool.join()

# %%
sweep_config = {
  "name" : "sweep",
  "method" : "random",
  "parameters": {
    "max_epochs": {
      "value": 500
    },
    "learning_rate": {
      "distribution": "log_uniform_values",
      "min": 0.00001,
      "max": 0.001
    },
    "lr_decay": {
      "min": 0.95,
      "max": 0.999
    },
    "dim" : {
      "distribution": "int_uniform",
      "min": 128,
      "max": 512
    },
    "layer_hidden": {
      "distribution": "int_uniform",
      "min": 8,
      "max": 32
    },
    "layer_output": {
      "distribution": "int_uniform",
      "min": 8,
      "max": 20
    }
  }
}


# %%
import math
train(sweep_config, 10, process=2)

# %%
import os 
from multiprocessing import Pool
def main(args):
    time.sleep(3)
    print(args.dim, '\n')
os.getpid()
# time.sleep(5)


