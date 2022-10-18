from argparse import ArgumentParser
from model.origgnn import MolecularGNN
# from utils.package import plot_fit_confidence_bond
from pytorch_lightning import Trainer
# import pytorch_lightning as pl
# from collections import defaultdict
# from torch.utils.data import DataLoader
# from torch.utils.data import random_split
# import torch
# import numpy as np
# import wandb
# import time
# from sklearn.metrics import r2_score
# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use('Agg')


def main(hparams):
    # model_name = f'SMTarRNA-3a6p-dim-{hparams.dim}-hlayer-{hparams.layer_hidden}-olayer-{hparams.layer_output}-' + time.strftime(
    #     "%Y%m%d_%H%M%S", time.localtime())
    # wandb.log({'checkpoint': model_name})
    dict_args = vars(hparams)
    model = MolecularGNN(**dict_args)
    # print(model.elements_dict)
    # raise ValueError
    # logger
    # wandb_logger = pl.loggers.WandbLogger(save_dir='log/3a6p')
    # callbacks
    # early stopping
    # early_stopping = pl.callbacks.early_stopping.EarlyStopping(monitor='val_loss', patience=20, mode='min')
    # checkpoint
    # checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss', mode='min', save_last=True,
    #                                                    dirpath='checkpoints', filename=model_name)
    # if hparams.checkpoint == None:
    #     trainer = Trainer.from_argparse_args(hparams)
    # else:
    #     trainer = Trainer(resume_from_checkpoint=hparams.checkpoint)
    # # trainer.tune(model)
    #
    # # Train
    # trainer.fit(model)
    # trainer.save_checkpoint(time.strftime("%Y%m%d_%H%M%S", time.localtime()) + ".ckpt")
    # trainer.test(model, dataloaders=model.val_dataloader(), verbose=False)
    # x = np.array(model.predictions['true'])
    # y = np.array(model.predictions['pred'])
    # val_r2 = r2_score(x, y)
    # # val_fig = plot_fit_confidence_bond(x, y, val_r2, annot=False)
    #
    # model.predictions = defaultdict(list)
    # trainer.test(model, dataloaders=model.train_dataloader(), verbose=False)
    # x = np.array(model.predictions['true'])
    # y = np.array(model.predictions['pred'])
    # train_r2 = r2_score(x, y)
    # train_fig = plot_fit_confidence_bond(x, y, train_r2, annot=False)
    # if True:
    #     wandb.log({'train_res': train_fig, 'val_res': val_fig})
    #     wandb.log({'val_r2': val_r2, 'train_r2': train_r2})
    #     wandb.finish()


def prepare_arg():
    parser = ArgumentParser()
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--layer_hidden",type=int, default=24)
    parser.add_argument("--layer_output",type=int, default=10)
    parser.add_argument("--batch_size",type=int, default=128)
    parser.add_argument("--data_path",type=str, default=None)
    parser.add_argument("--checkpoint",type=str, default=None)
    return parser


if __name__ == '__main__':
    project = 'SMTarRNA-3a6p-project'
    # wandb.init(project=project, dir='log/3a6p', notes='new dataset')
    # prepare args
    parser = prepare_arg()
    dataset_path = 'data/3a6p/3a6p_exhaus_96_100K.txt'
    # checkpoint = '20220618_211314.ckpt'
    # add model args
    parser = MolecularGNN.add_model_specific_args(parent_parser=parser)
    # add Trainer args
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args(['--data_path', dataset_path, '--learning_rate', '0.0005', '--gpus=1', '--max_epochs', '500'])
    # main(args)
    dict_args = vars(args)
    model = MolecularGNN(**dict_args)
    dataloder = model.train_dataloader()
    for data in dataloder:
        model.forward(data)
