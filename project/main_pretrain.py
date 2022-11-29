
import pytorch_lightning as pl
import yaml
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc

from model.model_interface_pretrain import MInterface
from data import DInterface
from utils import load_model_path_by_args
from ray.tune.integration.pytorch_lightning import TuneReportCallback
import time
from main import parse_args

def load_callbacks(args):
    checkpoint_dirpath = f'checkpoints/{args.model_name}'
    time_ = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    callbacks = [plc.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=10
    ), plc.ModelCheckpoint(
        dirpath=checkpoint_dirpath,
        monitor='val_loss',
        filename='best-{epoch:02d}-{train_loss:.2f}-{val_loss:.2f}'+time_,
        save_top_k=1,
        mode='min',
        save_last=True
    )]

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))
    if args.tune:
        callbacks.append(TuneReportCallback(
            metrics={'loss': 'val_loss'},
            on='validation_end'))
    return callbacks


def load_logger(args):
    tb_logger = TensorBoardLogger(save_dir=args.log_dir, default_hp_metric=False, name='tensorboard', comment=args.comment)
    return [tb_logger]


def main(args):
    # pl.seed_everything(args.seed)
    load_path = load_model_path_by_args(args)
    data_module = DInterface(**vars(args))

    if load_path is None:
        model = MInterface(**vars(args))
    else:
        model = MInterface(**vars(args))
        args.resume_from_checkpoint = load_path

    args.callbacks = load_callbacks(args)
    args.logger = load_logger(args)
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, data_module)
    trainer.test(model, data_module)


if __name__ == '__main__':
    parser = parse_args()
    parser.add_argument('--num_heads', type=int, default=1)
    # yaml 文件

    with open('config/default_3a6p_molecular_e3nn_transformer_pretrain.yaml', 'r') as f:
        default_arg = yaml.safe_load(f)

    # Reset Some Default Trainer Arguments' Default Values
    parser.set_defaults(**default_arg)
    args, unknown = parser.parse_known_args()

    main(args)
