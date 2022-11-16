
import pytorch_lightning as pl
import yaml
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc

from model.model_interface_hpc import MInterface
from data import DInterface
from utils import load_model_path_by_args
from ray.tune.integration.pytorch_lightning import TuneReportCallback

def load_callbacks(args):
    callbacks = [plc.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=20
    ), plc.ModelCheckpoint(
        dirpath='checkpoints/gcn',
        monitor='val_loss',
        filename='best-{epoch:02d}-{train_loss:.2f}-{val_loss:.2f}',
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
    pl.seed_everything(args.seed)
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
    parser = ArgumentParser()
    # Project info
    parser.add_argument('--project', type=str)
    parser.add_argument('--comment', type=str)

    # Basic Training Control
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)

    # LR Scheduler
    parser.add_argument('--lr_scheduler', choices=['step', 'cosine'], type=str, default='cosine')
    parser.add_argument('--lr_decay_steps', default=5, type=int)
    parser.add_argument('--lr_decay_rate', default=0.99, type=float)
    parser.add_argument('--lr_decay_min_lr', default=1e-5, type=float)

    # Restart Control
    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--load_dir', default=None, type=str)
    parser.add_argument('--load_ver', default=None, type=str)
    parser.add_argument('--load_v_num', default=None, type=int)

    # Training Info
    parser.add_argument('--dataset', default='zinc_complex3a6p_data_e3nn', type=str)
    parser.add_argument('--data_dir', default='data/3a6p/zinc_drug_like_100k/3a6p_pocket5_202020', type=str)
    parser.add_argument('--log_dir', default='log/e3nn_transformer', type=str)
    parser.add_argument('--model_name', default='molecular_e3nn_transformer', type=str)
    parser.add_argument('--loss', default='mse', type=str, choices=['l1, mse', 'smooth_l1'])
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--no_augment', action='store_true')

    # Model Hyperparameters
    parser.add_argument('--in_channels', default=10, type=int)
    parser.add_argument('--hidden_channels', default=256, type=int)
    parser.add_argument('--out_channels', default=1, type=int)
    parser.add_argument('--hidden_layers', default=16, type=int)
    parser.add_argument('--out_layers', default=6, type=int)

    # Add pytorch lightning's args to parser as a group.
    parser = pl.Trainer.add_argparse_args(parser)

    # yaml 文件

    with open('config/default_3a6p_molecular_e3nn_transformer.yaml', 'r') as f:
        default_arg = yaml.safe_load(f)

    # Reset Some Default Trainer Arguments' Default Values
    parser.set_defaults(**default_arg)

    args, unknown = parser.parse_known_args()

    main(args)
