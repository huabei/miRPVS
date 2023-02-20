'''训练过程中的一些工具函数'''
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import pytorch_lightning.callbacks as plc

def load_callbacks(args):
    # 早停和保存最优模型
    callbacks = [plc.EarlyStopping(
        monitor=args.trainer.early_stop.monitor,
        mode='min',
        patience=args.trainer.early_stop.patience,
    ), plc.ModelCheckpoint(
        dirpath=f'checkpoints/{args.pl_module.model_name}/',
        monitor=args.trainer.early_stop.monitor,
        filename=args.current_time + 'best-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
        save_last=False
    )]
    # 学习率调整
    if args.pl_module.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))
    # ray tune
    if args.tune:
        from ray.tune.integration.pytorch_lightning import TuneReportCallback        
        callbacks.append(TuneReportCallback(
            metrics={'loss': 'val_loss'},
            on='validation_end'))
    return callbacks


def load_logger(args):
    '''以Tesorboard为主，Wandb为辅，记录超参数和训练过程'''
    tb_logger = TensorBoardLogger(save_dir=args.log_dir,
                                  default_hp_metric=False, name='tensorboard', comment=args.comment)
    if args.wandb:
        wandb_logger = WandbLogger(project=args.project,
                                   dir=args.log_dir, notes=args.comment)
        return [wandb_logger, tb_logger]
    return [tb_logger]

