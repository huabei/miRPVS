
import pytorch_lightning as pl
import yaml
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from model import MInterface
from data import DInterface
import ml_collections as mlc
from train_utils import load_logger, load_callbacks
import logging
import os

def main(args: mlc.ConfigDict):
    logging.info('seed everything')
    pl.seed_everything(args.seed)
    if args.wandb:
        logging.info('Using wandb')
        import wandb
        wandb.login(key='local-8fe6e6b5840c4c05aaaf6aac5ca8c1fb58abbd1f', host='http://localhost:8080')
        wandb.init(project=args.project, save_code=False, dir=args.log_dir, reinit=True)
        wandb.config.update(args)
    logging.info('Loading data and model')
    data_module = DInterface(**args.pl_data_module)
    model = MInterface(**args.pl_module)

    logging.info('loading callbacks and logger')
    args.trainer.callbacks = load_callbacks(args)
    args.trainer.logger = load_logger(args)
    
    logging.info('creating trainer')
    trainer = Trainer(**args.trainer)
    logging.info('start training')
    trainer.fit(model, data_module)
    
    logging.info('start testing')
    trainer.test(model, data_module)


if __name__ == '__main__':
    from config.config import set_default_config
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    cfg = mlc.ConfigDict()
    set_default_config(cfg)
    '''
    return:
            config: ConfigDict
                tune: bool
                seed: int
                    trainer: ConfigDict
                    pl_module: ConfigDict
                        model: ConfigDict
    '''
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/default_3a6p_molecular_gin.yaml')
    args = parser.parse_args()
    cfg.config_file = args.config
    
    # 使用yaml文件中的参数更新默认参数
    # yaml 文件
    with open(args.config, 'r') as f:
        default_arg = yaml.safe_load(f)

    # 重置一些默认参数
    cfg.update(default_arg)
    # 设置日志文件夹
    cfg.log_dir = f'./log/{cfg.pl_module.model_name}/{cfg.pl_data_module.dataset}'
    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)

    main(cfg)
