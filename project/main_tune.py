from pytorch_lightning.loggers import TensorBoardLogger
from ray import air, tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from main import main
import yaml
import os
import time
from ml_collections import ConfigDict
from config.tune_config import config_tune
import logging
from argparse import ArgumentParser
import ray
# ray.init(num_cpus=4, num_gpus=1, include_dashboard=False, ignore_reinit_error=True)
# raise Exception('stop')
os.environ['WANDB_MODE'] = 'offline'

def to_absolute_path(path: str) -> str:
    '''将相对路径转换为绝对路径'''
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    return path


def trainable_decorator(func):
    '''装饰ray tune的trainable函数，使其能够接受ray tune的config参数'''
    def wrapper(config: dict, fixed_config: ConfigDict):
        # 加入可变的已采样的超参数
        fixed_config.update_from_flattened_dict(config)
        # 确保进行超参数搜索配置
        fixed_config.tune = True
        main(fixed_config)
    return wrapper
        

def load_scheduler(cfg: ConfigDict):
    '''加载超参数搜索算法'''
    if cfg.scheduler == 'ASHA':
        logging.info('Using ASHA scheduler')
        scheduler = ASHAScheduler(
            max_t=cfg.trainer.max_epochs,  # max epoch
            grace_period=cfg.grace_period,  # 延缓周期r, 规定起始epoch数目为{r * η^s}个，用于避免过早停止
            reduction_factor=cfg.reduction_factor,  # 降低因子η，每个周期保留{n/η}个trial
            brackets=1,  # 用于搜索的bracket数目s
        )
    elif cfg.scheduler is None:
        logging.info('Using default scheduler')
        scheduler = None
    else:
        raise ValueError(f'Unknown scheduler {cfg.scheduler}')
    return scheduler
    

def main_tune(trainable, cfg_path: str):
    '''主函数，用于进行超参数搜索
    Args:
        trainable: ray tune的trainable函数
        cfg_path: 配置文件路径
        num_samples: 总试验数
        gpus_per_trial: 每个试验使用的gpu数
    '''
    # 载入配置文件
    logging.info(f'Loading config from {cfg_path}')
    tune_cfg, fixed_cfg = config_tune(cfg_path)
    tune_cfg: dict
    fixed_cfg: ConfigDict
    # wandb程序会记录这次运行的配置文件
    fixed_cfg.config_file = to_absolute_path(cfg_path)
    # 设置日志文件夹
    fixed_cfg.log_dir = f'./log/tune/{fixed_cfg.pl_module.model_name}/{fixed_cfg.pl_data_module.dataset}/{fixed_cfg.current_time+"_".join(fixed_cfg.comment.split())}'
    if not os.path.exists(fixed_cfg.log_dir):
        os.makedirs(fixed_cfg.log_dir)
    # 将路径转换为绝对路径，相对路径可能会导致子进程无法找到文件
    fixed_cfg.log_dir = to_absolute_path(fixed_cfg.log_dir) # 日志文件夹
    fixed_cfg.pl_data_module.data_dir = to_absolute_path(fixed_cfg.pl_data_module.data_dir) # 数据文件夹
    # 超参数搜索算法
    scheduler = load_scheduler(fixed_cfg)
    # 命令行输出内容
    reporter = CLIReporter(
        parameter_columns=None, # 默认显示所有参数
        metric_columns=["loss", "training_iteration"],
        max_report_frequency=20)
    
    # 传入固定的参数
    train_fn_with_parameters = tune.with_parameters(trainable, fixed_config=fixed_cfg)
    # 设置每个试验的资源
    resources_per_trial = {"cpu": 4, "gpu": fixed_cfg.gpus_per_trial}
    # 设置tuner
    tuner = tune.Tuner(
        tune.with_resources(train_fn_with_parameters, resources_per_trial),
        tune_config=tune.TuneConfig(metric="loss", mode="min", scheduler=scheduler, num_samples=fixed_cfg.num_samples),
        run_config=air.RunConfig(name='tune', progress_reporter=reporter),
        param_space=tune_cfg,
    )
    results = tuner.fit()
    df_name = f'{fixed_cfg.log_dir}/{fixed_cfg.project}_{fixed_cfg.current_time}-tune.csv'
    results.get_dataframe().to_csv(df_name)
    print("Best hyperparameters found were: ", results.get_best_result().config)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c','--cfg_path', type=str, help='config file path')
    config_path = parser.parse_args().cfg_path
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    @trainable_decorator
    def main_trainable(args):
        main(args)

    main_tune(main_trainable, cfg_path=config_path)
