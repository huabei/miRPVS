from typing import Any, Callable, List
from pytorch_lightning.loggers import TensorBoardLogger
from ray import air, tune
from ray.tune import CLIReporter, TuneConfig
from ray.tune.schedulers import ASHAScheduler
import os
import time
# from config.tune_config import config_tune
import logging
from jsonargparse import Namespace, ArgumentParser, ActionConfigFile, util, capture_parser
import ray
# ray.init(num_cpus=4, num_gpus=1, include_dashboard=False, ignore_reinit_error=True)
# raise Exception('stop')
os.environ['WANDB_MODE'] = 'offline'
from cli_main import cli_main
from models import PLBaseModel
from datamodule import DInterface

# def cli_main(args):
#     with open('/home/huabei/tmp/test.txt', 'a') as f:
#         f.write(str(args))
#     tune.report(val_loss=0.5)
    # print(args)

class TuneParameter:
    def __init__(self, algorithm, **kwargs) -> None:
        self.sample_algorithm = {'uniform': tune.uniform,
                    'loguniform': tune.loguniform,
                    'randint': tune.randint,
                    'choice': tune.choice,
                    'grid_search': tune.grid_search}
        self.algorithm = self.sample_algorithm[algorithm](**kwargs)
    def __call__(self) -> Any:
        return self.algorithm


class RayTuneCLI():
    '''
    模仿pytorchlightningCLI使用命令行和配置文件进行超参数搜索配置的类, 核心类为tune.Tuner
    '''
    def __init__(self) -> None:
        self.parser = self.setup_parser()
        self.config = self.parse_arguments()
        
        # self.parser.instantiate_classes(self.config)
        # print(self.config)
        tuner = self.config.tuner.init
        results = tuner.fit()
        
    def setup_parser(self):
        parser = ArgumentParser()
        parser.add_argument('-c', '--config', action=ActionConfigFile, help=util.default_config_option_help)
        parser.add_class_arguments(tune.Tuner, 'tuner.init', fail_untyped=False)
        parser.add_argument('--tuner.resources_per_trial', type=dict, default={'cpu': 1, 'gpu': 1})
        parser.add_argument('--tuner.fix_config', type=Namespace, default={})
        return parser

    def parse_arguments(self):
        config = self.parser.parse_args()
        # 构造trainable函数
        @trainable_decorator
        def trainable(args):
            cli_main(args)
        # 构造param space
        config.tuner.init.param_space = {k: TuneParameter(**v) for k, v in config.tuner.init.param_space.items()}
        # 传入固定的参数
        fn = tune.with_parameters(trainable, fixed_config=config.tuner.fix_config)
        # 设置每个试验的资源
        config.tuner.init.trainable = tune.with_resources(fn, config.tuner.resources_per_trial)
        # 实例化tune config
        config.tuner.init.tune_config = TuneConfig(**config.tuner.init.tune_config)
        # 实例化tuner
        config.tuner.init = tune.Tuner(**config.tuner.init)
        return config


def to_absolute_path(path: str) -> str:
    '''将相对路径转换为绝对路径'''
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    return path


def trainable_decorator(func):
    '''装饰ray tune的trainable函数，使其能够接受ray tune的config参数'''
    def wrapper(config: Namespace, fixed_config: Namespace):
        # 加入可变的已采样的超参数
        fixed_config.update(config)
        config['tune']=True
        func(args=fixed_config)
    return wrapper


def load_scheduler(scheduler: str=None):
    '''加载超参数搜索算法'''
    if scheduler == 'ASHA':
        logging.info('Using ASHA scheduler')
        scheduler = ASHAScheduler(
            max_t=cfg.trainer.max_epochs,  # max epoch
            grace_period=cfg.grace_period,  # 延缓周期r, 规定起始epoch数目为{r * η^s}个，用于避免过早停止
            reduction_factor=cfg.reduction_factor,  # 降低因子η，每个周期保留{n/η}个trial
            brackets=1,  # 用于搜索的bracket数目s
        )
    elif scheduler is None:
        logging.info('Using default scheduler')
        scheduler = None
    else:
        raise ValueError(f'Unknown scheduler {cfg.scheduler}')
    return scheduler


def main_tune(param_space: Namespace,
              fixed_cfg: Namespace,
              ):
    '''主函数，用于进行超参数搜索
    '''
    # 超参数搜索算法
    scheduler = load_scheduler(fixed_cfg.scheduler)
    # 命令行输出内容
    reporter = CLIReporter(
        parameter_columns=None, # 默认显示所有参数
        metric_columns=["loss", "training_iteration"],
        max_report_frequency=20)
    
    @trainable_decorator
    def trainable(args, parser_kwargs={'default_config_files': ['project/config/pl_cli_config.yaml']}):
        cli_main(args, parser_kwargs=parser_kwargs)

    # 传入固定的参数
    train_fn_with_parameters = tune.with_parameters(trainable, fixed_config=fixed_cfg)
    # 设置每个试验的资源
    resources_per_trial = {"cpu": 4, "gpu": fixed_cfg.gpus_per_trial}
    # 设置tuner
    tuner = tune.Tuner(
        tune.with_resources(train_fn_with_parameters, resources_per_trial),
        tune_config=TuneConfig(metric="loss", mode="min", scheduler=scheduler, num_samples=fixed_cfg.num_samples),
        run_config=air.RunConfig(name='tune', progress_reporter=reporter),
        param_space=tune_cfg,
    )
    results = tuner.fit()
    df_name = f'{fixed_cfg.log_dir}/{fixed_cfg.project}_{fixed_cfg.current_time}-tune.csv'
    results.get_dataframe().to_csv(df_name)
    print("Best hyperparameters found were: ", results.get_best_result().config)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # parser = ArgumentParser()
    # parser = capture_parser(
    #     MyLightningCLI(datamodule_class=DInterface,
    #                    subclass_mode_model=PLBaseModel,
    #                    seed_everything_default=1234,
    #                    auto_configure_optimizers=False,
    #                    args=None,
    #                    parser_kwargs={'default_config_files': ['project/config/pl_cli_config.yaml']}
    #                    )
    # )
    # parser.add_class_arguments(tune.Tuner, 'Tuner', fail_untyped=False)
    RayTuneCLI()
    # a = parser.parse_args()
    # print(a)