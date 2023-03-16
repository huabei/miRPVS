from ml_collections import ConfigDict, config_dict
from time import strftime, localtime
# 获取当前时间
CURRENT_TIME = strftime("%Y-%m-%d-%H-%M-%S", localtime())


def set_default_config(config: ConfigDict):
    '''
    在程序当中用到的配置参数，如果不存在将会报错。各个模型自己的参数可以后续再加入。
    return:
            config: ConfigDict
                tune: bool
                seed: int
                    trainer: ConfigDict
                    pl_module: ConfigDict
                        model: ConfigDict
    '''
    config.project = 'project'
    config.comment = 'comment'
    config.group = None
    config.tune = False
    config.seed = 1234
    config.current_time = CURRENT_TIME
    config.wandb = True
    config.early_stop = config_dict.ConfigDict()
    config.early_stop.monitor = 'val_loss'
    config.early_stop.patience = 10
    log_dir = config_dict.FieldReference('log')
    config.log_dir = log_dir
    batch_size = config_dict.FieldReference(128)
    
    def trainer_cfg():
        '''pl.Trainer的参数,手动调用时的参数
        '''
        config.trainer = ConfigDict()
        config.trainer.default_root_dir = log_dir
        config.trainer.fast_dev_run = False
        config.trainer.max_epochs = 100
        config.trainer.accumulate_grad_batches = 1
        config.trainer.resume_from_checkpoint = None
        config.trainer.deterministic = False
        config.trainer.auto_lr_find = False
        config.trainer.auto_scale_batch_size = False
        config.trainer.enable_progress_bar = True
        config.trainer.enable_model_summary = True
        config.trainer.enable_checkpointing = True
        config.trainer.accelerator = 'gpu'
        config.trainer.devices = 1
        
    def pl_module_cfg():
        '''pl.LightningModule的参数,手动调用时的参数'''
        config.pl_module = ConfigDict(type_safe=False)
        config.pl_module.batch_size = batch_size
        config.pl_module.weight_decay = 0
        config.pl_module.loss = 'smooth_l1'
        config.pl_module.model_name = 'egnn'
        config.pl_module.lr_scheduler = 'cosine'
        config.pl_module.lr = 5e-4
        config.pl_module.lr_t_max = 100
        config.pl_module.lr_decay_steps = 100
        config.pl_module.lr_decay_rate = 0.8
        config.pl_module.lr_decay_min_lr = 0
        
        # model cfg
        config.pl_module.model = ConfigDict(type_safe=False)
    
    def pl_data_module_cfg():
        '''pl.LightningDataModule的参数,手动调用时的参数'''
        config.pl_data_module = ConfigDict(type_safe=False)
        config.pl_data_module.dataset = 'zinc_complex3a6p_data'
        config.pl_data_module.data_dir = 'data/3a6p/zinc_drug_like_100k/3a6p_pocket5_202020'
        config.pl_data_module.batch_size = batch_size
        config.pl_data_module.num_workers = 0
    
    trainer_cfg()
    pl_module_cfg()
    pl_data_module_cfg()