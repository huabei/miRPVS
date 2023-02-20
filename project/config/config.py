from ml_collections import ConfigDict, config_dict
from time import strftime, localtime
# 获取当前时间
CURRENT_TIME = strftime("%Y-%m-%d-%H-%M-%S", localtime())


def set_default_config(config: ConfigDict):
    '''
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
    config.tune = False
    config.seed = 1234
    config.current_time = CURRENT_TIME
    config.wandb = True
    config.early_stop = config_dict.ConfigDict()
    config.early_stop.monitor = 'val_loss'
    config.early_stop.patience = 10
    log_dir = config_dict.FieldReference('log')
    config.log_dir = log_dir
    def trainer_cfg():
        '''pl.Trainer的参数,手动调用时的参数
        '''
        config.trainer = ConfigDict()
        config.trainer.default_root_dir = log_dir
        config.trainer.fast_dev_run = False
        config.trainer.max_epochs = 100
        config.trainer.accumulate_grad_batches = 1
        config.trainer.resume_from_checkpoint = None
        config.trainer.deterministic = True
        config.trainer.auto_lr_find = False
        config.trainer.auto_scale_batch_size = False
        config.trainer.accelerator = 'gpu'
        config.trainer.devices = 1
        
    def pl_module_cfg():
        '''pl.LightningModule的参数,手动调用时的参数'''
        config.pl_module = ConfigDict()
        
        config.pl_module.weight_decay = 0
        config.pl_module.loss = 'mse'
        config.pl_module.model_name = 'gcn'
        config.pl_module.lr_scheduler = 'step'
        config.pl_module.lr = 5e-4
        config.pl_module.lr_decay_steps = 100
        config.pl_module.lr_decay_rate = 0.8
        config.pl_module.lr_decay_min_lr = 1e-6
        
        # model cfg
        if config.pl_module.model_name == 'molecular_gnn':
            config.pl_module.model = ConfigDict()
            config.pl_module.model.input_dim = 96
            config.pl_module.model.hidden_dim = 128
            config.pl_module.model.output_dim = 1
            config.pl_module.model.hidden_layers = 3
            config.pl_module.model.out_layers = 1
    
    def pl_data_module_cfg():
        '''pl.LightningDataModule的参数,手动调用时的参数'''
        config.pl_data_module = ConfigDict()
        config.pl_data_module.dataset = 'zinc_complex3a6p_data'
        config.pl_data_module.data_dir = 'data/3a6p/zinc_drug_like_100k/3a6p_pocket5_202020'
        config.pl_data_module.batch_size = 128
        config.pl_data_module.num_workers = 8
    
    trainer_cfg()
    pl_module_cfg()
    pl_data_module_cfg()
    # 设置日志文件夹
    config.log_dir = f'./log/{config.pl_module.model_name}/{config.pl_data_module.dataset}'