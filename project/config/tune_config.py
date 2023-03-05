from ml_collections import ConfigDict, config_dict
from time import strftime, localtime
from ray import tune
import yaml
from .config import set_default_config

def config_tune(yaml_path: str) -> tuple:
    '''根据yaml文件，设定ray tune超参数搜索的参数，包括实验超参数，冗余超参数，固定超参数'''
    def set_tune_hyperparameter(hyperparameter: dict):
        '''将yaml文件中的超参数转换为ray tune的超参数'''
        sample_algorithm = {'uniform': tune.uniform,
                            'loguniform': tune.loguniform,
                            'randint': tune.randint,
                            'choice': tune.choice,
                            'grid_search': tune.grid_search}
        for key, value in hyperparameter.items():
            assert isinstance(value, dict)
            hyperparameter[key] = sample_algorithm[value.pop('type')](**value)
        return hyperparameter

    with open(yaml_path, 'r') as f:
        config_d = yaml.safe_load(f)
    
    tune_config = dict()
    fixed_config = config_dict.ConfigDict(type_safe=False)
    set_default_config(fixed_config)
    
    # 设置实验超参数
    scientific_hyperparameter = config_d['scientific_hyperparameter']
    if scientific_hyperparameter is not None:
        hyperparameter = set_tune_hyperparameter(scientific_hyperparameter)
        tune_config.update(hyperparameter)
    # 设置冗余超参数
    nuisance_hyperparameter = config_d['nuisance_hyperparameter']
    if nuisance_hyperparameter is not None:
        hyperparameter = set_tune_hyperparameter(nuisance_hyperparameter)
        tune_config.update(hyperparameter)
    
    # 设置固定超参数
    fixed_hyperparameter = config_d['fixed_hyperparameter']
    if fixed_hyperparameter is not None:
        fixed_config.update(fixed_hyperparameter)
    return tune_config, fixed_config

if __name__ == '__main__':
    config = config_tune('config/tune_cfg_temp.yaml')
    print(config)