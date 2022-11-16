from pytorch_lightning.loggers import TensorBoardLogger
from ray import air, tune
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from main_hpc import main
import yaml
import os
from main_tune import decorator_trainable, main_tune
# import ray
# ray.init(num_cpus=32)

if __name__ == '__main__':
    @decorator_trainable
    def trainable(args):
        main(args)
    num_samples = 50
    gpu_per_trial = 0.5
    project = 'molecular_gnn_tune'
    comment = 'first tune'
    constant_dir = 'config/constant_config_hpc.yaml'
    config = {
        "lr_decay_rate": tune.uniform(0.8, 1.0),
        "lr": tune.loguniform(1e-4, 1e-1),
        "hidden_channel": tune.randint(64, 512),
        "out_layers": tune.randint(1, 10),
        "hidden_layers": tune.randint(1, 10),
        "batch_size": tune.randint(32, 256),
        "weight_decay": tune.loguniform(1e-5, 1e-3),
    }
    main_tune(trainable, config, num_samples, gpu_per_trial, project, comment, constant_dir)
