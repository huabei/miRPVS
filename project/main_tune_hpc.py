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
    num_samples = 256
    gpu_per_trial = 0.5
    project = '3a6p_smiles_radius0_molecule_gnn_tune'
    comment = 'mature_tune'
    constant_dir = 'config/constant_config_hpc.yaml'
    config = {
        "lr_decay_rate": tune.uniform(0.8, 1.0),
        "lr": tune.loguniform(1e-4, 1e-2),
        "hidden_channels": tune.randint(64, 512),
        "out_layers": tune.randint(1, 10),
        "hidden_layers": tune.randint(1, 16),
        "batch_size": tune.randint(32, 256),
        "weight_decay": tune.loguniform(1e-6, 1e-4),
        "lr_decay_min_lr": tune.loguniform(1e-6, 1e-4),
        "lr_scheduler": tune.choice(['cosine', 'step']),
        # "heads": tune.randint(1, 16)
    }
    main_tune(trainable, config, num_samples, gpu_per_trial, project, comment, constant_dir)
