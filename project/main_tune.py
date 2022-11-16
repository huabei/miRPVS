from pytorch_lightning.loggers import TensorBoardLogger
from ray import air, tune
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from main import main, parse_args
import yaml
import os


def main_trainable(config: dict, constant: dict, project: str, comment: str):
    constant['project'] = project
    constant['comment'] = comment
    parser = parse_args()
    # tune config
    parser.set_defaults(**config)
    # const config
    parser.set_defaults(**constant)
    args, unknown = parser.parse_known_args()
    print(os.getcwd())
    main(args)


def main_tune(config: dict, gpus_per_trial: int, project: str, comment: str):

    num_epochs = 501
    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["lr_decay_rate", "lr", "hidden_channel", "out_layers", "hidden_layers", "batch_size"],
        metric_columns=["loss", "training_iteration"])

    gpus_per_trial = 1
    # yaml 文件
    with open('config/constant_config.yaml', 'r') as f:
        default_arg = yaml.safe_load(f)
    train_fn_with_parameters = tune.with_parameters(main_tune, constant=default_arg, project=project, comment=comment)
    resources_per_trial = {"cpu": 1, "gpu": gpus_per_trial}
    tuner = tune.Tuner(
        tune.with_resources(train_fn_with_parameters, resources_per_trial),
        tune_config=tune.TuneConfig(metric="loss", mode="min", scheduler=scheduler, num_samples=num_samples),
        run_config=air.RunConfig(name='tune', progress_reporter=reporter),
        param_space=config,
    )
    results = tuner.fit()
    print("Best hyperparameters found were: ", results.get_best_result().config)


if __name__ == '__main__':
    num_samples = 10
    gpu_per_trial = 1
    project = 'tune_test'
    comment = 'test'
    config = {
        "lr_decay_rate": tune.uniform(0.8, 1.0),
        "lr": tune.loguniform(1e-4, 1e-1),
        "hidden_channel": tune.randint(64, 512),
        "out_layers": tune.randint(1, 10),
        "hidden_layers": tune.randint(1, 10),
        "batch_size": tune.randint(32, 256),
    }
    main_tune(config, gpu_per_trial, project, comment)
