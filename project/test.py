from jsonargparse import ArgumentParser

parser_subcomm1 = ArgumentParser()

parser_subcomm1.add_argument('--num_samples', type=int, default=1, help='Number of trials')
parser_subcomm2 = ArgumentParser()
parser_subcomm2.add_argument('--gpus_per_trial', type=int, default=1, help='Number of gpus per trial')
parser = ArgumentParser()
parser.add_argument('--cfg_path', type=str, help='config file path')
subcommands = parser.add_subcommands()

subcommands.add_subcommand('subcomm1', parser_subcomm1)

subcommands.add_subcommand('subcomm2', parser_subcomm2)

arg = parser.parse_args(['subcomm1', '--num_samples', '2'])

