import argparse as ap
import collections as col
from functools import partial

import pytorch_lightning as pl
import torch
from torch_scatter import scatter_mean

from e3nn.kernel import Kernel
from e3nn.linear import Linear
from e3nn import o3
from e3nn.non_linearities.norm import Norm
# from e3nn.non_linearities.nonlin import Nonlinearity
from e3nn.point.message_passing import Convolution
from e3nn.radial import GaussianRadialModel
import wandb


class E_Molecular_NN(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ap.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=1e-2)
        return parser

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)






