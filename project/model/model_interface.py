
import inspect
import torch
import numpy as np
import importlib
from torch import nn
from sklearn.metrics import r2_score
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
# from ..utils.package import plot_fit_confidence_bond
import pytorch_lightning as pl
from torch_geometric.data import Data
from collections import defaultdict
import wandb
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger


class MInterface(pl.LightningModule):
    def __init__(self, model_name, loss, lr, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()
        # Project-Specific Definitions

    def forward(self, batch):
        return self.model(batch)

    def on_train_epoch_start(self) -> None:
        self.train_predictions = defaultdict(list)

    def training_step(self, batch: Data, batch_idx):
        # x, edge_index, edge_weight, t_energy = batch.x, batch.edge_index, batch.edge_attr, batch.y
        p_energy = self(batch)
        self.train_predictions['prediction'].extend(p_energy.detach().cpu().numpy())
        self.train_predictions['true'].extend(batch.y.detach().cpu().numpy())
        loss = self.loss_function(torch.squeeze(p_energy), batch.y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        return loss

    def on_train_end(self) -> None:
        x_train = np.array(self.train_predictions['true'])
        y_train = np.array(self.train_predictions['prediction'])
        x_val = np.array(self.val_predictions['true'])
        y_val = np.array(self.val_predictions['prediction'])
        train_r2 = r2_score(x_train, y_train)
        val_r2 = r2_score(x_val, y_val)
        train_fig = plot_fit_confidence_bond(x_train, y_train, train_r2, annot=False)
        val_fig = plot_fit_confidence_bond(x_val, y_val, val_r2, annot=False)
        self.train_r2 = train_r2
        self.val_r2 = val_r2
        for logger in self.loggers:
            logger.log_metrics({'train_r2': train_r2, 'val_r2': val_r2})
            if type(logger) is TensorBoardLogger:
                logger.experiment.add_figure('train_fig', train_fig)
                logger.experiment.add_figure('val_fig', val_fig)
        wandb.log({'train_fig': train_fig, 'val_fig': val_fig})

    def on_validation_epoch_start(self) -> None:
        self.val_predictions = defaultdict(list)

    def validation_step(self, batch, batch_idx):
        # x, edge_index, edge_weight, t_energy = batch.x, batch.edge_index, batch.edge_attr, batch.y
        p_energy = self(batch)
        self.val_predictions['prediction'].extend(p_energy.cpu().numpy())
        self.val_predictions['true'].extend(batch.y.cpu().numpy())
        loss = self.loss_function(torch.squeeze(p_energy), batch.y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        return {'loss': loss, 'preds': p_energy}

    def on_test_start(self) -> None:
        self.test_predictions = defaultdict(list)

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        p_energy = self(batch)
        self.test_predictions['prediction'].extend(p_energy.cpu().numpy())
        self.test_predictions['true'].extend(batch.y.cpu().numpy())
        # loss = self.loss_function(torch.squeeze(p_energy), batch.y)
        return self.loss_function(torch.squeeze(p_energy), batch.y)

    def on_test_end(self):
        # Make the Progress Bar leave there
        x = np.array(self.test_predictions['true'])
        y = np.array(self.test_predictions['prediction'])
        test_r2 = r2_score(x, y)
        test_fig = plot_fit_confidence_bond(x, y, test_r2, annot=False)
        for logger in self.loggers:
            logger.log_metrics({'test_r2': test_r2})
            if type(logger) is TensorBoardLogger:
                logger.experiment.add_figure('test_fig', test_fig)
                logger.log_hyperparams(self.hparams,
                                       {'test_r2': test_r2, 'train_r2': self.train_r2, 'val_r2': self.val_r2})
        wandb.log({'test_fig': test_fig})

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    def configure_loss(self):
        loss = self.hparams.loss.lower()
        if loss == 'mse':
            self.loss_function = F.mse_loss
        elif loss == 'l1':
            self.loss_function = F.l1_loss
        elif loss == 'smooth_l1':
            self.loss_function = F.smooth_l1_loss
        else:
            raise ValueError("Invalid Loss Type!")

    def load_model(self):
        name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.' + name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getfullargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)


def plot_fit_confidence_bond(x, y, r2, annot=True):
    # fit a linear curve an estimate its y-values and their error.
    a, b = np.polyfit(x, y, deg=1)
    y_est = a * x + b
    # y_err = x.std() * np.sqrt(1 / len(x) +
    #                           (x - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))

    fig, ax = plt.subplots()
    ax.plot([-20, 0], [-20, 0], '-')
    ax.plot(x, y_est, '-')
    # ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)
    ax.plot(x, y, 'o', color='tab:brown')
    if annot:
        num = 0
        for x_i, y_i in zip(x, y):
            ax.annotate(str(num), (x_i, y_i))
            # if y_i > -3:
            #     print(num)
            num += 1
    ax.set_xlabel('True Energy(Kcal/mol)')
    ax.set_ylabel('Predict Energy(Kcal/mol)')
    # ax.text(0.1, 0.5, 'r2:  ' + str(r2))
    ax.text(0.4, 0.9,
            'r2:  ' + str(r2), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
            fontsize=12)
    return fig
