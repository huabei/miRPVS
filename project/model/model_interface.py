
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
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger


class MInterface(pl.LightningModule):
    def __init__(self, model_name, loss, lr, **kargs):
        super().__init__()
        # 保存超参数
        self.save_hyperparameters()
        
        # 加载模型
        self.load_model()
        
        # 加载损失函数
        self.configure_loss()
        # Project-Specific Definitions

    @staticmethod
    def add_specific_args(parser):
        # LightningModule specific arguments
        parser.add_argument('--model_name', type=str, default='GCN')
        parser.add_argument('--loss', type=str, default='MSELoss')
        parser.add_argument('--lr', type=float, default=0.001)
        
        return parser
    
    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch: Data, batch_idx):
        p_energy = self(batch)
        y = batch.y
        loss = self.loss_function(torch.squeeze(p_energy), y)
        self.log('train_loss',prog_bar=True, batch_size=self.hparams.batch_size)
        return {'loss': loss, 'preds': p_energy, 'true': y}

    def training_epoch_end(self, training_step_outputs) -> None:
        '''计算训练集的mae'''
        self.train_pred_y = torch.stack([x['preds'] for x in training_step_outputs]).detach().cpu().numpy()
        self.train_true_y = torch.stack([x['true'] for x in training_step_outputs]).detach().cpu().numpy()
        mae_train = np.mean(np.abs(self.train_pred_y - self.train_true_y)) # 平均绝对误差
        self.log('train_mae', mae_train)

    def on_train_end(self) -> None:
        self._share_val_step(self.train_pred_y, self.train_true_y, 'train')
        self._share_val_step(self.val_pred_y, self.val_true_y, 'val')

    def validation_step(self, batch, batch_idx):
        p_energy = self(batch)
        y = batch.y
        loss = self.loss_function(torch.squeeze(p_energy), y)
        self.log('val_loss', loss, prog_bar=True, batch_size=self.hparams.batch_size)
        return {'loss': loss, 'preds': p_energy, 'true': y}

    def validation_epoch_end(self, validation_step_outputs) -> None:
        '''计算验证集的mae'''
        self.val_pred_y = torch.stack([x['preds'] for x in validation_step_outputs]).detach().cpu().numpy()
        self.val_true_y = torch.stack([x['true'] for x in validation_step_outputs]).detach().cpu().numpy()
        mae_val = np.mean(np.abs(self.val_pred_y - self.val_true_y))
        self.log('val_mae', mae_val)
        
    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        p_energy = self(batch)
        y = batch.y
        # loss = self.loss_function(torch.squeeze(p_energy), batch.y)
        return {'preds': p_energy, 'true': y}

    def test_epoch_end(self, outputs):
        # Make the Progress Bar leave there
        self.test_pred_y = torch.stack([x['preds'] for x in outputs]).detach().cpu().numpy()
        self.test_true_y = torch.stack([x['true'] for x in outputs]).detach().cpu().numpy()
        mae_test = np.mean(np.abs(self.test_pred_y - self.test_true_y))
        self.log('test_mae', mae_test)
        self._share_val_step(self.test_pred_y, self.test_true_y, 'test')

    def _share_val_step(self, pred, true, stage: str):
        r2 = r2_score(true, pred)
        fig = plot_fit_confidence_bond(true, pred, r2, annot=False)
        tensorboard_logger = self.logger.experiment
        self.log(f'{stage}_r2', r2, prog_bar=False)
        tensorboard_logger.add_figure(f'{stage}_fig', fig, global_step=self.global_step)
        if len(self.loggers) > 1:
            self.loggers[1].experiment.log({f'{stage}_fig': fig, 'global_step': self.global_step})
        
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
        Model = getattr(importlib.import_module(
            '.' + name, package=__package__), camel_name)

        # self.model = self.instancialize(Model)
        self.model = Model(**self.hparams.model)

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
