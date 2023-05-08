from typing import Any

import numpy as np
import torch
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from torch_geometric.data import Data
from torchmetrics import MaxMetric, MeanMetric, PearsonCorrCoef, R2Score

from src.utils.plot_fig import plot_fig


class SMTARRNAModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = torch.nn.SmoothL1Loss()

        # pearson correlation coefficient and r2 score
        self.train_pearson = PearsonCorrCoef()
        self.val_pearson = PearsonCorrCoef()
        self.test_pearson = PearsonCorrCoef()

        self.train_r2 = R2Score()
        self.val_r2 = R2Score()
        self.test_r2 = R2Score()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_pearson_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_pearson.reset()
        self.val_r2.reset()
        self.val_pearson_best.reset()

    def on_train_epoch_start(self) -> None:
        self.train_results = {"preds": [], "targets": []}

    def model_step(self, batch: Data):
        y = batch.y
        preds = self.forward(batch)
        loss = self.criterion(preds, y)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)
        self.train_results["preds"].append(preds[:, 0])
        self.train_results["targets"].append(targets[:, 0])
        # update and log metrics
        self.train_loss(loss)

        self.train_pearson(preds[:, 0], targets[:, 0])
        self.train_r2(preds[:, 0], targets[:, 0])
        self.log("lr", self.optimizers().optimizer.param_groups[0]["lr"])
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/pearson", self.train_pearson, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/r2", self.train_r2, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_start(self) -> None:
        self.val_results = {"preds": [], "targets": []}
        return super().on_validation_epoch_start()

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)
        # save results for visualization
        self.val_results["preds"].append(preds[:, 0])
        self.val_results["targets"].append(targets[:, 0])
        # update and log metrics
        self.val_loss(loss)
        self.val_pearson(preds[:, 0], targets[:, 0])
        self.val_r2(preds[:, 0], targets[:, 0])
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/pearson", self.val_pearson, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/r2", self.val_r2, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        pearson = self.val_pearson.compute()  # get current val acc
        self.val_pearson_best(pearson)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            "val/pearson_best", self.val_pearson_best.compute(), sync_dist=True, prog_bar=True
        )

    def on_train_end(self) -> None:
        # 绘制最终结果
        train_pred_y = (
            torch.concat([x for x in self.train_results["preds"]]).detach().cpu().numpy()
        )
        train_targets_y = (
            torch.concat([x for x in self.train_results["targets"]]).detach().cpu().numpy()
        )
        val_pred_y = torch.concat([x for x in self.val_results["preds"]]).detach().cpu().numpy()
        val_targets_y = (
            torch.concat([x for x in self.val_results["targets"]]).detach().cpu().numpy()
        )

        train_fig = plot_fig(train_pred_y, train_targets_y)
        val_fig = plot_fig(val_pred_y, val_targets_y)
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                import wandb

                logger.experiment.log(
                    {"train/fig": wandb.Image(train_fig), "val/fig": wandb.Image(val_fig)}
                )
                break
        return super().on_train_end()

    def on_test_epoch_start(self) -> None:
        self.test_results = {"preds": [], "targets": []}
        return super().on_test_epoch_start()

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)
        # save results for visualization
        self.test_results["preds"].append(preds[:, 0])
        self.test_results["targets"].append(targets[:, 0])
        # update and log metrics
        self.test_loss(loss)
        self.test_pearson(preds[:, 0], targets[:, 0])
        self.test_r2(preds[:, 0], targets[:, 0])
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/pearson", self.test_pearson, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/r2", self.test_r2, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        test_pred_y = torch.concat([x for x in self.test_results["preds"]]).detach().cpu().numpy()
        test_targets_y = (
            torch.concat([x for x in self.test_results["targets"]]).detach().cpu().numpy()
        )
        test_fig = plot_fig(test_pred_y, test_targets_y)
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                import wandb

                logger.experiment.log({"test/fig": wandb.Image(test_fig)})
                break

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
