'''
作为所有pytorch lightning模型的基类，提供一些通用的方法，其它模型继承该类，重写一些方法即可
'''

import torch
from torch import nn
import numpy as np
from sklearn.metrics import r2_score
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
import lightning.pytorch as pl
from torch_geometric.data import Data
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')


class PLBaseModel(pl.LightningModule):
    def __init__(self, loss, lr, lr_scheduler, batch_size, **kargs):
        super().__init__()
        # 保存超参数
        self.save_hyperparameters()
        # 加载损失函数
        self.configure_loss()
        self.training_step_outputs = defaultdict(list)
        self.validation_step_outputs = defaultdict(list)
        self.test_step_outputs = defaultdict(list)
    
    def forward(self):
        raise NotImplementedError('forward() is not implemented')

    def training_step(self, batch: Data, batch_idx):
        p_energy = self(batch)
        y = batch.y
        self.training_step_outputs['preds'].append(p_energy)
        self.training_step_outputs['true'].append(y)
        loss = self.loss_function(torch.squeeze(p_energy), y)
        self.log('train_loss', loss, prog_bar=True, batch_size=self.hparams.batch_size)
        return {'loss': loss, 'preds': p_energy, 'true': y}

    def on_train_epoch_end(self) -> None:
        '''计算训练集的mae'''
        self.train_pred_y = torch.concat([x for x in self.training_step_outputs['preds']]).detach().cpu().numpy()
        self.train_true_y = torch.concat([x for x in self.training_step_outputs['true']]).detach().cpu().numpy()
        self.training_step_outputs.clear()
        mae_train = np.mean(np.abs(self.train_pred_y - self.train_true_y)) # 平均绝对误差
        self.log('train_mae', mae_train)

    def on_train_end(self) -> None:
        self._share_val_step(self.train_pred_y, self.train_true_y, 'train')
        self._share_val_step(self.val_pred_y, self.val_true_y, 'val')

    def validation_step(self, batch, batch_idx):
        p_energy = self(batch)
        y = batch.y
        self.validation_step_outputs['preds'].append(p_energy)
        self.validation_step_outputs['true'].append(y)
        loss = self.loss_function(torch.squeeze(p_energy), y)
        self.log('val_loss', loss, prog_bar=True, batch_size=self.hparams.batch_size)
        return {'loss': loss, 'preds': p_energy, 'true': y}

    def on_validation_epoch_end(self) -> None:
        '''计算验证集的mae'''
        self.val_pred_y = torch.concat([x for x in self.validation_step_outputs['preds']]).detach().cpu().numpy()
        self.val_true_y = torch.concat([x for x in self.validation_step_outputs['true']]).detach().cpu().numpy()
        self.validation_step_outputs.clear()
        mae_val = np.mean(np.abs(self.val_pred_y - self.val_true_y))
        self.log('val_mae', mae_val)
        
    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        p_energy = self(batch)
        y = batch.y
        self.test_step_outputs['preds'].append(p_energy)
        self.test_step_outputs['true'].append(y)
        # loss = self.loss_function(torch.squeeze(p_energy), batch.y)
        return {'preds': p_energy, 'true': y}

    def on_test_epoch_end(self):
        # Make the Progress Bar leave there
        self.test_pred_y = torch.concat([x for x in self.test_step_outputs['preds']]).detach().cpu().numpy()
        self.test_true_y = torch.concat([x for x in self.test_step_outputs['true']]).detach().cpu().numpy()
        self.test_step_outputs.clear()
        mae_test = np.mean(np.abs(self.test_pred_y - self.test_true_y))
        self.log('test_mae', mae_test)
        self._share_val_step(self.test_pred_y, self.test_true_y, 'test')

    def _share_val_step(self, pred, true, stage: str):
        r2 = r2_score(true, pred)
        pearson = np.corrcoef(true, pred)[0, 1]
        fig = plot_fit_confidence_bond(true, pred, r2, annot=False)
        tensorboard_logger = self.loggers[0].experiment
        tensorboard_logger.add_figure(f'{stage}_fig', fig, global_step=self.global_step)
        tensorboard_logger.add_scalar(f'{stage}_r2', r2, global_step=self.global_step)
        if len(self.loggers) > 1:
            wandb_logger = self.loggers[1].experiment
            wandb_logger.log({f'{stage}_fig': fig, f'{stage}_r2': r2, f'{stage}_pearson': pearson, 'global_step': self.global_step})
        
    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_t_max, # 多少个epoch衰减四分之一周期
                                                  eta_min=self.hparams.lr_decay_min_lr)
            elif self.hparams.lr_scheduler == 'cosine_warmup':
                scheduler = lrs.CosineAnnealingWarmRestarts(optimizer, T_0=self.hparams.lr_t_0,
                                                            T_mult=self.hparams.lr_t_mul,
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


class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    re
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2 # source + target dim
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1 # edge length

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        # 读出层
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf), # edge massage + node feature
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        # 产生coord的权重
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        # 产生edge的特征
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg
        return coord # (N, 3) 更新后的坐标

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1) # 半径

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff # 半径和相对位置

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord) # 计算半径和相对位置

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr) # 产生edge的特征:(E, hidden_nf)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat) # 更新坐标:(N, 3)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr) # 产生node的特征:(N, hidden_nf)

        return h, coord, edge_attr # 新的node特征和坐标



def plot_fit_confidence_bond(x, y, r2, annot=True):

    fig, ax = plt.subplots()
    ax.plot([-20, 0], [-20, 0], '-')
    ax.plot(x, y, 'o', color='tab:brown')
    if annot:
        num = 0
        for x_i, y_i in zip(x, y):
            ax.annotate(str(num), (x_i, y_i))
            num += 1
    ax.set_xlabel('True Energy(Kcal/mol)')
    ax.set_ylabel('Predict Energy(Kcal/mol)')
    ax.text(0.4, 0.9,
            'r2:  ' + str(r2), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
            fontsize=12)
    return fig


def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    return edges


def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr