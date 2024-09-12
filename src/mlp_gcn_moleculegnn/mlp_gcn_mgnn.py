
from torch_geometric.data import InMemoryDataset
from optuna_integration import PyTorchLightningPruningCallback
import optuna
from torch_scatter import scatter_sum
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Batch
from torch.nn import functional as F
from torch.nn import Embedding, Linear, Module, ModuleList, Parameter
from torch import Tensor
from torch_geometric.nn.models import MLP as GMLP
from torch_scatter import scatter_mean
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch import nn

from rdkit import Chem
import numpy as np
import pickle
import pandas as pd
from torch.utils.data import TensorDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.data import Dataset as GeometricDataset
from torch.utils.data import DataLoader, random_split
import lightning as L
from typing import Any
import os
import sys
from functools import partial
from lightning import Trainer, seed_everything
import torchmetrics as tm
import torch

seed_everything(42)
sys.path.append("/home/huabei/project/SMTarRNA")


class ZincSampleDataModule(L.LightningDataModule):
    def __init__(self, dataset, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.save_hyperparameters(ignore=['dataset'])
        self.dataset = dataset
        self.train_dataset = None

    def setup(self, stage=None):
        dataset = self.dataset
        if self.train_dataset is None:
            if isinstance(dataset, GeometricDataset):
                self.data_type = 'geometric'
            else:
                self.data_type = 'normal'
            # 划分数据集, 8:1:1
            train_size = int(0.8 * len(dataset))
            val_size = int(0.1 * len(dataset))
            test_size = len(dataset) - train_size - val_size
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        if self.data_type == 'geometric':
            return GeometricDataLoader(self.train_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True)
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True)

    def val_dataloader(self):
        if self.data_type == 'geometric':
            return GeometricDataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        if self.data_type == 'geometric':
            return GeometricDataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)


def split_batch(batch: Any):
    if isinstance(batch, list):
        x = batch[0]
        y = batch[1]
        return x, y
    elif isinstance(batch, Data):
        return batch, batch.y
    else:
        raise TypeError(f'Unknown batch type {type(batch)}')


class PLModel(L.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        criteria: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['net', 'criteria'])
        self.net = net
        self.criteria = criteria

        self.train_mae = tm.MeanAbsoluteError()
        self.val_mae = tm.MeanAbsoluteError()
        self.test_mae = tm.MeanAbsoluteError()

        self.train_mse = tm.MeanSquaredError()
        self.val_mse = tm.MeanSquaredError()
        self.test_mse = tm.MeanSquaredError()

        self.train_r2 = tm.R2Score()
        self.val_r2 = tm.R2Score()
        self.test_r2 = tm.R2Score()

        self.train_pearson = tm.PearsonCorrCoef()
        self.val_pearson = tm.PearsonCorrCoef()
        self.test_pearson = tm.PearsonCorrCoef()

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {'optimizer': optimizer,
                    'lr_scheduler':
                        {'scheduler': scheduler,
                         'monitor': 'val/loss',
                         'interval': 'epoch',
                         'frequency': 1}}
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = split_batch(batch)
        pre = self(x)
        loss = self.criteria(pre, y)
        self.train_mae(pre[:, 0], y[:, 0])
        self.train_mse(pre[:, 0], y[:, 0])
        self.train_r2(pre[:, 0], y[:, 0])
        self.train_pearson(pre[:, 0], y[:, 0])
        self.log('train/loss', loss, on_step=True, on_epoch=True,
                 prog_bar=True, batch_size=self.trainer.datamodule.hparams.batch_size)
        self.log_dict({'train/mae': self.train_mae,
                       'train/mse': self.train_mse,
                       'train/r2': self.train_r2,
                       'train/pearson': self.train_pearson},
                      on_step=False,
                      on_epoch=True,
                      prog_bar=True,
                      batch_size=self.trainer.datamodule.hparams.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = split_batch(batch)
        pre = self(x)
        loss = self.criteria(pre, y)
        self.val_mae(pre[:, 0], y[:, 0])
        self.val_mse(pre[:, 0], y[:, 0])
        self.val_r2(pre[:, 0], y[:, 0])
        self.val_pearson(pre[:, 0], y[:, 0])
        self.log('val/loss', loss, on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=self.trainer.datamodule.hparams.batch_size)
        self.log_dict({'val/mae': self.val_mae,
                       'val/mse': self.val_mse,
                       'val/r2': self.val_r2,
                       'val/pearson': self.val_pearson}, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = split_batch(batch)
        pre = self(x)
        loss = self.criteria(pre, y)
        self.test_mae(pre[:, 0], y[:, 0])
        self.test_mse(pre[:, 0], y[:, 0])
        self.test_r2(pre[:, 0], y[:, 0])
        self.test_pearson(pre[:, 0], y[:, 0])
        self.log('test/loss', loss, on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=self.trainer.datamodule.hparams.batch_size)
        self.log_dict({'test/mae': self.test_mae,
                       'test/mse': self.test_mse,
                       'test/r2': self.test_r2,
                       'test/pearson': self.test_pearson}, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.trainer.datamodule.hparams.batch_size)
        return loss


class FPDataset(TensorDataset):
    def __init__(self, data_dir: str, cmpx: str):
        zinc_1_600_smiles_ecfp = pd.read_hdf(os.path.join(
            data_dir, 'raw/zinc_id_smiles_ecfp.h5'), key='data')
        dock_energy = pickle.load(
            open(os.path.join(data_dir, 'raw/total_data_dock_energy.pkl'), 'rb'))
        # 提取最佳对接能量
        total_data_best = {k: v[0] for k, v in dock_energy.items()}
        # 生成最佳能量表
        total_data_best_df = pd.DataFrame.from_dict(
            total_data_best,
            columns=["total", "inter", "intra", "torsions", "intra best pose"],
            orient="index",
        )
        total_data_best_df.index.name = "zinc_id"
        total_data_best_df = zinc_1_600_smiles_ecfp.join(
            total_data_best_df, how="left")
        x = total_data_best_df['fingerprint'].to_list()
        x = np.array(x)
        y = total_data_best_df[['total', 'inter', 'intra',
                                'torsions', 'intra best pose']].to_numpy()
        super().__init__(torch.tensor(x, dtype=torch.float32),
                         torch.tensor(y, dtype=torch.float32))


elements = [6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
ele2idx = {ele: i for i, ele in enumerate(elements)}


def smiles_to_graph(smiles, label):
    label = torch.tensor(label, dtype=torch.float32).view(1, -1)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([ele2idx[atom.GetAtomicNum()]])
    atom_features = torch.tensor(atom_features, dtype=torch.long)
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.extend([[start, end], [end, start]])
        edge_attr.extend([[bond.GetBondTypeAsDouble()],
                         [bond.GetBondTypeAsDouble()]])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    # 芳香键设置为0
    edge_attr[edge_attr == 1.5] = 0
    edge_attr = edge_attr.to(torch.long)
    return Data(x=atom_features, edge_index=edge_index, edge_attr=edge_attr, y=label)


class MoleculeDataset(InMemoryDataset):
    def __init__(self, data_dir: str):
        super(MoleculeDataset, self).__init__(root=data_dir)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def raw_file_names(self):
        return ['zinc_id_smiles_ecfp.h5', 'total_data_dock_energy.pkl']

    def processed_file_names(self):
        return ['Ligands_Graph_Data_Multi_Label_GCN.pt']

    def process(self):
        zinc_1_600_smiles_ecfp = pd.read_hdf(self.raw_paths[0], key='data')
        dock_energy = pickle.load(open(self.raw_paths[1], 'rb'))
        # 提取最佳对接能量
        total_data_best = {k: v[0] for k, v in dock_energy.items()}
        # 生成最佳能量表
        total_data_best_df = pd.DataFrame.from_dict(
            total_data_best,
            columns=["total", "inter", "intra", "torsions", "intra best pose"],
            orient="index",
        )
        total_data_best_df.index.name = "zinc_id"
        total_data_best_df = zinc_1_600_smiles_ecfp.join(
            total_data_best_df, how="left")
        smiles = total_data_best_df['smiles'].to_list()
        labels = total_data_best_df[[
            'total', 'inter', 'intra', 'torsions', 'intra best pose']].to_numpy()
        data_list = []
        for i in range(len(smiles)):
            data = smiles_to_graph(smiles[i], labels[i])
            if data is not None:
                data_list.append(data)
        # random.shuffle(data_list)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def get_molecule_dataset(cmpx, model_type):
    if model_type == 'MLP':
        dataset = FPDataset(
            data_dir=f'data/dataset/dataset/{cmpx}_100w', cmpx=cmpx)
    elif model_type == 'GCN':
        dataset = MoleculeDataset(data_dir=f'data/dataset/dataset/{cmpx}_100w')
    elif model_type == 'MoleculeGNN':
        from src.data.components.zinc_complex_data_molecule_gnn import ZincComplexDataMoleculeGnn
        dataset = ZincComplexDataMoleculeGnn(
            data_dir=f'data/dataset/dataset/{cmpx}_100w', cmpx=cmpx)
    return dataset


# 构建MLP模型
class MLP(nn.Module):
    def __init__(self, hidden_dim=512, hidden_layers=3):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2048, hidden_dim)  # 创建第一个全连接层
        self.norm = nn.LayerNorm(hidden_dim)
        self.h_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_layers - 1)])
        self.h_norm = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(hidden_layers - 1)])
        self.out_layer = nn.Linear(hidden_dim, 5)  # 创建输出层

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.norm(x)
        for i, layer in enumerate(self.h_layers):
            x = F.relu(layer(x))
            x = self.h_norm[i](x)
        x = self.out_layer(x)
        return x


class GCN(torch.nn.Module):
    def __init__(self, in_channels: int = 10, hidden_channels: int = 64, num_layers: int = 3, out_channels: int = 5):
        """in_channels: 输入特征维度, [6, 7, 8, 9, 14, 15, 16, 17, 35, 53]"""
        super().__init__()
        self.in_emb = torch.nn.Embedding(in_channels, hidden_channels)
        self.in_enc = GMLP([hidden_channels, hidden_channels,
                           hidden_channels], norm='graph_norm')
        self.edge_emb = torch.nn.Embedding(4, hidden_channels)
        self.edge_weight = GMLP([hidden_channels, hidden_channels, 1])

        # self.gconv = GCN(hidden_channels, hidden_channels, num_layers)
        # self.gconv = GIN(hidden_channels, hidden_channels, num_layers)
        self.gconv = torch.nn.ModuleList(
            [GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers)])
        self.gnorm = torch.nn.ModuleList(
            [GraphNorm(hidden_channels) for _ in range(num_layers)])
        self.decoder = GMLP(
            [hidden_channels, hidden_channels, hidden_channels], norm='graph_norm')
        self.out_decoder = GMLP(
            [hidden_channels, hidden_channels, out_channels], norm='layer_norm')

    def forward(self, batch_data):
        x = self.in_emb(batch_data.x.squeeze())
        x = self.in_enc(x, batch=batch_data.batch)
        edge_weight = self.edge_weight(
            self.edge_emb(batch_data.edge_attr.squeeze()))
        edge_weight = F.sigmoid(edge_weight)
        # print(edge_weight)
        # print(edge_weight.shape, x.shape)
        # x = self.gconv(x, batch_data.edge_index, edge_weight, batch=batch_data.batch)
        for i, gc in enumerate(self.gconv):
            x = gc(x, batch_data.edge_index, edge_weight)
            x = F.relu(x) + x
            x = self.gnorm[i](x)
        x = self.decoder(x, batch=batch_data.batch)
        m = scatter_mean(x, batch_data.batch, dim=0)
        return self.out_decoder(m)


class MoleculeGCN(MessagePassing):
    def __init__(self, input_dim, output_dim):
        """
        :param input_dim: N_atoms
        :param output_dim: dim
        """
        super().__init__(aggr="add")  # "Add" aggregation (Step 5).
        self.gamma = Embedding(input_dim, 1)
        self.w_atom = Linear(output_dim, output_dim)
        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.weight.data = Parameter(
            torch.ones(self.gamma.weight.data.shape))
        # self.w_atom.reset_parameters()

    def forward(self, x, x_ori, edge_index, edge_attr):
        # x has shape [N, input_dim]
        # edge_index has shape [2, E]
        # Step 2: Linearly transform node feature matrix.
        h_x = torch.relu(self.w_atom(x))
        gammas = torch.sigmoid(self.gamma(x_ori))[edge_index[1]]
        # 计算边的权重
        edge_weight = torch.exp(-gammas * (edge_attr**2))

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=h_x, edge_weight=edge_weight)

        return F.normalize(out + x, 2, 1)

    def message(self, x_j, edge_weight):
        # x_j has shape [E, output_dim]
        # x_j是j节点的特征，即edge_index的第二行的节点的特征
        # Step 4: Normalize node features.
        return edge_weight * x_j


class MoleculeGnn(Module):
    """这是MolecularGNN那篇文献的pyg实现."""

    def __init__(
        self,
        in_node_nf: int = 11,
        hidden_nf: int = 64,
        n_layers: int = 3,
        out_layers: int = 3,
        out_node_nf=5,
        dropout=0,
    ):
        """这个是自定义的模型.

        :param input_dim: 元素个数
        :param hidden_nf: dim
        :param n_layers: 隐藏层个数
        :param out_layers: Dense层个数
        :param out_node_nf: 输出维度
        :param dropout:
        """
        super().__init__()
        self.n_layers = n_layers
        self.out_layers = out_layers

        self.embd = Embedding(in_node_nf, hidden_nf, dtype=torch.float32)

        self.m_gcn = ModuleList(
            [MoleculeGCN(in_node_nf, hidden_nf) for _ in range(n_layers)])

        self.lin = ModuleList([Linear(hidden_nf, hidden_nf)
                              for _ in range(out_layers)])
        self.w_property = Linear(hidden_nf, out_node_nf)
        # self.mlp = MLP(input_dim=hidden_nf, hidden_nf=hidden_dim,
        #                out_node_nf=out_node_nf, num_layers=out_layers)

    def forward(self, batch: Batch):
        x = self.embd(batch.x)
        for m in range(self.n_layers):
            x = self.m_gcn[m](
                x=x, x_ori=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr
            )
        for i in range(self.out_layers):
            x = torch.relu(self.lin[i](x))
        m_x = scatter_sum(x, batch.batch, dim=0)
        properties = self.w_property(m_x)
        return torch.squeeze(properties, dim=1)


def MLP_objective(trial: optuna.trial.Trial) -> float:
    hidden_layers = trial.suggest_int('n_layers', 1, 3)
    hidden_nf = trial.suggest_int('hidden_nf', 64, 1024)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    net = MLP(hidden_dim=hidden_nf, hidden_layers=hidden_layers)
    criteria = torch.nn.MSELoss()
    optimizer = partial(torch.optim.Adamax, lr=lr, weight_decay=weight_decay)
    scheduler = partial(torch.optim.lr_scheduler.ReduceLROnPlateau,
                        mode='min', factor=0.1, patience=10, min_lr=1e-5, verbose=True)
    model = PLModel(net, criteria, optimizer, scheduler=scheduler)
    trainer = Trainer(max_epochs=MAX_EPOCHS, enable_checkpointing=False, callbacks=[
                      PyTorchLightningPruningCallback(trial, monitor='val/loss')],
                      accelerator='gpu', enable_progress_bar=True,
                      devices=DEVICE)
    hyperparameters = {'hidden_layers': hidden_layers,
                       'hidden_nf': hidden_nf, 'lr': lr, 'weight_decay': weight_decay}
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=data_module)
    return trainer.callback_metrics['val/loss'].item()


def GCN_objective(trial: optuna.trial.Trial) -> float:
    hidden_nf = trial.suggest_int('hidden_nf', 64, 512)
    n_layers = trial.suggest_int('n_layers', 1, 8)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    net = GCN(in_channels=10, hidden_channels=hidden_nf,
              num_layers=n_layers, out_channels=5)
    criteria = torch.nn.MSELoss()
    optimizer = partial(torch.optim.Adamax, lr=lr, weight_decay=weight_decay)
    scheduler = partial(torch.optim.lr_scheduler.ReduceLROnPlateau,
                        mode='min', factor=0.1, patience=10, min_lr=1e-5, verbose=True)
    model = PLModel(net, criteria, optimizer, scheduler=scheduler)
    trainer = Trainer(max_epochs=MAX_EPOCHS, enable_checkpointing=False, callbacks=[
                      PyTorchLightningPruningCallback(trial, monitor='val/loss')],
                      accelerator='gpu', enable_progress_bar=True,
                      devices=DEVICE)
    hyperparameters = {'hidden_nf': hidden_nf,
                       'n_layers': n_layers, 'lr': lr, 'weight_decay': weight_decay}
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=data_module)
    return trainer.callback_metrics['val/loss'].item()


def MoleculeGNN_objective(trial: optuna.trial.Trial) -> float:
    hidden_nf = trial.suggest_int('hidden_nf', 64, 512)
    n_layers = trial.suggest_int('n_layers', 3, 6)
    out_layers = trial.suggest_int('out_layers', 3, 6)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    net = MoleculeGnn(in_node_nf=11, hidden_nf=hidden_nf,
                      n_layers=n_layers, out_layers=out_layers, out_node_nf=5)
    criteria = torch.nn.MSELoss()
    optimizer = partial(torch.optim.Adamax, lr=lr, weight_decay=weight_decay)
    scheduler = partial(torch.optim.lr_scheduler.ReduceLROnPlateau,
                        mode='min', factor=0.1, patience=10, min_lr=1e-5, verbose=True)
    model = PLModel(net, criteria, optimizer, scheduler=scheduler)
    trainer = Trainer(max_epochs=MAX_EPOCHS, enable_checkpointing=False, callbacks=[
                      PyTorchLightningPruningCallback(trial, monitor='val/loss')],
                      accelerator='gpu', enable_progress_bar=True,
                      devices=DEVICE)
    hyperparameters = {'hidden_nf': hidden_nf, 'n_layers': n_layers,
                       'out_layers': out_layers, 'lr': lr, 'weight_decay': weight_decay}
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=data_module)
    return trainer.callback_metrics['val/loss'].item()


def MLP_Metrics(trial: optuna.trial.Trial, MAX_EPOCHS=50, DEVICES=[0]):
    # table = PrettyTable(field_names=['Stage', 'MAE', 'MSE', 'R2', 'Pearson'])
    pd_table = pd.DataFrame(columns=['Stage', 'MAE', 'MSE', 'R2', 'Pearson'])
    hidden_layers = trial.suggest_int('n_layers', 1, 3)
    hidden_nf = trial.suggest_int('hidden_nf', 64, 1024)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    net = MLP(hidden_dim=hidden_nf, hidden_layers=hidden_layers)
    criteria = torch.nn.MSELoss()
    optimizer = partial(torch.optim.Adamax, lr=lr, weight_decay=weight_decay)
    scheduler = partial(torch.optim.lr_scheduler.ReduceLROnPlateau,
                        mode='min', factor=0.1, patience=10, min_lr=1e-5, verbose=True)
    model = PLModel(net, criteria, optimizer, scheduler=scheduler)
    trainer = Trainer(max_epochs=MAX_EPOCHS, enable_checkpointing=False, devices=DEVICES)
    hyperparameters = {'hidden_layers': hidden_layers,
                       'hidden_nf': hidden_nf, 'lr': lr, 'weight_decay': weight_decay}
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=data_module)
    # table.add_row(['train', trainer.callback_metrics['train/mae'].item(), trainer.callback_metrics['train/mse'].item(), trainer.callback_metrics['train/r2'].item(), trainer.callback_metrics['train/pearson'].item()])
    # table.add_row(['val', trainer.callback_metrics['val/mae'].item(), trainer.callback_metrics['val/mse'].item(), trainer.callback_metrics['val/r2'].item(), trainer.callback_metrics['val/pearson'].item()])
    pd_table.loc[0] = ['train', trainer.callback_metrics['train/mae'].item(), trainer.callback_metrics['train/mse'].item(),
                       trainer.callback_metrics['train/r2'].item(), trainer.callback_metrics['train/pearson'].item()]
    pd_table.loc[1] = ['val', trainer.callback_metrics['val/mae'].item(), trainer.callback_metrics['val/mse'].item(),
                       trainer.callback_metrics['val/r2'].item(), trainer.callback_metrics['val/pearson'].item()]
    trainer.test(model, datamodule=data_module)
    # table.add_row(['test', trainer.callback_metrics['test/mae'].item(), trainer.callback_metrics['test/mse'].item(), trainer.callback_metrics['test/r2'].item(), trainer.callback_metrics['test/pearson'].item()])
    pd_table.loc[2] = ['test', trainer.callback_metrics['test/mae'].item(), trainer.callback_metrics['test/mse'].item(),
                       trainer.callback_metrics['test/r2'].item(), trainer.callback_metrics['test/pearson'].item()]
    # print(trial.params)
    # print(table)
    return pd_table


def GCN_Metrics(trial: optuna.trial.Trial, MAX_EPOCHS=50, DEVICES=[0]):
    pd_table = pd.DataFrame(columns=['Stage', 'MAE', 'MSE', 'R2', 'Pearson'])
    hidden_nf = trial.suggest_int('hidden_nf', 64, 512)
    n_layers = trial.suggest_int('n_layers', 1, 8)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    net = GCN(in_channels=10, hidden_channels=hidden_nf,
              num_layers=n_layers, out_channels=5)
    criteria = torch.nn.MSELoss()
    optimizer = partial(torch.optim.Adamax, lr=lr, weight_decay=weight_decay)
    scheduler = partial(torch.optim.lr_scheduler.ReduceLROnPlateau,
                        mode='min', factor=0.1, patience=10, min_lr=1e-5, verbose=True)
    model = PLModel(net, criteria, optimizer, scheduler=scheduler)
    trainer = Trainer(max_epochs=MAX_EPOCHS, enable_checkpointing=False, devices=DEVICES)
    hyperparameters = {'hidden_nf': hidden_nf,
                       'n_layers': n_layers, 'lr': lr, 'weight_decay': weight_decay}
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=data_module)
    pd_table.loc[0] = ['train', trainer.callback_metrics['train/mae'].item(), trainer.callback_metrics['train/mse'].item(),
                       trainer.callback_metrics['train/r2'].item(), trainer.callback_metrics['train/pearson'].item()]
    pd_table.loc[1] = ['val', trainer.callback_metrics['val/mae'].item(), trainer.callback_metrics['val/mse'].item(),
                       trainer.callback_metrics['val/r2'].item(), trainer.callback_metrics['val/pearson'].item()]
    trainer.test(model, datamodule=data_module)
    pd_table.loc[2] = ['test', trainer.callback_metrics['test/mae'].item(), trainer.callback_metrics['test/mse'].item(),
                       trainer.callback_metrics['test/r2'].item(), trainer.callback_metrics['test/pearson'].item()]
    return pd_table


def MoleculeGNN_Metrics(trial: optuna.trial.Trial, MAX_EPOCHS=50, DEVICES=[0]):
    pd_table = pd.DataFrame(columns=['Stage', 'MAE', 'MSE', 'R2', 'Pearson'])
    hidden_nf = trial.suggest_int('hidden_nf', 64, 512)
    n_layers = trial.suggest_int('n_layers', 3, 6)
    out_layers = trial.suggest_int('out_layers', 3, 6)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    net = MoleculeGnn(in_node_nf=11, hidden_nf=hidden_nf,
                      n_layers=n_layers, out_layers=out_layers, out_node_nf=5)
    criteria = torch.nn.MSELoss()
    optimizer = partial(torch.optim.Adamax, lr=lr, weight_decay=weight_decay)
    scheduler = partial(torch.optim.lr_scheduler.ReduceLROnPlateau,
                        mode='min', factor=0.1, patience=10, min_lr=1e-5, verbose=True)
    model = PLModel(net, criteria, optimizer, scheduler=scheduler)
    trainer = Trainer(max_epochs=MAX_EPOCHS, enable_checkpointing=False, devices=DEVICES)
    hyperparameters = {'hidden_nf': hidden_nf, 'n_layers': n_layers,
                       'out_layers': out_layers, 'lr': lr, 'weight_decay': weight_decay}
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=data_module)
    pd_table.loc[0] = ['train', trainer.callback_metrics['train/mae'].item(), trainer.callback_metrics['train/mse'].item(),
                       trainer.callback_metrics['train/r2'].item(), trainer.callback_metrics['train/pearson'].item()]
    pd_table.loc[1] = ['val', trainer.callback_metrics['val/mae'].item(), trainer.callback_metrics['val/mse'].item(),
                       trainer.callback_metrics['val/r2'].item(), trainer.callback_metrics['val/pearson'].item()]
    trainer.test(model, datamodule=data_module)
    pd_table.loc[2] = ['test', trainer.callback_metrics['test/mae'].item(), trainer.callback_metrics['test/mse'].item(),
                       trainer.callback_metrics['test/r2'].item(), trainer.callback_metrics['test/pearson'].item()]
    return pd_table


def train_cmpx_model(cmpx, model_name, STORAGE_PATH, BATCH_SIZE, MAX_EPOCHS, N_TRIALS):
    dataset = get_molecule_dataset(cmpx, model_type=model_name)
    global data_module
    data_module = ZincSampleDataModule(
        dataset, batch_size=BATCH_SIZE, num_workers=8)
    if model_name == 'MLP':
        objective = MLP_objective
    elif model_name == 'GCN':
        objective = GCN_objective
    elif model_name == 'MoleculeGNN':
        objective = MoleculeGNN_objective
    else:
        raise ValueError(f'Unknown study name {model_name}')
    pruner = optuna.pruners.MedianPruner()
    study = optuna.load_study(pruner=pruner,
                              storage=STORAGE_PATH,
                              study_name=f'{cmpx}_{model_name}')
    study.set_user_attr('Batch Size', BATCH_SIZE)
    study.set_user_attr('Max Epochs', MAX_EPOCHS)
    n_trials = N_TRIALS - len(study.trials)
    if n_trials > 0:
        study.optimize(objective, n_trials=n_trials)
    del data_module  # 释放内存
    # return study.best_params


def test_cmpx_model(cmpx, model_name, BATCH_SIZE, MAX_EPOCHS, DEVICES, STORAGE_PATH):
    dataset = get_molecule_dataset(cmpx, model_type=model_name)
    global data_module
    data_module = ZincSampleDataModule(
        dataset, batch_size=BATCH_SIZE, num_workers=8)
    if model_name == 'MLP':
        metrics = MLP_Metrics
    elif model_name == 'GCN':
        metrics = GCN_Metrics
    elif model_name == 'MoleculeGNN':
        metrics = MoleculeGNN_Metrics
    else:
        raise ValueError(f'Unknown study name {model_name}')
    study = optuna.load_study(storage=STORAGE_PATH,
                              study_name=f'{cmpx}_{model_name}')
    best_trial = study.best_trial
    results = metrics(best_trial, MAX_EPOCHS, DEVICES)
    del data_module  # 释放内存
    return results

