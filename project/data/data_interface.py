# Copyright 2021 Zhongyang Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import importlib
import pytorch_lightning as pl
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import torch


# import torchvision.transforms as transforms


class DInterface(pl.LightningDataModule):

    def __init__(self, num_workers=8,
                 dataset='',
                 **kwargs):
        super().__init__()
        self.num_workers = num_workers
        self.dataset = dataset
        self.kwargs = kwargs
        self.batch_size = kwargs['batch_size']
        self.load_data_module()
        # Assign train/val datasets for use in dataloaders
        dataset = self.instancialize(train=True)
        train_size = int(len(dataset) * 0.8)
        val_size = int(len(dataset) * 0.1)
        test_size = len(dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset,
                                                                               [train_size, val_size, test_size],
                                                                               generator=torch.Generator().manual_seed(1234))

    def setup(self, stage=None):
        """这个函数会在PL初始化中自动调用，根据不同的stage,选择生成不同的Dataloader，"""

        if stage == 'fit' or stage is None:
            # self.trainset = self.instancialize(train=True)
            # self.valset = self.instancialize(train=False)
            self.trainset = self.train_dataset
            self.valset = self.val_dataset

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            # self.testset = self.instancialize(train=False)
            self.testset = self.test_dataset

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def load_data_module(self):
        """导入数据模块中的类，其中模块名和类的名需要满足格式要求。
        """
        name = self.dataset
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            self.data_module = getattr(importlib.import_module(
                '.' + name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}')

    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
            使用给定的参数实例化一个数据类，这个数据类是在数据模块里面定义的，达到了PL接口和数据的分离。
        """
        class_args = inspect.getfullargspec(self.data_module.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(**args1)
