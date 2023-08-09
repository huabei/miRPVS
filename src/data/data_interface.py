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

import importlib
import inspect
import logging
from typing import Optional

import lightning.pytorch as pl
import torch
from torch.utils.data import Dataset, random_split
from torch_geometric.loader import DataLoader

# import torchvision.transforms as transforms


class DInterface(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        dataset: str = "",
        batch_size: int = 64,
        num_workers: int = 8,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.kwargs = kwargs
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage=None):
        """这个函数会在PL初始化中自动调用，根据不同的stage,选择生成不同的Dataloader，"""
        self.load_data_module()
        if stage == "fit" or stage is None:
            # Assign train/val datasets for use in dataloaders
            dataset = self.instancialize(data_dir=self.hparams.data_dir, train=True)
            train_size = int(len(dataset) * 0.8)
            # val_size = int(len(dataset) * 0.2)
            val_size = len(dataset) - train_size
            self.data_train, self.data_val = random_split(
                dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(1234),
            )
        elif stage == "test":
            self.data_test = self.instancialize(data_dir=self.hparams.data_dir, train=False)

        elif stage == "predict":
            logging.warning("No predict dataset!")
        else:
            raise ValueError(f"Stage {stage} not recognized")

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=self.hparams.pin_memory,
        )

    def load_data_module(self):
        """导入数据模块中的类，其中模块名和类的名需要满足格式要求。"""
        name = self.hparams.dataset
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = "".join([i.capitalize() for i in name.split("_")])
        # try:
        self.data_module = getattr(
            importlib.import_module(".components." + name, package=__package__), camel_name
        )
        # except (ImportError, AttributeError):
        #     raise ValueError(
        #         f"Invalid Dataset File Name or Invalid Class Name data.components.{name}.{camel_name}"
        #     )

    def instancialize(self, **other_args):
        """Instancialize a model using the corresponding parameters from self.hparams dictionary.

        You can also input any args to overwrite the corresponding value in self.kwargs.
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
