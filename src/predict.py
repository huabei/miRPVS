# 此文件作为模型的应用

# 导入必要的库
import logging
from typing import List, Tuple

import omegaconf
import pyrootutils
import torch
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import gc
import os
import time
import zipfile

import ray
from ray.experimental.tqdm_ray import tqdm
from torch_geometric.data import InMemoryDataset

# 导入模型
from src.models.components.egnn import Egnn

ray.init(num_gpus=4, num_cpus=40, include_dashboard=False)
ray_logger = logging.getLogger(__name__)
ray_logger.setLevel(logging.INFO)
# 保存日志到文件
ray_logger.addHandler(logging.FileHandler("predict.log", mode="a"))
# logging.basicConfig(level=logging.INFO, filename='predict.log', filemode='a', format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')


@ray.remote(num_gpus=0, num_cpus=1)
def load_dataset(zip_file: str, dataset_file: str):
    archive = zipfile.ZipFile(zip_file, "r")
    extract_file = archive.open(archive.namelist()[0])
    return torch.load(extract_file), dataset_file


class ZincInferDataset(InMemoryDataset):
    """前向推理使用的数据集."""

    def __init__(self, data, transform=None, pre_transform=None, pre_filter=None):
        # data为（data, slices）
        super().__init__(
            None, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter
        )
        self.data, self.slices = data


@ray.remote(num_gpus=1, num_cpus=6)
class Predictor(torch.nn.Module):
    """预测器，用于预测数据集中的数据，继承torch.nn.Module,是为了便于载入模型."""

    def __init__(self, net: torch.nn.Module, cfg: DictConfig):
        super().__init__()
        logging.basicConfig(level=logging.INFO)
        # 将net设为推理模式
        net.eval()
        # 将net转移到gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = net
        self.net.to(self.device)
        self.cfg = cfg
        logging.info("Predictor initialized")

    def forward(self, batch):
        return self.net(batch)

    def predict(self, dataset):
        """传入一个Data的列表，载入Dataloader，预测结果."""
        logging.info("Predicting")
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        predictions = []
        zinc_id = []
        with torch.inference_mode():
            for batch in tqdm(dataloader):
                batch = batch.to(self.device)
                predictions.append(self(batch).cpu())
                zinc_id.append(batch.id.cpu())
        # 转换为numpy cpu
        predictions = torch.concat([x for x in predictions]).numpy()
        zinc_id = torch.concat([x for x in zinc_id]).numpy()
        return predictions, zinc_id

    def predict_one(self, data):
        """传入一个Data，预测结果."""
        data = data.to(self.device)
        with torch.inference_mode():
            predictions = self(data).cpu().numpy()
        return predictions


@ray.remote(num_cpus=1)
def save_results(predictions, output_path):
    """保存预测结果."""
    torch.save(predictions, output_path)
    return True


def predict(
    dataset_refers: List[ray.ObjectRef],
    model_refers: dict,
    res: List[ray.ObjectRef],
    predictions: List[ray.ObjectRef],
    cfg: DictConfig,
):
    data_complete_refers, dataset_refers = ray.wait(dataset_refers, num_returns=1)
    # print(ray.get(data_complete_refers))
    data_complete_refers: List[ray.ObjectRef]

    data, dataset_file = ray.get(data_complete_refers[0])
    print("pop data success")
    # dataset_refer = dataset_refers.pop(0)

    dataset = ray.put(ZincInferDataset(data))
    # print("load data success")
    # 预测
    if len(predictions) > 0:
        # 进入下一次预测之前等待上一次预测完成
        ray.get(predictions)

        # 等待save task完成
    while len(res) > 8:
        print(f"wait for save{len(res)}")
        time.sleep(1)
        _, res = ray.wait(res)

    predictions = []
    for complex, model_refer in model_refers.items():
        # 如果文件存在则不运行
        if os.path.exists(os.path.join(cfg.output_dir, f"{complex}/{dataset_file}.results")):
            print(f"{complex} {dataset_file}.results already exists")
            continue
        pre = model_refer.predict.remote(dataset)
        predictions.append(pre)

        ray_logger.info(
            f"Saving {complex} predictions to {os.path.join(cfg.output_dir, f'{complex}_{dataset_file}.results')}"
        )
        res.append(
            save_results.remote(
                pre, os.path.join(cfg.output_dir, f"{complex}/{dataset_file}.results")
            )
        )
    return dataset_refers, res, predictions


def main(cfg: DictConfig) -> Tuple[dict, dict]:
    """主函数."""
    for k, v in cfg.ckpt_path.items():
        ray_logger.info(f"Will load {k} model from {v}")

    cfg.ckpt_path: dict
    # 实例化actor
    # logging.info(f"Loading model from {cfg.ckpt_path}")
    ray_logger.info("Loading model")
    net = Egnn(**cfg.model)
    # 生成对应模型的actor
    ray_logger.info("Generating actor")

    # TODO：需要知道net是否是在不同的actor中共享
    model_refers = {k: Predictor.remote(net, cfg) for k in cfg.ckpt_path.keys()}
    for k, v in cfg.ckpt_path.items():
        # 载入模型
        logging.info(f"Loading {k} model from {v}")
        state_dict = torch.load(v)["state_dict"]
        model_refers[k].load_state_dict.remote(state_dict, strict=False)
    # res = []
    dataset_refers = []
    res = []
    predictions = []
    # 遍历文件夹内的数据集
    for dataset_f in os.listdir(cfg.data_dir):
        if not dataset_f.endswith(".pt.zip"):
            continue
        dataset_path = os.path.join(cfg.data_dir, dataset_f)
        ray_logger.info(f"Loading data from {dataset_path}")

        # 多线程载入数据集,同时最多载入5个数据集
        if len(dataset_refers) > 4:
            # 弹出第一个数据集
            # print("pop data")
            dataset_refers, res, predictions = predict(
                dataset_refers, model_refers, res, predictions, cfg
            )
            print("clear memory")
            # 强制回收内存
            gc.collect()
            # 等待所有actor预测任务,防止内存溢出
            # if len(predictions) > 0:
            #     # print("wait for save")
            #     ray.get(predictions)
        dataset_refers.append(load_dataset.remote(dataset_path, dataset_f))
    # 预测最后5个数据集
    while len(dataset_refers) > 0:
        dataset_refers, res, predictions = predict(
            dataset_refers, model_refers, res, predictions, cfg
        )
        # 强制回收内存
        gc.collect()
    print("wait for save")
    ray.get(res)

    # ray.get(res)


if __name__ == "__main__":
    # 加载配置文件
    # 计时
    ray_logger.info("Start")
    start = time.time()
    cfg = omegaconf.OmegaConf.load("configs/predict.yaml")
    # print(cfg.tags)
    # raise NotImplementedError
    main(cfg)
    end = time.time()
    ray_logger.info(f"Time cost: {end-start}")
