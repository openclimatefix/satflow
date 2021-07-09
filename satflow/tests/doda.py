import numpy as np
from satflow.data.datasets import SatFlowDataset, CloudFlowDataset
from satflow.data.datamodules import SatFlowDataModule
import webdataset as wds
import yaml


def load_config(config_file):
    with open(config_file, "r") as cfg:
        return yaml.load(cfg, Loader=yaml.FullLoader)


# d = next(iter(dataset))
# print(d["time.pyd"])
config = load_config(
    "/home/bieker/Development/satflow/satflow/configs/datamodule/unet_datamodule.yaml"
)
print(config)
datamodule = SatFlowDataModule(**config)
datamodule.setup("fit")
data = next(iter(datamodule.train_dataloader()))
print(data)
