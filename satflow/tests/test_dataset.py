# NIR1.6, VIS0.8 and VIS0.6 RGB for near normal view
import numpy as np
from satflow.data.datasets import SatFlowDataset, CloudFlowDataset
import webdataset as wds
import yaml


def load_config(config_file):
    with open(config_file, "r") as cfg:
        return yaml.load(cfg, Loader=yaml.FullLoader)["config"]


def test_satflow():
    dataset = wds.WebDataset("../../datasets/satflow-multi-test.tar").decode()
    pass


def test_cloudflow():
    dataset = wds.WebDataset("../../datasets/satflow-test.tar").decode()
    # d = next(iter(dataset))
    # print(d["time.pyd"])
    config = load_config("configs/satflow.yaml")
    cloudflow = CloudFlowDataset([dataset], config)
    data = next(iter(cloudflow))
    pass
