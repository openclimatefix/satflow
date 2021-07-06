# NIR1.6, VIS0.8 and VIS0.6 RGB for near normal view
import numpy as np
from satflow.data.datasets import SatFlowDataset, CloudFlowDataset
import webdataset as wds
import yaml


def load_config(config_file):
    with open(config_file, "r") as cfg:
        return yaml.load(cfg, Loader=yaml.FullLoader)["config"]


def test_satflow_cloudmask():
    dataset = wds.WebDataset("datasets/satflow-test.tar").decode()
    # d = next(iter(dataset))
    # print(d["time.pyd"])
    config = load_config("satflow/tests/configs/satflow.yaml")
    cloudflow = SatFlowDataset([dataset], config)
    data = next(iter(cloudflow))
    x, y = data
    assert x.shape == (13, 12, 128, 128)
    assert y.shape == (24, 1, 128, 128)


def test_satflow_all():
    dataset = wds.WebDataset("datasets/satflow-test.tar").decode()
    # d = next(iter(dataset))
    # print(d["time.pyd"])
    config = load_config("satflow/tests/configs/satflow_all.yaml")
    cloudflow = SatFlowDataset([dataset], config)
    data = next(iter(cloudflow))
    x, image, y = data
    assert x.shape == (13, 12, 128, 128)
    assert y.shape == (24, 1, 128, 128)
    assert image.shape == (24, 12, 128, 128)


def test_satflow_large():
    dataset = wds.WebDataset("datasets/satflow-test.tar").decode()
    # d = next(iter(dataset))
    # print(d["time.pyd"])
    config = load_config("satflow/tests/configs/satflow_large.yaml")
    cloudflow = SatFlowDataset([dataset], config)
    data = next(iter(cloudflow))
    x, y = data
    assert x.shape == (13, 12, 256, 256)
    assert y.shape == (24, 1, 256, 256)


def test_satflow_crop():
    dataset = wds.WebDataset("datasets/satflow-test.tar").decode()
    # d = next(iter(dataset))
    # print(d["time.pyd"])
    config = load_config("satflow/tests/configs/satflow_crop.yaml")
    cloudflow = SatFlowDataset([dataset], config)
    data = next(iter(cloudflow))
    x, y = data
    assert x.shape == (13, 12, 256, 256)
    assert y.shape == (24, 1, 64, 64)


def test_cloudflow():
    dataset = wds.WebDataset("datasets/satflow-test.tar").decode()
    # d = next(iter(dataset))
    # print(d["time.pyd"])
    config = load_config("satflow/tests/configs/satflow.yaml")
    cloudflow = CloudFlowDataset([dataset], config)
    data = next(iter(cloudflow))
    x, y = data
    assert x.shape == (13, 1, 128, 128)
    assert y.shape == (24, 1, 128, 128)
