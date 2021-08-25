import numpy as np
from satflow.models import MetNet, Perceiver, NowcastingGAN
from satflow.models.base import list_models, create_model
from satflow.data.datamodules import SatFlowDataModule
import yaml
import torch


def load_config(config_file):
    with open(config_file, "r") as cfg:
        return yaml.load(cfg, Loader=yaml.FullLoader)


TRAIN_SOURCES = {
    "train": "satflow-test.tar",
    "val": "satflow-test.tar",
    "test": "satflow-test.tar",
}
DATA_DIR = "datasets/"


def test_metnet_e2e():
    config = load_config("satflow/configs/model/metnet.yaml")
    config.pop("_target_")  # This is only for Hydra
    model = MetNet(**config)

    data_config = load_config("satflow/configs/datamodule/metnet.yaml")
    data_config.pop("_target_")
    data_config["sources"] = TRAIN_SOURCES
    data_config["data_dir"] = DATA_DIR
    datamodule = SatFlowDataModule(**data_config)
    datamodule.setup()
    x, y = next(iter(datamodule.train_dataloader()))
    assert not torch.isnan(x).any(), "Input included NaNs"
    assert not torch.isnan(y).any(), "Target included NaNs"
    model.eval()
    with torch.no_grad():
        out = model(x)
    # MetNet creates predictions for the center 1/4th
    assert out.size() == (
        2,
        config["forecast_steps"],
        config["output_channels"],
        config["input_size"] // 4,
        config["input_size"] // 4,
    )
    assert not torch.isnan(out).any(), "Output included NaNs"
