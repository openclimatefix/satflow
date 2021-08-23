import numpy as np
from satflow.models import MetNet, Perceiver, NowcastingGAN
import yaml
import torch


def load_config(config_file):
    with open(config_file, "r") as cfg:
        return yaml.load(cfg, Loader=yaml.FullLoader)


def test_perceiver_creation():
    config = load_config("satflow/configs/model/perceiver_metnet.yaml")
    config.pop("_target_")  # This is only for Hydra
    model = Perceiver(**config)
    x = torch.randn((2, 12, config["input_channels"], config["input_size"], config["input_size"]))
    model.eval()
    # TODO Get the Query/etc. like in the train/etc. Will currently return the embedding otherwise as of now
    with torch.no_grad():
        out = model(x)
    # MetNet creates predictions for the center 1/4th
    assert out.size() == (
        2,
        config["forecast_steps"],
        config["output_channels"],
        config["input_size"],
        config["input_size"],
    )


def test_nowcasting_gan_creation():
    config = load_config("satflow/configs/model/nowcasting_gan.yaml")
    config.pop("_target_")  # This is only for Hydra
    model = NowcastingGAN(**config)
    x = torch.randn(
        (2, 4, config["input_channels"], config["output_shape"], config["output_shape"])
    )
    model.eval()
    with torch.no_grad():
        out = model(x)
    # MetNet creates predictions for the center 1/4th
    assert out.size() == (
        2,
        config["forecast_steps"],
        config["input_channels"],
        config["output_shape"],
        config["output_shape"],
    )


def test_metnet_creation():
    config = load_config("satflow/configs/model/metnet.yaml")
    config.pop("_target_")  # This is only for Hydra
    model = MetNet(**config)
    x = torch.randn((2, 12, config["input_channels"], config["input_size"], config["input_size"]))
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
