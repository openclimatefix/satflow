import numpy as np
from satflow.models import MetNet, Perceiver, NowcastingGAN
from satflow.models.base import list_models
import yaml
import torch
import pytest


def load_config(config_file):
    with open(config_file, "r") as cfg:
        return yaml.load(cfg, Loader=yaml.FullLoader)


def test_perceiver_creation():
    config = load_config("satflow/configs/model/perceiver.yaml")
    config.pop("_target_")  # This is only for Hydra
    model = Perceiver(**config)
    x = {
        "timeseries": torch.randn(
            (2, 6, config["input_size"], config["input_size"], config["sat_channels"])
        ),
        "base": torch.randn((2, config["input_size"], config["input_size"], 4)),
        "forecast_time": torch.randn(2, config["forecast_steps"], 1),
    }
    query = torch.randn((2, config["input_size"] * config["sat_channels"], config["queries_dim"]))
    model.eval()
    with torch.no_grad():
        out = model(x, query=query)
    # MetNet creates predictions for the center 1/4th
    assert out.size() == (
        2,
        config["forecast_steps"],
        config["sat_channels"] * config["input_size"],
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
    # MetNet expects original HxW to be 4x the input size
    x = torch.randn(
        (2, 12, config["input_channels"], config["input_size"] * 4, config["input_size"] * 4)
    )
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


@pytest.mark.parametrize("model_name", list_models())
def test_create_model(model_name):
    """
    Test that create model works for all registered models
    Args:
        model_name:

    Returns:

    """
    pass
