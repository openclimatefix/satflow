import pytest
import torch
import yaml
from nowcasting_dataloader.fake import FakeDataset
from nowcasting_dataset.config import Configuration, InputData
from nowcasting_dataset.consts import NWP_DATA, SATELLITE_DATA, TOPOGRAPHIC_DATA
from nowcasting_utils.models.base import create_model, list_models

from satflow.models import JointPerceiver, LitMetNet, Perceiver


def load_config(config_file):
    with open(config_file, "r") as cfg:
        return yaml.load(cfg, Loader=yaml.FullLoader)


@pytest.fixture
def configuration():
    """Create Configuration object for tests"""
    con = Configuration()
    con.input_data = InputData.set_all_to_defaults()
    con.process.batch_size = 4
    return con


def test_joint_perceiver(configuration):
    config = load_config("satflow/configs/model/perceiver.yaml")
    config.pop("_target_")  # This is only for Hydra
    model = JointPerceiver(**config)
    dataset = FakeDataset(configuration)
    example = next(iter(dataset))
    model(example)
    pass


def test_perceiver_creation():
    config = load_config("satflow/configs/model/perceiver.yaml")
    config.pop("_target_")  # This is only for Hydra
    model = Perceiver(**config)
    x = {
        SATELLITE_DATA: torch.randn(
            (2, 6, config["input_size"], config["input_size"], config["sat_channels"])
        ),
        TOPOGRAPHIC_DATA: torch.randn((2, config["input_size"], config["input_size"], 1)),
        NWP_DATA: torch.randn(
            (2, 6, config["input_size"], config["input_size"], config["nwp_channels"])
        ),
        "forecast_time": torch.randn(2, config["forecast_steps"], 1),
    }
    query = torch.randn((2, config["input_size"] * config["sat_channels"], config["queries_dim"]))
    model.eval()
    with torch.no_grad():
        out = model(x, query=query)
    # MetNet creates predictions for the center 1/4th
    assert out.size() == (
        2,
        config["forecast_steps"] * config["input_size"],
        config["sat_channels"] * config["input_size"],
    )
    assert not torch.isnan(out).any(), "Output included NaNs"


def test_metnet_creation():
    config = load_config("satflow/configs/model/metnet.yaml")
    config.pop("_target_")  # This is only for Hydra
    model = LitMetNet(**config)
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
    assert not torch.isnan(out).any(), "Output included NaNs"


@pytest.mark.parametrize("model_name", list_models())
def test_create_model(model_name):
    """
    Test that create model works for all registered models
    Args:
        model_name:

    Returns:

    """
    # TODO Load options from all configs and make sure they work
    model = create_model(model_name)
    pass


@pytest.mark.skip(
    "Perceiver has changed in SatFlow, doesn't have the same options as the one on HF"
)
def test_load_hf():
    """
    Current only HF model is PerceiverIO, change in future to do all ones
    Returns:

    """
    model = create_model("hf_hub:openclimatefix/perceiver-io")
    pass


@pytest.mark.skip(
    "Perceiver has changed in SatFlow, doesn't have the same options as the one on HF"
)
def test_load_hf_pretrained():
    """
    Current only HF model is PerceiverIO, change in future to do all ones
    Returns:

    """
    model = create_model("hf_hub:openclimatefix/perceiver-io", pretrained=True)
    pass
