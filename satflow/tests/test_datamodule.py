# NIR1.6, VIS0.8 and VIS0.6 RGB for near normal view
import numpy as np
from satflow.data.datamodules import SatFlowDataModule
import yaml


def load_config(config_file):
    with open(config_file, "r") as cfg:
        return yaml.load(cfg, Loader=yaml.FullLoader)


def check_channels(config):
    channels = len(config.get("bands", []))
    print(f"Channels: (Bands) {channels}")
    channels = channels + 1 if config.get("use_mask", False) else channels
    print(f"Channels: (Use Mask) {channels}")
    channels = (
        channels + 3
        if config.get("use_time", False) and not config.get("time_aux", False)
        else channels
    )
    print(f"Channels: (Use Time) {channels}")
    if config.get("time_as_channels", False):
        # Calc number of channels + inital ones
        channels = channels * (config["num_timesteps"] + 1)
    print(f"Channels: (Time as Channels) {channels}")
    channels = channels + 1 if config.get("use_topo", False) else channels
    print(f"Channels: (Use Topo) {channels}")
    channels = channels + 3 if config.get("use_latlon", False) else channels
    print(f"Channels: (Use Latlon) {channels}")
    channels = channels + 2 if config.get("add_pixel_coords", False) else channels
    print(f"Channels: (Add Pixel Coordinates) {channels}")
    channels = channels + 1 if config.get("add_polar_coords", False) else channels
    print(f"Channels: (Add Polar Coordinates) {channels}")
    return channels


def test_satflow_cloudmask():
    config = load_config("satflow/tests/configs/satflow.yaml")
    cloudflow = SatFlowDataModule(**config)
    cloudflow.setup()
    data = next(iter(cloudflow.train_dataloader()))
    x, y = data
    channels = check_channels(config["config"])
    assert x.shape == (12, 13, channels, 128, 128)
    assert y.shape == (12, 24, 1, 128, 128)


def test_satflow_all():
    config = load_config("satflow/tests/configs/satflow_all.yaml")
    cloudflow = SatFlowDataModule(**config)
    cloudflow.setup()
    data = next(iter(cloudflow.train_dataloader()))
    x, image = data
    channels = check_channels(config["config"])
    assert x.shape == (12, 13, channels, 128, 128)
    assert image.shape == (12, 24, 12, 128, 128)
    assert not np.allclose(image[0].numpy(), image[-1].numpy())


def test_satflow_large():
    config = load_config("satflow/tests/configs/satflow_large.yaml")
    cloudflow = SatFlowDataModule(**config)
    cloudflow.setup()
    data = next(iter(cloudflow.train_dataloader()))
    x, image = data
    channels = check_channels(config["config"])
    assert x.shape == (12, 13, channels, 256, 256)
    assert image.shape == (12, 24, 12, 256, 256)
    assert not np.allclose(image[0].numpy(), image[-1].numpy())


def test_satflow_crop():
    config = load_config("satflow/tests/configs/satflow_crop.yaml")
    cloudflow = SatFlowDataModule(**config)
    cloudflow.setup()
    data = next(iter(cloudflow.train_dataloader()))
    x, image = data
    channels = check_channels(config["config"])
    assert x.shape == (12, 13, channels, 256, 256)
    assert image.shape == (12, 24, 12, 64, 64)
    assert not np.allclose(image[0].numpy(), image[-1].numpy())
    assert not np.allclose(x[0].numpy(), x[-1].numpy())


def test_satflow_channels():
    config = load_config("satflow/tests/configs/satflow_channels.yaml")
    cloudflow = SatFlowDataModule(**config)
    cloudflow.setup()
    data = next(iter(cloudflow.train_dataloader()))
    x, y = data
    channels = check_channels(config["config"])
    assert x.shape == (12, 13, channels, 128, 128)
    assert y.shape == (12, 24, 1, 128, 128)


def test_satflow_time_channels():
    config = load_config("satflow/tests/configs/satflow_time_channels.yaml")
    cloudflow = SatFlowDataModule(**config)
    cloudflow.setup()
    data = next(iter(cloudflow.train_dataloader()))
    x, y = data
    channels = check_channels(config["config"])
    assert x.shape[1] == channels
    assert x.shape == (12, 156, 128, 128)
    assert y.shape == (12, 24, 128, 128)


def test_satflow_time_channels_all():
    config = load_config("satflow/tests/configs/satflow_time_channels_all.yaml")
    cloudflow = SatFlowDataModule(**config)
    cloudflow.setup()
    data = next(iter(cloudflow.train_dataloader()))
    x, image = data
    channels = check_channels(config["config"])
    assert x.numpy().shape == (12, channels, 128, 128)
    assert image.shape == (12, 12 * 24, 128, 128)
    assert not np.allclose(image[0].numpy(), image[-1].numpy())


def test_cloudflow():
    config = load_config("satflow/tests/configs/satflow.yaml")
    cloudflow = SatFlowDataModule(**config)
    cloudflow.setup()
    data = next(iter(cloudflow.train_dataloader()))
    x, y = data
    assert x.numpy().shape == (12, 13, 12, 128, 128)
    assert y.numpy().shape == (12, 24, 1, 128, 128)
