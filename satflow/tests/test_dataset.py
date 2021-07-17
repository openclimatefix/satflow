# NIR1.6, VIS0.8 and VIS0.6 RGB for near normal view
import numpy as np
from satflow.data.datasets import SatFlowDataset, CloudFlowDataset
import webdataset as wds
import yaml


def load_config(config_file):
    with open(config_file, "r") as cfg:
        return yaml.load(cfg, Loader=yaml.FullLoader)["config"]


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
    dataset = wds.WebDataset("datasets/satflow-test.tar")
    config = load_config("satflow/tests/configs/satflow.yaml")
    channels = check_channels(config)
    cloudflow = SatFlowDataset([dataset], config)
    data = next(iter(cloudflow))
    x, y = data
    assert x.shape == (13, channels, 128, 128)
    assert y.shape == (24, 1, 128, 128)


def test_satflow_all():
    dataset = wds.WebDataset("datasets/satflow-test.tar")
    config = load_config("satflow/tests/configs/satflow_all.yaml")
    cloudflow = SatFlowDataset([dataset], config)
    data = next(iter(cloudflow))
    x, image, y = data
    channels = check_channels(config)
    assert x.shape == (13, channels, 128, 128)
    assert y.shape == (24, 1, 128, 128)
    assert image.shape == (24, 12, 128, 128)
    assert not np.allclose(image[0], image[-1])
    assert not np.allclose(y[0], y[-1])


def test_satflow_large():
    dataset = wds.WebDataset("datasets/satflow-test.tar")
    config = load_config("satflow/tests/configs/satflow_large.yaml")
    cloudflow = SatFlowDataset([dataset], config)
    data = next(iter(cloudflow))
    x, image, y = data
    channels = check_channels(config)
    assert x.shape == (13, channels, 256, 256)
    assert y.shape == (24, 1, 256, 256)
    assert image.shape == (24, 12, 256, 256)
    assert not np.allclose(image[0], image[-1])
    assert not np.allclose(y[0], y[-1])


def test_satflow_crop():
    dataset = wds.WebDataset("datasets/satflow-test.tar")
    config = load_config("satflow/tests/configs/satflow_crop.yaml")
    cloudflow = SatFlowDataset([dataset], config)
    data = next(iter(cloudflow))
    x, image, y = data
    channels = check_channels(config)
    assert x.shape == (13, channels, 256, 256)
    assert y.shape == (24, 1, 64, 64)
    assert image.shape == (24, 12, 64, 64)
    assert not np.allclose(image[0], image[-1])
    assert not np.allclose(y[0], y[-1])


def test_satflow_channels():
    dataset = wds.WebDataset("datasets/satflow-test.tar")
    config = load_config("satflow/tests/configs/satflow_channels.yaml")
    cloudflow = SatFlowDataset([dataset], config)
    data = next(iter(cloudflow))
    x, y = data
    channels = check_channels(config)
    assert x.shape == (13, channels, 128, 128)
    assert y.shape == (24, 1, 128, 128)


def test_satflow_time_channels():
    dataset = wds.WebDataset("datasets/satflow-test.tar")
    config = load_config("satflow/tests/configs/satflow_time_channels.yaml")
    cloudflow = SatFlowDataset([dataset], config)
    data = next(iter(cloudflow))
    x, y = data
    channels = check_channels(config)
    assert x.shape[0] == channels
    assert x.shape == (156, 128, 128)
    assert y.shape == (24, 128, 128)


def test_satflow_time_channels_all():
    dataset = wds.WebDataset("datasets/satflow-test.tar")
    config = load_config("satflow/tests/configs/satflow_time_channels_all.yaml")
    cloudflow = SatFlowDataset([dataset], config)
    data = next(iter(cloudflow))
    x, image, y = data
    channels = check_channels(config)
    assert x.shape == (channels, 128, 128)
    assert y.shape == (24, 128, 128)
    assert image.shape == (12 * 24, 128, 128)
    assert not np.allclose(image[0], image[-1])
    assert not np.allclose(y[0], y[-1])


def test_cloudflow():
    dataset = wds.WebDataset("datasets/satflow-test.tar")
    config = load_config("satflow/tests/configs/satflow.yaml")
    cloudflow = CloudFlowDataset([dataset], config)
    data = next(iter(cloudflow))
    x, y = data
    assert x.shape == (13, 1, 128, 128)
    assert y.shape == (24, 1, 128, 128)


def test_satflow_all_deterministic_validation():
    dataset = wds.WebDataset("datasets/satflow-test.tar")
    config = load_config("satflow/tests/configs/satflow_all.yaml")
    cloudflow = SatFlowDataset([dataset], config, train=False)
    data = next(iter(cloudflow))
    x, image, y = data
    dataset2 = wds.WebDataset("datasets/satflow-test.tar")
    cloudflow2 = SatFlowDataset([dataset2], config, train=False)
    data = next(iter(cloudflow2))
    x2, image2, y2 = data
    np.testing.assert_almost_equal(x, x2)
    np.testing.assert_almost_equal(image, image2)
    np.testing.assert_almost_equal(y, y2)
    assert not np.allclose(image[0], image[-1])
    assert not np.allclose(y[0], y[-1])


def test_satflow_all_deterministic_validation_restart():
    dataset = wds.WebDataset("datasets/satflow-test.tar")
    config = load_config("satflow/tests/configs/satflow_all.yaml")
    cloudflow = SatFlowDataset([dataset], config, train=False)
    data = next(iter(cloudflow))
    x, image, y = data
    data = next(iter(cloudflow))
    x2, image2, y2 = data
    np.testing.assert_almost_equal(x, x2)
    np.testing.assert_almost_equal(image, image2)
    np.testing.assert_almost_equal(y, y2)
    assert not np.allclose(image[0], image[-1])
    assert not np.allclose(y[0], y[-1])
