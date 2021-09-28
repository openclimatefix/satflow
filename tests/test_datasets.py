from satflow.data.datasets import SatFlowDataset
from nowcasting_dataset.consts import (
    SATELLITE_DATA,
    SATELLITE_X_COORDS,
    SATELLITE_Y_COORDS,
    SATELLITE_DATETIME_INDEX,
    NWP_DATA,
    NWP_Y_COORDS,
    NWP_X_COORDS,
    NWP_TARGET_TIME,
    DATETIME_FEATURE_NAMES,
)
import os
from nowcasting_dataset.config.load import load_yaml_configuration


def test_dataset():

    # load configuration, this can be changed to a different filename as needed
    filename = os.path.join("tests", "config", "test.yaml")
    config = load_yaml_configuration(filename)
    train_dataset = SatFlowDataset(
        1,
        "tests/",
        "tests/",
        cloud="local",
        required_keys=[
            NWP_DATA,
            SATELLITE_DATA,
            SATELLITE_DATETIME_INDEX,
            NWP_TARGET_TIME,
        ],
        history_minutes=10,
        forecast_minutes=10,
        configuration=config,
    )

    sample, target = next(iter(train_dataset))
    assert sample[SATELLITE_DATA].shape[1] == 3
    assert target[SATELLITE_DATA].shape[1] == 2
