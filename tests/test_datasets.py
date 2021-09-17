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


def test_dataset():
    train_dataset = SatFlowDataset(
        1,
        "./",
        "./",
        cloud="local",
        required_keys=[
            NWP_DATA,
            SATELLITE_DATA,
            SATELLITE_DATETIME_INDEX,
            NWP_TARGET_TIME,
        ],
        history_minutes=10,
        forecast_minutes=10,
    )

    sample, target = next(iter(train_dataset))
    print(sample.keys())
