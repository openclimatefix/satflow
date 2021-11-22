import logging
import os
from typing import List, Optional, Tuple, Union

import torch
from nowcasting_dataset.config.load import load_yaml_configuration
from nowcasting_dataset.consts import (
    DATETIME_FEATURE_NAMES,
    NWP_DATA,
    NWP_X_COORDS,
    NWP_Y_COORDS,
    SATELLITE_DATA,
    SATELLITE_DATETIME_INDEX,
    SATELLITE_X_COORDS,
    SATELLITE_Y_COORDS,
    TOPOGRAPHIC_DATA,
    TOPOGRAPHIC_X_COORDS,
    TOPOGRAPHIC_Y_COORDS,
)
from nowcasting_dataset.dataset.datasets import worker_init_fn
from pytorch_lightning import LightningDataModule

from satflow.data.datasets import SatFlowDataset

_LOG = logging.getLogger(__name__)
_LOG.setLevel(logging.DEBUG)


class SatFlowDataModule(LightningDataModule):
    """
    Example of LightningDataModule for NETCDF dataset.
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))
    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        temp_path: str = ".",
        n_train_data: int = 24900,
        n_val_data: int = 1000,
        cloud: str = "aws",
        num_workers: int = 8,
        pin_memory: bool = True,
        configuration_filename="satflow/configs/local.yaml",
        fake_data: bool = False,
        required_keys: Union[Tuple[str], List[str]] = [
            NWP_DATA,
            NWP_X_COORDS,
            NWP_Y_COORDS,
            SATELLITE_DATA,
            SATELLITE_X_COORDS,
            SATELLITE_Y_COORDS,
            SATELLITE_DATETIME_INDEX,
            TOPOGRAPHIC_DATA,
            TOPOGRAPHIC_X_COORDS,
            TOPOGRAPHIC_Y_COORDS,
        ]
        + list(DATETIME_FEATURE_NAMES),
        history_minutes: Optional[int] = None,
        forecast_minutes: Optional[int] = None,
    ):
        """
        fake_data: random data is created and used instead. This is useful for testing
        """
        super().__init__()

        self.temp_path = temp_path
        self.configuration = load_yaml_configuration(configuration_filename)
        self.cloud = cloud
        self.n_train_data = n_train_data
        self.n_val_data = n_val_data
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.fake_data = fake_data
        self.required_keys = required_keys
        self.forecast_minutes = forecast_minutes
        self.history_minutes = history_minutes

        self.dataloader_config = dict(
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            prefetch_factor=8,
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
            # Disable automatic batching because dataset
            # returns complete batches.
            batch_size=None,
        )

    def train_dataloader(self):
        if self.fake_data:
            train_dataset = FakeDataset(
                history_minutes=self.history_minutes, forecast_minutes=self.forecast_minutes
            )
        else:
            train_dataset = SatFlowDataset(
                self.n_train_data,
                os.path.join(self.configuration.output_data.filepath, "train"),
                os.path.join(self.temp_path, "train"),
                configuration=self.configuration,
                cloud=self.cloud,
                required_keys=self.required_keys,
                history_minutes=self.history_minutes,
                forecast_minutes=self.forecast_minutes,
            )

        return torch.utils.data.DataLoader(train_dataset, **self.dataloader_config)

    def val_dataloader(self):
        if self.fake_data:
            val_dataset = FakeDataset(
                history_minutes=self.history_minutes, forecast_minutes=self.forecast_minutes
            )
        else:
            val_dataset = SatFlowDataset(
                self.n_val_data,
                os.path.join(self.configuration.output_data.filepath, "validation"),
                os.path.join(self.temp_path, "validation"),
                configuration=self.configuration,
                cloud=self.cloud,
                required_keys=self.required_keys,
                history_minutes=self.history_minutes,
                forecast_minutes=self.forecast_minutes,
            )

        return torch.utils.data.DataLoader(val_dataset, **self.dataloader_config)

    def test_dataloader(self):
        if self.fake_data:
            test_dataset = FakeDataset(
                history_minutes=self.history_minutes, forecast_minutes=self.forecast_minutes
            )
        else:
            # TODO need to change this to a test folder
            test_dataset = SatFlowDataset(
                self.n_val_data,
                os.path.join(self.configuration.output_data.filepath, "test"),
                os.path.join(self.temp_path, "test"),
                configuration=self.configuration,
                cloud=self.cloud,
                required_keys=self.required_keys,
                history_minutes=self.history_minutes,
                forecast_minutes=self.forecast_minutes,
            )

        return torch.utils.data.DataLoader(test_dataset, **self.dataloader_config)


class FakeDataset(torch.utils.data.Dataset):
    """Fake dataset."""

    def __init__(
        self,
        batch_size=32,
        width=16,
        height=16,
        number_sat_channels=12,
        length=10,
        history_minutes=30,
        forecast_minutes=30,
    ):
        self.batch_size = batch_size
        if history_minutes is None or forecast_minutes is None:
            history_minutes = 30  # Half an hour
            forecast_minutes = 240  # 4 hours
        self.history_steps = history_minutes // 5
        self.forecast_steps = forecast_minutes // 5
        self.seq_length = self.history_steps + 1
        self.width = width
        self.height = height
        self.number_sat_channels = number_sat_channels
        self.length = length

    def __len__(self):
        return self.length

    def per_worker_init(self, worker_id: int):
        pass

    def __getitem__(self, idx):

        x = {
            SATELLITE_DATA: torch.randn(
                self.batch_size, self.seq_length, self.width, self.height, self.number_sat_channels
            ),
            NWP_DATA: torch.randn(self.batch_size, 10, self.seq_length, 2, 2),
            "hour_of_day_sin": torch.randn(self.batch_size, self.seq_length),
            "hour_of_day_cos": torch.randn(self.batch_size, self.seq_length),
            "day_of_year_sin": torch.randn(self.batch_size, self.seq_length),
            "day_of_year_cos": torch.randn(self.batch_size, self.seq_length),
        }

        # add fake x and y coords, and make sure they are sorted
        x[SATELLITE_X_COORDS], _ = torch.sort(torch.randn(self.batch_size, self.seq_length))
        x[SATELLITE_Y_COORDS], _ = torch.sort(
            torch.randn(self.batch_size, self.seq_length), descending=True
        )

        # add sorted (fake) time series
        x[SATELLITE_DATETIME_INDEX], _ = torch.sort(torch.randn(self.batch_size, self.seq_length))

        y = {
            SATELLITE_DATA: torch.randn(
                self.batch_size,
                self.forecast_steps,
                self.width,
                self.height,
                self.number_sat_channels,
            ),
        }
        return x, y
