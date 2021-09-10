import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional
from nowcasting_dataset.dataset import NowcastingDataset, NetCDFDataset
import os


class SatFlowDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: dict,
        sources: dict,
        batch_size: int = 2,
        shuffle: int = 0,
        data_dir: str = "./",
        num_workers: int = 1,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.config = config
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sources = sources
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor

        self.training_dataloader_ref = None

    def prepare_data(self):
        # download
        pass

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            train_dset = wds.WebDataset(
                self.sources["train"]
                if is_streaming(self.sources["train"])
                else os.path.join(self.data_dir, self.sources["train"])
            )
            val_dset = wds.WebDataset(
                self.sources["val"]
                if is_streaming(self.sources["val"])
                else os.path.join(self.data_dir, self.sources["val"])
            )
            if self.shuffle > 0:
                # Add shuffling, each sample is still quite large, so too many examples ends up running out of ram
                train_dset = train_dset.shuffle(self.shuffle)
            self.train_dataset = SatFlowDataset([train_dset], config=self.config, train=True)
            self.val_dataset = SatFlowDataset([val_dset], config=self.config, train=False)
            # This seems necessary for the reload_dataloader to not reload the training_dataloader
            if self.training_dataloader_ref is None:
                training_dataloader = DataLoader(
                    self.train_dataset,
                    batch_size=self.batch_size,
                    pin_memory=self.pin_memory,
                    num_workers=self.num_workers,
                    prefetch_factor=self.prefetch_factor,
                )
                self.training_dataloader_ref = training_dataloader

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            test_dset = wds.WebDataset(
                self.sources["test"]
                if is_streaming(self.sources["test"])
                else os.path.join(self.data_dir, self.sources["test"])
            )
            self.test_dataset = SatFlowDataset([test_dset], config=self.config, train=False)

    def train_dataloader(self):
        return self.training_dataloader_ref

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers // 2,
            prefetch_factor=self.prefetch_factor,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers // 2,
            prefetch_factor=self.prefetch_factor,
        )


import os
from nowcasting_dataset.dataset import NetCDFDataset, worker_init_fn
from typing import Tuple
import logging
import torch
from pytorch_lightning import LightningDataModule


_LOG = logging.getLogger(__name__)
_LOG.setLevel(logging.DEBUG)


def get_dataloaders(
    n_train_data: int = 24900,
    n_validation_data: int = 900,
    cloud: str = "gcp",
    temp_path=".",
    data_path="prepared_ML_training_data/v4/",
) -> Tuple:

    data_module = NetCDFDataModule(
        temp_path=temp_path,
        data_path=data_path,
        cloud=cloud,
        n_train_data=n_train_data,
        n_val_data=n_validation_data,
    )

    train_dataloader = data_module.train_dataloader()
    validation_dataloader = data_module.val_dataloader()

    return train_dataloader, validation_dataloader


class NetCDFDataModule(LightningDataModule):
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
        data_path="prepared_ML_training_data/v4/",
        fake_data: bool = False,
    ):
        """
        fake_data: random data is created and used instead. This is useful for testing
        """
        super().__init__()

        self.temp_path = temp_path
        self.data_path = data_path
        self.cloud = cloud
        self.n_train_data = n_train_data
        self.n_val_data = n_val_data
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.fake_data = fake_data

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
            train_dataset = FakeDataset()
        else:
            train_dataset = NetCDFDataset(
                self.n_train_data,
                os.path.join(self.data_path, "train"),
                os.path.join(self.temp_path, "train"),
                cloud=self.cloud,
            )

        return torch.utils.data.DataLoader(train_dataset, **self.dataloader_config)

    def val_dataloader(self):
        if self.fake_data:
            val_dataset = FakeDataset()
        else:
            val_dataset = NetCDFDataset(
                self.n_val_data,
                os.path.join(self.data_path, "validation"),
                os.path.join(self.temp_path, "validation"),
                cloud=self.cloud,
            )

        return torch.utils.data.DataLoader(val_dataset, **self.dataloader_config)

    def test_dataloader(self):
        if self.fake_data:
            test_dataset = FakeDataset()
        else:
            # TODO need to change this to a test folder
            test_dataset = NetCDFDataset(
                self.n_val_data,
                os.path.join(self.data_path, "validation"),
                os.path.join(self.temp_path, "validation"),
                cloud=self.cloud,
            )

        return torch.utils.data.DataLoader(test_dataset, **self.dataloader_config)


class FakeDataset(torch.utils.data.Dataset):
    """Fake dataset."""

    def __init__(
        self, batch_size=32, seq_length=19, width=16, height=16, number_sat_channels=8, length=10
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
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
            "sat_data": torch.randn(
                self.batch_size, self.seq_length, self.width, self.height, self.number_sat_channels
            ),
            "pv_yield": torch.randn(self.batch_size, self.seq_length, 128),
            "pv_system_id": torch.randn(self.batch_size, 128),
            "nwp": torch.randn(self.batch_size, 10, self.seq_length, 2, 2),
            "hour_of_day_sin": torch.randn(self.batch_size, self.seq_length),
            "hour_of_day_cos": torch.randn(self.batch_size, self.seq_length),
            "day_of_year_sin": torch.randn(self.batch_size, self.seq_length),
            "day_of_year_cos": torch.randn(self.batch_size, self.seq_length),
        }

        # add a nan
        x["pv_yield"][0, 0, :] = float("nan")

        # add fake x and y coords, and make sure they are sorted
        x["sat_x_coords"], _ = torch.sort(torch.randn(self.batch_size, self.seq_length))
        x["sat_y_coords"], _ = torch.sort(
            torch.randn(self.batch_size, self.seq_length), descending=True
        )

        # add sorted (fake) time series
        x["sat_datetime_index"], _ = torch.sort(torch.randn(self.batch_size, self.seq_length))
        x["nwp_target_time"], _ = torch.sort(torch.randn(self.batch_size, self.seq_length))

        return xv
