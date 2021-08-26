import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional
import webdataset as wds
from satflow.data.datasets import SatFlowDataset, CloudFlowDataset, FileDataset, PerceiverDataset
import os
from glob import glob


def is_streaming(pattern):
    """
    Determine whether Webdataset is being streamed in or not

    Very simple for now and kinda hacky
    Args:
        pattern:

    Returns:

    """
    if "pipe" in pattern:
        return True
    else:
        return False


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


class FileDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: dict,
        sources: dict,
        batch_size: int = 2,
        shuffle: int = 0,
        data_dir: str = "./",
        num_workers: int = 1,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.config = config
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sources = sources
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.training_dataloader_ref = None

    def prepare_data(self):
        # download
        pass

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = FileDataset(
                os.path.join(self.data_dir, self.sources["train"]),
                use_image=self.config.get("use_image", True),
                train=True,
            )
            self.val_dataset = FileDataset(
                os.path.join(self.data_dir, self.sources["val"]),
                use_image=self.config.get("use_image", True),
                train=False,
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = FileDataset(
                os.path.join(self.data_dir, self.sources["test"]),
                use_image=self.config.get("use_image", True),
                train=False,
            )

    def train_dataloader(self):
        if self.training_dataloader_ref:
            return self.training_dataloader_ref

        training_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            shuffle=True,
        )
        self.training_dataloader_ref = training_dataloader

        return self.training_dataloader_ref

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )


class MaskFlowDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: dict,
        sources: dict,
        batch_size: int = 2,
        shuffle: int = 0,
        data_dir: str = "./",
        num_workers: int = 1,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.config = config
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sources = sources
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.training_dataloader_ref = None

    def prepare_data(self):
        # download
        pass

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            train_dset = wds.WebDataset(os.path.join(self.data_dir, self.sources["train"]))
            val_dset = wds.WebDataset(os.path.join(self.data_dir, self.sources["val"]))
            if self.shuffle > 0:
                # Add shuffling, each sample is still quite large, so too many examples ends up running out of ram
                train_dset = train_dset.shuffle(self.shuffle)
            self.train_dataset = CloudFlowDataset([train_dset], config=self.config, train=True)
            self.val_dataset = CloudFlowDataset([val_dset], config=self.config, train=False)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            test_dset = wds.WebDataset(os.path.join(self.data_dir, self.sources["test"]))
            self.test_dataset = CloudFlowDataset([test_dset], config=self.config, train=False)

    def train_dataloader(self):
        # Stores reference and returns it for the reload_dataloaders_every_n_epochs so that the training dataloader keeps
        # iterating through all the data, but the validation dataloader is reset and helps with keeping the same examples
        if self.training_dataloader_ref:
            return self.training_dataloader_ref

        training_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )
        self.training_dataloader_ref = training_dataloader

        return self.training_dataloader_ref

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )


class PerceiverDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: dict,
        sources: dict,
        batch_size: int = 2,
        shuffle: int = 0,
        data_dir: str = "./",
        num_workers: int = 1,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.config = config
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sources = sources
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.training_dataloader_ref = None

    def prepare_data(self):
        # download
        pass

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            train_dset = wds.WebDataset(os.path.join(self.data_dir, self.sources["train"]))
            val_dset = wds.WebDataset(os.path.join(self.data_dir, self.sources["val"]))
            if self.shuffle > 0:
                # Add shuffling, each sample is still quite large, so too many examples ends up running out of ram
                train_dset = train_dset.shuffle(self.shuffle)
            self.train_dataset = PerceiverDataset([train_dset], config=self.config, train=True)
            self.val_dataset = PerceiverDataset([val_dset], config=self.config, train=False)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            test_dset = wds.WebDataset(os.path.join(self.data_dir, self.sources["test"]))
            self.test_dataset = PerceiverDataset([test_dset], config=self.config, train=False)

    def train_dataloader(self):
        if self.training_dataloader_ref:
            return self.training_dataloader_ref

        training_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )
        self.training_dataloader_ref = training_dataloader

        return self.training_dataloader_ref

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )
