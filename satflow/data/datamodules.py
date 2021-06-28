import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional
import webdataset as wds
from satflow.data.datasets import SatFlowDataset
import os


class SatFlowDataModule(pl.LightningDataModule):

    def __init__(self, config, batch_size, data_dir: str = './'):
        super().__init__()
        self.data_dir = data_dir
        self.config = config
        self.batch_size = batch_size

    def prepare_data(self):
        # download
        pass

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            train_dset = wds.WebDataset(os.path.join(self.data_dir,self.config["sources"]["train"]))
            val_dset = wds.WebDataset(os.path.join(self.data_dir,self.config["sources"]["val"]))
            if self.config.get("shuffle_data", False):
                # Add shuffling, each sample is still quite large, so too many examples ends up running out of ram
                train_dset = train_dset.shuffle(10)
            self.train_dataset = SatFlowDataset([train_dset], config=self.config, train=True)
            self.val_dataset = SatFlowDataset([val_dset], config=self.config, train=False)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            test_dset = wds.WebDataset(os.path.join(self.data_dir,self.config["sources"]["test"]))
            self.test_dataset = SatFlowDataset([test_dset], config=self.config, train=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.config["num_workers"])

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.config["num_workers"])

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.config["num_workers"])