import torch
import webdataset as wds
import satflow.data.datasets
from satflow.data.datasets import get_dataset
from torch.utils.data import DataLoader


def get_loaders(config):
    """
    Get Dataloaders for train/test
    Args:
        config: Dict, configuration dictionary for the dataset

    Returns:
        Dict[Dataloader] containing the train and test dataloaders
    """

    train_dset = wds.WebDataset(config['train_pattern'])
    test_dset = wds.WebDataset(config["test_pattern"])
    train_dataset = get_dataset(config['name'])([train_dset], config=config['training'], train=True)
    test_dataset = get_dataset(config['name'])([test_dset], config=config['test'], train=False)

    train_dataloader = DataLoader(train_dataset, num_workers=config['num_workers'], batch_size=config['batch_size'])
    test_dataloader = DataLoader(test_dataset, num_workers=config['num_workers'], batch_size=config['batch_size'])

    return {
        "train": train_dataloader,
        "test": test_dataloader
    }

