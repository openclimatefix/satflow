import argparse

import torch
import webdataset as wds
from torch.utils.data import DataLoader

import satflow.data.datasets
from satflow.core.utils import load_config
from satflow.data.datasets import get_dataset


def get_loaders(config):
    """
    Get Dataloaders for train/test
    Args:
        config: Dict, configuration dictionary for the dataset

    Returns:
        Dict[Dataloader] containing the train and test dataloaders
    """
    print(config)
    train_dset = wds.WebDataset(config["sources"]["train"])
    val_dset = wds.WebDataset(config["sources"]["val"])
    test_dset = wds.WebDataset(config["sources"]["test"])
    train_dataset = get_dataset(config["name"])([train_dset], config=config, train=True)
    val_dataset = get_dataset(config["name"])([val_dset], config=config, train=False)
    test_dataset = get_dataset(config["name"])([test_dset], config=config, train=False)

    train_dataloader = DataLoader(
        train_dataset,
        num_workers=config["num_workers"],
        batch_size=config["batch_size"],
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        num_workers=config["num_workers"],
        batch_size=config["batch_size"],
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        num_workers=config["num_workers"],
        batch_size=config["batch_size"],
        pin_memory=True,
    )

    return {"train": train_dataloader, "val": val_dataloader, "test": test_dataloader}


def setup_experiment(args):
    """
    Sets up the basic logging, etc. common things for running experiments

    Args:
        args: Commandline arguments

    Returns:

    """

    config = load_config(args.config)

    config["dataset"]["num_workers"] = args.num_workers
    return config


def get_args():

    parser = argparse.ArgumentParser(description="SatFlow")

    # cuda
    parser.add_argument(
        "--with_cpu",
        default=False,
        action="store_true",
        help="use CPU in case there's no GPU support",
    )

    # train
    parser.add_argument(
        "-c", "--config", default="./config.yaml", type=str, help="Path to Config File"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher",
    )

    parser.add_argument(
        "-nw",
        "--num_workers",
        type=int,
        default=1,
        help="Number of dataloader workers",
    )

    # Include DeepSpeed configuration arguments
    # parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args
