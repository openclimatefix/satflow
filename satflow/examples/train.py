import torch
from satflow.data.datasets import SatFlowDataset
from satflow.core.utils import load_config
import webdataset as wds

config = load_config("/home/bieker/Development/satflow/satflow/configs/base.yaml")


dset = wds.WebDataset("/run/media/bieker/data/EUMETSAT/satflow-flow-{00000...00067}.tar").shuffle(20)
dataset = SatFlowDataset(dset, config=config)

