import torch
from satflow.data.datasets import SatFlowDataset
from satflow.core.utils import load_config
import webdataset as wds
from torch.utils.data import DataLoader

config = load_config("/home/bieker/Development/satflow/satflow/configs/base.yaml")
print(config)

dset = wds.WebDataset("/run/media/bieker/data/EUMETSAT/satflow-flow-{00000..00067}.tar")
dataset = SatFlowDataset(dset, config=config['training'])

dataloader = DataLoader(dataset, num_workers=1, batch_size=1)

for data in dataloader:
    print(data)
    exit()
