import torch
from satflow.data.datasets import SatFlowDataset
from satflow.core.utils import load_config
import webdataset as wds
from torch.utils.data import DataLoader

config = load_config("/home/bieker/Development/satflow/satflow/configs/base.yaml")
print(config)

dset = wds.WebDataset("/run/media/bieker/data/EUMETSAT/satflow-flow-{00000..0060}.tar")
dataset = SatFlowDataset([dset], config=config['training'])

dataloader = DataLoader(dataset, num_workers=4, batch_size=16)

for data in dataloader:
    image, target_image, target_mask = data
    print(f"Shapes Image: {image.shape}, T Image: {target_image.shape} T Mask: {target_mask.shape}")
    exit()
