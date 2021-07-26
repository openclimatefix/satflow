import torch

from satflow.data.datasets import SatFlowDataset
import numpy as np
from torch.utils.data import DataLoader
from satflow.data.datasets import SatFlowDataset, CloudFlowDataset
import webdataset as wds
import yaml


def load_config(config_file):
    with open(config_file, "r") as cfg:
        return yaml.load(cfg, Loader=yaml.FullLoader)["config"]


config = load_config("preprocess.yaml")

train = wds.WebDataset("/run/media/jacob/data/satflow-flow-144-tiled-{00001..00105}.tar")
val = wds.WebDataset("/run/media/jacob/data/satflow-flow-144-tiled-{00106..00149}.tar")

t_flow = SatFlowDataset([train], config, train=True)
v_flow = SatFlowDataset([val], config, train=False)

training_dataloader = DataLoader(
    t_flow,
    batch_size=1,
    pin_memory=True,
    num_workers=16,
)
val_dataloader = DataLoader(
    v_flow,
    batch_size=1,
    pin_memory=True,
    num_workers=8,
)

output_dir = "/run/media/jacob/data/satflow_prev6_skip3_notime/"

for i, data in enumerate(training_dataloader):
    if i > 40000:
        break
for i, data in enumerate(val_dataloader):
    image, target_image, target_mask = data
    image = image[0].numpy()
    target_image = target_image[0].numpy()
    target_mask = target_mask[0].numpy()
    np.savez_compressed(
        output_dir + f"val_{i}.npz", images=image, future_images=target_image, masks=target_mask
    )
    if i > 10000:
        break
