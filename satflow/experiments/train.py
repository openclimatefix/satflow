import torch
from satflow.data.datasets import SatFlowDataset
from satflow.core.utils import load_config
import webdataset as wds
from torch.utils.data import DataLoader
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import satflow.models
from satflow.models import get_model
from satflow.core.training_utils import get_loaders

config = load_config("/home/bieker/Development/satflow/satflow/configs/base.yaml")

dset = wds.WebDataset("/run/media/bieker/data/EUMETSAT/satflow-flow-{00000..0067}.tar")
dataset = SatFlowDataset([dset], config=config['training'])

dataloader = DataLoader(dataset, num_workers=6, batch_size=16)

for data in tqdm(dataloader):
    image, target_image, target_mask = data
    print(f"Shapes Image: {image.shape}, T Image: {target_image.shape} T Mask: {target_mask.shape}")

def run_experiment(config):
    # Load Model
    model = get_model(config['model']['name']).from_config(config['model'])



    # Load Datasets
    loaders = get_loaders(config['dataset'])
    # Run training
    global_iteration = 0
    for batch, data in loaders['train']:
        outputs = model(data)
