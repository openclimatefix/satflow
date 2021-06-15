import torch
from satflow.data.datasets import SatFlowDataset
from satflow.core.utils import load_config
import webdataset as wds
from torch.utils.data import DataLoader
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import satflow.models
import torch.nn.functional as F
from satflow.models import get_model
from satflow.core.training_utils import get_loaders, get_args
import deepspeed


def run_experiment(args):
    config = load_config(args.config)
    config['device'] = ("cuda" if torch.cuda.is_available() else "cpu") if not args.with_cpu else "cpu"

    # Load Model
    model = get_model(config['model']['name']).from_config(config['model']).to(config['device'])
    criterion = F.nll_loss
    # Load Datasets
    loaders = get_loaders(config['dataset'])
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args, model=model, model_parameters=parameters)
    # Run training
    global_iteration = 0
    running_loss = 0.0
    for i, data in enumerate(loaders['train']):
        # get the inputs; data is a list of [inputs, target_image, target_mask]
        inputs, labels = data[0].to(model_engine.local_rank), data[2].to(
            model_engine.local_rank)

        outputs = model_engine(inputs)
        loss = criterion(outputs, labels)

        model_engine.backward(loss)
        model_engine.step()

        # print statistics
        running_loss += loss.item()
        global_iteration += 1
        if global_iteration > config['iterations']:
            break


if __name__ == "__main__":
    args = get_args()
    run_experiment(args)
