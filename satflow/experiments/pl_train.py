import torch
from satflow.data.datasets import SatFlowDataset
from satflow.core.utils import load_config, make_logger
import webdataset as wds
from torch.utils.data import DataLoader
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import satflow.models
import torch.nn.functional as F
from satflow.models import get_model
from satflow.core.training_utils import get_loaders, get_args, setup_experiment
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.plugins import DeepSpeedPlugin



logger = make_logger("satflow.train")


def run_experiment(args):
    config = setup_experiment(args)
    config["device"] = (
        ("cuda" if torch.cuda.is_available() else "cpu") if not args.with_cpu else "cpu"
    )

    # Load Model
    model = (
        get_model(config["model"]["name"])
            .from_config(config["model"])
    )
    criterion = F.mse_loss
    # Load Datasets
    loaders = get_loaders(config["dataset"])

    # Get batch size that fits in memory

    #trainer = Trainer(auto_scale_batch_size='power')
    #trainer.tune(model)
    #tuner = Tuner(trainer)

    #new_batch_size = tuner.scale_batch_size(model)
    trainer = Trainer(gpus=1, plugins='deepspeed_stage_2_offload', precision=16)

    trainer.fit(model, loaders['train'], loaders['test'])


if __name__ == "__main__":
    args = get_args()

    run_experiment(args)
