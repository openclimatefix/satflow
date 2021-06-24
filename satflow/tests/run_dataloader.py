from satflow.data.datasets import SatFlowDataset
from satflow.core.training_utils import get_args, get_loaders, setup_experiment
from satflow.core.utils import load_config, make_logger
from satflow.data.datasets import SatFlowDataset
from satflow.models import get_model
from satflow.core.utils import load_config

config = load_config("/home/jacob/Development/satflow/satflow/configs/datamodule/test_dataloaders.yaml")

config["num_workers"] = 1
# Load Datasets
loaders = get_loaders(config)

for b in loaders['train']:
    print(b)
    exit()
