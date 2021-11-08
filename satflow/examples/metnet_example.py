"""Example of metnet model"""
from satflow.models import LitMetNet
import torch
import urllib.request


def get_input_target(number: int):
    """
    Load a single input

    Args:
        number: input number
    """
    url = f"https://github.com/openclimatefix/satflow/releases/download/v0.0.3/input_{number}.pth"
    filename, headers = urllib.request.urlretrieve(url, filename=f"input_{number}.pth")
    input_data = torch.load(filename)
    return input_data


# Setup the model (need to add loading weights from HuggingFace :)
# 12 satellite channels + 1 Topographic + 3 Lat/Lon + 1 Cloud Mask
# Output Channels: 1 Cloud mask, 12 for Satellite image
model = LitMetNet(input_channels=17, sat_channels=13, input_size=64, out_channels=1)
torch.set_grad_enabled(False)
model.eval()
# The inputs are Tensors of size (Batch, Curr+Prev Timesteps, Channel, Width, Height)
# MetNet uses the last 90min of data, the previous 6 timesteps + Current one
# This gives an input of (Batch, 7, 256, 256, 286), for Satflow, we use (Batch, 7, 17, 256, 256) and do the preprocessing
# in the model

# Data processing from raw satellite to Tensors is described in satflow/examples/create_webdataset.py and satflow/data/datasets.py
# This just takes the output from the Dataloader, which has been stored here

for i in range(11):
    forecast = model(get_input_target(i))
    print(forecast.size())

# Output for this segmentation model is (Batch, 24, 1, 16, 16) for Satflow, MetNet has an output of (Batch, 480, 1, 256, 256)
