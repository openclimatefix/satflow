import torch
from satflow.models.layers import CoordConv


def get_conv_layer(conv_type: str = "standard") -> torch.nn.Module:
    if conv_type == "standard":
        conv2d = torch.nn.Conv2d
    elif conv_type == "coord":
        conv2d = CoordConv
    elif conv_type == "antialiased":
        # TODO Add anti-aliased coordconv here
        conv2d = torch.nn.Conv2d
    else:
        raise ValueError(f"{conv_type} is not a recognized Conv method")
    return conv2d
