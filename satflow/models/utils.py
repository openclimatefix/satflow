import torch
import torch.nn.functional as F
from satflow.models.layers import CoordConv
from satflow.models.losses import FocalLoss, MS_SSIMLoss, SSIMLoss


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


def get_loss(loss: str = "mse", **kwargs) -> torch.nn.Module:
    if isinstance(loss, torch.nn.Module):
        return loss
    assert loss in [
        "mse",
        "bce",
        "binary_crossentropy",
        "crossentropy",
        "focal",
        "ssim",
        "ms_ssim",
        "l1",
    ]
    if loss == "mse":
        criterion = F.mse_loss
    elif loss in ["bce", "binary_crossentropy", "crossentropy"]:
        criterion = F.nll_loss
    elif loss in ["focal"]:
        criterion = FocalLoss()
    elif loss in ["ssim"]:
        criterion = SSIMLoss(data_range=1.0, size_average=True, **kwargs)
    elif loss in ["ms_ssim"]:
        criterion = MS_SSIMLoss(data_range=1.0, size_average=True, **kwargs)
    elif loss in ["l1"]:
        criterion = torch.nn.L1Loss()
    else:
        raise ValueError(f"loss {loss} not recognized")
    return criterion
