import einops
import numpy as np
import torch

from satflow.models.layers import CoordConv


def get_conv_layer(conv_type: str = "standard") -> torch.nn.Module:
    if conv_type == "standard":
        conv_layer = torch.nn.Conv2d
    elif conv_type == "coord":
        conv_layer = CoordConv
    elif conv_type == "antialiased":
        # TODO Add anti-aliased coordconv here
        conv_layer = torch.nn.Conv2d
    elif conv_type == "3d":
        conv_layer = torch.nn.Conv3d
    else:
        raise ValueError(f"{conv_type} is not a recognized Conv method")
    return conv_layer


def reverse_space_to_depth(
    frames: np.ndarray, temporal_block_size: int = 1, spatial_block_size: int = 1
) -> np.ndarray:
    """Reverse space to depth transform."""
    if len(frames.shape) == 4:
        return einops.rearrange(
            frames,
            "b h w (dh dw c) -> b (h dh) (w dw) c",
            dh=spatial_block_size,
            dw=spatial_block_size,
        )
    elif len(frames.shape) == 5:
        return einops.rearrange(
            frames,
            "b t h w (dt dh dw c) -> b (t dt) (h dh) (w dw) c",
            dt=temporal_block_size,
            dh=spatial_block_size,
            dw=spatial_block_size,
        )
    else:
        raise ValueError(
            "Frames should be of rank 4 (batch, height, width, channels)"
            " or rank 5 (batch, time, height, width, channels)"
        )


def space_to_depth(
    frames: np.ndarray, temporal_block_size: int = 1, spatial_block_size: int = 1
) -> np.ndarray:
    """Space to depth transform."""
    if len(frames.shape) == 4:
        return einops.rearrange(
            frames,
            "b (h dh) (w dw) c -> b h w (dh dw c)",
            dh=spatial_block_size,
            dw=spatial_block_size,
        )
    elif len(frames.shape) == 5:
        return einops.rearrange(
            frames,
            "b (t dt) (h dh) (w dw) c -> b t h w (dt dh dw c)",
            dt=temporal_block_size,
            dh=spatial_block_size,
            dw=spatial_block_size,
        )
    else:
        raise ValueError(
            "Frames should be of rank 4 (batch, height, width, channels)"
            " or rank 5 (batch, time, height, width, channels)"
        )
