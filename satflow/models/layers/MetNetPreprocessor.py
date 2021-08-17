import torch
from torch import nn
import torchvision
import torch.nn.functional as F


class MetNetPreprocessor(nn.Module):
    def __init__(self, sat_channels: int = 12, crop_size: int = 256, use_space2depth: bool = True):
        """
        Performs the MetNet preprocessing of mean pooling Sat channels, followed by
        concatenating the center crop and mean pool

        In the paper, the radar data is space2depth'd, while satellite channel is mean pooled, but for this different
        task, we choose to do either option for satellites
        Args:
            sat_channels: Number of satellite channels
            crop_size: Center crop size
            use_space2depth: Whether to use space2depth on satellite channels, or mean pooling, like in paper

        """
        super().__init__()
        self.sat_channels = sat_channels

        # Split off sat + mask channels into own image, and the rest, which we just take a center crop
        # For this,
        self.sat_downsample = (
            torch.nn.PixelUnshuffle(downscale_factor=2)
            if use_space2depth
            else torch.nn.AvgPool3d(kernel_size=(1, 2, 2))
        )
        self.center_crop = torchvision.transforms.CenterCrop(size=crop_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sat_channels = x[:, :, : self.sat_channels, :, :]
        other_channels = x[:, :, self.sat_channels :, :, :]
        sat_channels = self.sat_downsample(sat_channels)
        other_channels = torchvision.transforms.CenterCrop(size=other_channels.size()[-1] // 2)(
            other_channels
        )  # center crop to same as downsample
        # In paper, satellite and radar data is concatenated here
        # We are just going to skip that bit

        sat_center = self.center_crop(sat_channels)
        sat_mean = F.avg_pool3d(sat_channels, (1, 2, 2))
        other_channels = self.center_crop(other_channels)
        # All the same size now, so concatenate together, already have time, lat/long, and elevation image
        x = torch.cat([sat_center, sat_mean, other_channels], dim=2)
        return x
