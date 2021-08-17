import torch
import torch.nn.functional as F
from satflow.models.utils import extract_image_patches
import numpy as np


class Conv2DDownsample(torch.nn.Module):
    def __init__(
        self,
        num_layers: int = 1,
        input_channels: int = 12,
        output_channels: int = 64,
        use_batchnorm: bool = True,
    ):
        """
        Constructs a Conv2DDownsample model

        Args:
            num_layers: Number of conv -> maxpool layers
            output_channels: Number of output channels
            input_channels: Number of input channels to first layer
            use_batchnorm: Whether to use Batch Norm
        """
        super().__init__()
        self.layers = torch.nn.ModuleList()
        conv1 = torch.nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=(7, 7),
            stride=(2, 2),
            bias=False,
            padding="same",
        )
        self.layers.append(conv1)
        if use_batchnorm:
            self.layers.append(torch.nn.BatchNorm2d(num_features=output_channels))
        self.layers.append(torch.nn.ReLU())
        self.layers.append(torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding="same"))
        for _ in range(num_layers - 1):
            conv = torch.nn.Conv2d(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=(7, 7),
                stride=(2, 2),
                bias=False,
                padding="same",
            )
            self.layers.append(conv)
            if use_batchnorm:
                self.layers.append(torch.nn.BatchNorm2d(num_features=output_channels))
            self.layers.append(torch.nn.ReLU())
            self.layers.append(
                torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding="same")
            )

    def forward(self, x):
        return self.layers.forward(x)


class Conv2DUpsample(torch.nn.Module):
    def __init__(self, input_channels: int = 12, output_channels: int = 12):
        """
        Upsamples 4x using 2 2D transposed convolutions
        Args:
            input_channels: Input channels to the first layer
            output_channels: Number of output channels
        """

        self.transpose_conv1 = torch.nn.ConvTranspose2d(
            in_channels=input_channels,
            out_channels=output_channels * 2,
            kernel_size=(4, 4),
            stride=(2, 2),
        )
        self.transpose_conv2 = torch.nn.ConvTranspose2d(
            in_channels=output_channels * 2,
            out_channels=output_channels,
            kernel_size=(4, 4),
            stride=(2, 2),
        )

    def forward(self, x):
        x = self.transpose_conv1(x)
        x = F.relu(x)
        x = self.transpose_conv2(x)
        return x


class Conv3DUpsample(torch.nn.Module):
    def __init__(
        self,
        input_channels: int = 12,
        output_channels: int = 12,
        num_temporal_upsamples: int = 2,
        num_space_upsamples: int = 4,
    ):
        """
        Simple convolutional auto-encoder
        Args:
            output_channels:
            num_temporal_upsamples:
            num_space_upsamples:
        """

        temporal_stride = 2
        space_stride = 2
        num_upsamples = max(num_space_upsamples, num_temporal_upsamples)
        self.layers = torch.nn.ModuleList()
        for i in range(num_upsamples):
            if i >= num_temporal_upsamples:
                temporal_stride = 1
            if i >= num_space_upsamples:
                space_stride = 1

            channels = output_channels * pow(2, num_upsamples - 1 - i)
            conv = torch.nn.ConvTranspose3d(
                in_channels=input_channels,
                out_channels=channels,
                stride=(temporal_stride, space_stride, space_stride),
                kernel_size=(4, 4, 4),
            )
            self.layers.append(conv)
            if i != num_upsamples - i:
                self.layers.append(torch.nn.ReLU())

    def forward(self, x):
        return self.layers.forward(x)


class ImageEncoder(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass
