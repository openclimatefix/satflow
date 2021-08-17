import torch
import torch.nn.functional as F
from satflow.models.utils import extract_image_patches, space_to_depth, reverse_space_to_depth
import numpy as np
import math


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
    def __init__(
        self,
        input_channels: int = 12,
        prep_type: str = "conv",
        spatial_downsample: int = 4,
        temporal_downsample: int = 1,
        output_channels: int = 64,
        conv_after_patching: bool = False,
        conv2d_use_batchnorm: bool = True,
    ):
        super().__init__()
        self.conv_after_patching = conv_after_patching
        self.prep_type = prep_type
        self.temporal_downsample = temporal_downsample
        self.spatial_downsample = spatial_downsample
        self.output_channels = output_channels

        if prep_type not in ("conv", "patches", "pixels", "conv1x1"):
            raise ValueError("Invalid prep_type!")

        if self.prep_type == "conv":
            # Downsampling with conv is currently restricted
            convnet_num_layers = math.log(spatial_downsample, 4)
            convnet_num_layers_is_int = convnet_num_layers == np.round(convnet_num_layers)
            if not convnet_num_layers_is_int or temporal_downsample != 1:
                raise ValueError(
                    "Only powers of 4 expected for spatial "
                    "and 1 expected for temporal "
                    "downsampling with conv."
                )

            self.convnet = Conv2DDownsample(
                num_layers=int(convnet_num_layers),
                output_channels=output_channels,
                input_channels=input_channels,
                use_batchnorm=conv2d_use_batchnorm,
            )
        elif self.prep_type == "conv1x1":
            assert temporal_downsample == 1, "conv1x1 does not downsample in time."
            self.convnet_1x1 = torch.nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=(1, 1),
                # spatial_downsample is unconstrained for 1x1 convolutions.
                stride=(spatial_downsample, spatial_downsample),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prep_type == "conv":
            if len(x.shape) == 5:
                # Timeseries, do it to each timestep independently
                outs = []
                for i in range(x.shape[1]):
                    outs.append(self.convnet(x[:, i, :, :, :]))
                x = torch.stack(outs, dim=1)
            else:
                x = self.convnet(x)

        elif self.prep_type == "conv1x1":
            if len(x.shape) == 5:
                # Timeseries, do it to each timestep independently
                outs = []
                for i in range(x.shape[1]):
                    outs.append(self.convnet_1x1(x[:, i, :, :, :]))
                x = torch.stack(outs, dim=1)
            else:
                x = self.convnet_1x1(x)

        elif self.prep_type == "patches":

            x = space_to_depth(
                x,
                temporal_block_size=self.temporal_downsample,
                spatial_block_size=self.spatial_downsample,
            )

            # For flow
            if x.ndim == 5 and x.shape[1] == 1:
                x = x.squeeze(axis=1)
        elif self.prep_type == "pixels":
            # If requested, will downsample in simplest way
            if x.ndim == 4:
                x = x[:, :: self.spatial_downsample, :: self.spatial_downsample]
            elif x.ndim == 5:
                x = x[
                    :,
                    :: self.temporal_downsample,
                    :: self.spatial_downsample,
                    :: self.spatial_downsample,
                ]
            else:
                raise ValueError("Unsupported data format for pixels")

        return x


class ImageDecoder(torch.nn.Module):
    def __init__(
        self,
        postprocess_type: str = "pixels",
        spatial_upsample: int = 1,
        temporal_upsample: int = 1,
        output_channels: int = -1,
        input_channels: int = 12,
        input_reshape_size=None,
    ):
        super().__init__()

        if postprocess_type not in ("conv", "patches", "pixels", "raft", "conv1x1"):
            raise ValueError("Invalid postproc_type!")

        # Architecture parameters:
        self.postprocess_type = postprocess_type

        self.temporal_upsample = temporal_upsample
        self.spatial_upsample = spatial_upsample
        self.input_reshape_size = input_reshape_size

        if self.postprocess_type == "pixels":
            # No postprocessing.
            if self.temporal_upsample != 1 or self.spatial_upsample != 1:
                raise ValueError("Pixels postprocessing should not currently upsample.")
        elif self.postprocess_type == "conv1x1":
            assert self._temporal_upsample == 1, "conv1x1 does not upsample in time."
            if output_channels == -1:
                raise ValueError("Expected value for n_outputs")
            self.conv1x1 = torch.nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=(1, 1),
                # spatial_downsample is unconstrained for 1x1 convolutions.
                stride=(self.spatial_upsample, self.spatial_upsample),
            )
        elif self.postprocess_type == "conv":
            if output_channels == -1:
                raise ValueError("Expected value for n_outputs")
            if self.temporal_upsample != 1:

                def int_log2(x):
                    return int(np.round(np.log(x) / np.log(2)))

                self.convnet = Conv3DUpsample(
                    input_channels=input_channels,
                    output_channels=output_channels,
                    num_temporal_upsamples=int_log2(temporal_upsample),
                    num_space_upsamples=int_log2(spatial_upsample),
                )
            else:
                self.convnet = Conv2DUpsample(
                    input_channels=input_channels, output_channels=output_channels
                )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.input_reshape_size is not None:
            inputs = torch.reshape(
                inputs, [inputs.shape[0]] + list(self.input_reshape_size) + [inputs.shape[-1]]
            )

        if self.postprocess_type == "conv" or self.postprocess_type == "raft":
            # Convnet image featurization.
            if len(inputs.shape) == 5 and self.temporal_upsample == 1:
                # Timeseries, do it to each timestep independently
                outs = []
                for i in range(inputs.shape[1]):
                    outs.append(self.convnet(inputs[:, i, :, :, :]))
                inputs = torch.stack(outs, dim=1)
            else:
                inputs = self.convnet(inputs)
        elif self.postprocess_type == "conv1x1":
            inputs = self.conv1x1(inputs)
        elif self.postprocess_type == "patches":
            inputs = reverse_space_to_depth(inputs, self.temporal_upsample, self.spatial_upsample)

        return inputs
