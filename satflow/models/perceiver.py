from perceiver_pytorch import PerceiverIO
import torch
from satflow.models.losses import get_loss, NowcastingLoss, GridCellLoss
import pytorch_lightning as pl
import torchvision
from functools import reduce
from typing import List
from satflow.models.base import register_model
from math import pi, log
from einops import rearrange, repeat


@register_model
class Perceiver(pl.LightningModule):
    def __init__(
        self,
        input_channels: int = 12,
        forecast_steps: int = 48,
        depth: int = 6,
        num_latents: int = 256,
        cross_heads: int = 1,
        latent_heads: int = 8,
        cross_dim_heads: int = 8,
        latent_dim: int = 512,
        weight_tie_layers: bool = False,
        self_per_cross_attention: int = 2,
        dim: int = 32,
        logits_dim: int = 100,
        queries_dim: int = 32,
        latent_dim_heads: int = 64,
    ):
        super().__init__()
        self.model = PerceiverIO(
            dim=dim,  # dimension of sequence to be encoded
            queries_dim=queries_dim,  # dimension of decoder queries
            logits_dim=logits_dim,  # dimension of final logits
            depth=depth,  # depth of net
            num_latents=num_latents,  # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim=latent_dim,  # latent dimension
            cross_heads=cross_heads,  # number of heads for cross attention. paper said 1
            latent_heads=latent_heads,  # number of heads for latent self attention, 8
            cross_dim_head=cross_dim_heads,  # number of dimensions per cross attention head
            latent_dim_head=latent_dim_heads,  # number of dimensions per latent self attention head
            weight_tie_layers=weight_tie_layers,  # whether to weight tie layers (optional, as indicated in the diagram)
            self_per_cross_attn=self_per_cross_attention,  # number of self attention blocks per cross attention
        )

    def encode_inputs(self, x):
        image_inputs = torch.rand(size=(3, 260, 260, 3), requires_grad=True)
        video_inputs = torch.rand(size=(3, 32, 260, 260, 3), requires_grad=True)
        audio_inputs = torch.rand(size=(3, 48, 1), requires_grad=True)

        # Timeseries input
        video_modality = InputModality(
            name="timeseries",
            input_channels=12,  # number of channels for each token of the input -> 12 or 13 for sat channels + mask
            input_axis=3,  # number of axes, 3 for video
            num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
            max_freq=4.0,  # maximum frequency, hyperparameter depending on how fine the data is
        )
        # Use image modality for latlon, elevation, other base data?
        image_modality = InputModality(
            name="base",
            input_channels=4,  # number of channels for each token of the input, 3 for latlon + 1 topo
            input_axis=2,  # number of axes, 2 for images
            num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
            max_freq=4.0,  # maximum frequency, hyperparameter depending on how fine the data is
        )
        # Sort audio for timestep one-hot encode? Or include under other modality?
        timestep_modality = InputModality(
            name="timestep",
            input_channels=1,  # number of channels for mono audio
            input_axis=1,  # number of axes, 2 for images
            num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
            max_freq=8.0,  # maximum frequency, hyperparameter depending on how fine the data is
        )

    def decode_outputs(self, x):
        pass

    def training_step(self, batch, batch_idx):
        x, y = batch
        pass

    def configure_optimizers(self):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def forward(self, x):
        pass

    def visualize_step(
        self, x: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor, batch_idx: int, step: str
    ) -> None:
        # the logger you used (in this case tensorboard)
        tensorboard = self.logger.experiment[0]
        # Timesteps per channel
        images = x[0].cpu().detach()
        future_images = y[0].cpu().detach()
        generated_images = y_hat[0].cpu().detach()
        for i, t in enumerate(images):  # Now would be (C, H, W)
            t = [torch.unsqueeze(img, dim=0) for img in t]
            image_grid = torchvision.utils.make_grid(t, nrow=self.input_channels)
            tensorboard.add_image(
                f"{step}/Input_Image_Stack_Frame_{i}", image_grid, global_step=batch_idx
            )
            t = [torch.unsqueeze(img, dim=0) for img in future_images[i]]
            image_grid = torchvision.utils.make_grid(t, nrow=self.input_channels)
            tensorboard.add_image(
                f"{step}/Target_Image_Frame_{i}", image_grid, global_step=batch_idx
            )
            t = [torch.unsqueeze(img, dim=0) for img in generated_images[i]]
            image_grid = torchvision.utils.make_grid(t, nrow=self.input_channels)
            tensorboard.add_image(
                f"{step}/Generated_Image_Frame_{i}", image_grid, global_step=batch_idx
            )


def fourier_encode(x, max_freq, num_bands=4, base=2):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.logspace(
        0.0, log(max_freq / 2) / log(base), num_bands, base=base, device=device, dtype=dtype
    )
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1)
    return x


class PerceiverSat(torch.nn.Module):
    def __init__(
        self,
        modalities: Iterable[InputModality],
        fourier_encode_data: bool = True,
        input_axis: int = 3,
        num_freq_bands: int = 64,
        input_channels: int = 3,
        forecast_steps: int = 48,
        encode_time: bool = False,
        **kwargs,
    ):
        """
        PerceiverIO made to work more specifically with timeseries images
        Not a recurrent model, so like MetNet somewhat, can optionally give a one-hot encoded vector for the future
        timestep
        Args:
            fourier_encode_data: Whether to add fourier position encoding, like in the papers default True
            input_axis:
            num_freq_bands: Number of frequency bands for the Fourier encoding
            input_channels: Number of input channels
            forecast_steps: Number of forecast steps to make
            encode_time: Whether to encode the future timestep as a one-hot encded vector, iterates through all timesteps in forward.
            **kwargs:
        """
        super().__init__()
        self.fourier_encode_data = fourier_encode_data
        self.forecast_steps = forecast_steps
        self.input_channels = input_channels
        self.modalities = {modality.name: modality for modality in modalities}
        # we encode modality with one hot encoding, so need one dim per modality:
        modality_encoding_dim = sum([1 for _ in modalities])
        # input_dim is the maximum dimension over all input modalities:
        input_dim = max(modality.input_dim for modality in modalities) + modality_encoding_dim
        fourier_channels = (input_axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0
        input_dim = fourier_channels + input_channels
        self.perceiver = PerceiverIO(dim=input_dim, **kwargs)

    def decode_output(self, data):
        pass

    def forward(self, multi_modality_data: Dict[str, torch.Tensor], mask=None, queries=None):
        batch_sizes = set()
        num_modalities = len(multi_modality_data)
        linearized_data = []
        linearized_data_per_layer: Dict[int, List[torch.Tensor]] = {}

        for modality_index, modality_name in enumerate(sorted(multi_modality_data.keys())):
            assert (
                modality_name in self.modalities
            ), f"modality {modality_name} was not defined in constructor"
            data = multi_modality_data[modality_name]
            modality = self.modalities[modality_name]
            b, *axis, _, device = data.size()
            assert len(axis) == modality.input_axis, (
                f"input data must have the right number of  for modality {modality_name}. "
                f"Expected {modality.input_axis} while forward argument offered {len(axis)}"
            )
            batch_sizes.add(b)
            assert len(batch_sizes) == 1, "batch size must be the same across all modalities"
            enc_pos = []
            if self.fourier_encode_data:
                # calculate fourier encoded positions in the range of [-1, 1], for all axis

                axis_pos = list(
                    map(lambda size: torch.linspace(-1.0, 1.0, steps=size).type_as(data), axis)
                )
                pos = torch.stack(torch.meshgrid(*axis_pos), dim=-1)
                enc_pos = fourier_encode(
                    pos, modality.max_freq, modality.num_freq_bands, modality.freq_base
                )
                enc_pos = rearrange(enc_pos, "... n d -> ... (n d)")
                enc_pos = repeat(enc_pos, "... -> b ...", b=b)

                data = torch.cat((data, enc_pos), dim=-1)

            # Figure out padding for this modality, given max dimension across all modalities:
            padding_size = self.max_modality_dim - modality.input_dim - num_modalities

            padding = torch.zeros(size=data.size()[0:-1] + (padding_size,)).to(device)
            # concat to channels of data and flatten axis
            modality_encodings = modality_encoding(
                b, axis, modality_index, num_modalities, device=device
            )

            to_concat = (
                (data, padding, enc_pos, modality_encodings)
                if enc_pos
                else (data, padding, modality_encodings)
            )

            data = torch.cat(to_concat, dim=-1)
            data = rearrange(data, "b ... d -> b (...) d")
            linearized_data.append(data)
        b = batch_sizes.pop()
        x = repeat(self.latents, "n d -> b n d", b=b)

        # Concatenate all the modalities:
        data = torch.cat(linearized_data, dim=1)
        # concat to channels of data and flatten axis
        data = rearrange(data, "b ... d -> b (...) d")

        # After this is the PerceiverIO backbone, still would need to decode it back to an image though
        perceiver_output = self.perceiver.forward(data, mask, queries)

        # Have to decode back into future Sat image frames

        return NotImplementedError


class MultiModalityPerceiver(nn.Module):
    def __init__(
        self,
        *,
        modalities: Iterable[InputModality],
        depth,
        num_latents=512,
        latent_dim=512,
        cross_heads=1,
        latent_heads=8,
        cross_dim_head=64,
        latent_dim_head=64,
        num_classes=None,
        attn_dropout=0.0,
        ff_dropout=0.0,
        weight_tie_layers=False,
        num_latent_blocks_per_layer=1,
        use_gelu: bool = False,
    ):
        """
        :param modalities:
        :param depth: Number of times the perceiver will perform cross-attention between latent and input.
        :param num_latents:
        :param latent_dim:
        :param cross_heads:
        :param latent_heads:
        :param cross_dim_head:
        :param latent_dim_head:
        :param num_classes: Number of classes to predict, or if None, return the hidden state (num latents x hidden_dim)
        :param attn_dropout:
        :param ff_dropout:
        :param weight_tie_layers: True: share weights across layers, False no shared weights.
        :param num_latent_blocks_per_layer: Number of blocks in the latent transformer.
        :param use_gelu: Use GELU activation like the Perceiver preprint indicates. False,
               with Lucidrains' GEGLU activation in feed forward instead.
        """
        super().__init__()
        self.modalities = {modality.name: modality for modality in modalities}
        # we encode modality with one hot encoding, so need one dim per modality:
        modality_encoding_dim = sum([1 for _ in modalities])
        # input_dim is the maximum dimension over all input modalities:
        input_dim = max(modality.input_dim for modality in modalities) + modality_encoding_dim
        self.max_modality_dim = input_dim
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

    def forward(self, multi_modality_data: Dict[str, Tensor], mask=None):
        """
        :param data: a dictionary where keys are modality names and Tensor contain a batch
        of modality input data.
        :param mask:
        :return:
        """
        batch_sizes = set()
        num_modalities = len(multi_modality_data)
        linearized_data = []
        linearized_data_per_layer: Dict[int, List[Tensor]] = {}

        for modality_index, modality_name in enumerate(sorted(multi_modality_data.keys())):
            assert (
                modality_name in self.modalities
            ), f"modality {modality_name} was not defined in constructor"
            data = multi_modality_data[modality_name]
            modality = self.modalities[modality_name]
            b, *axis, _, device = *data.shape, data.device
            assert len(axis) == modality.input_axis, (
                f"input data must have the right number of  for modality {modality_name}. "
                f"Expected {modality.input_axis} while forward argument offered {len(axis)}"
            )
            batch_sizes.add(b)
            assert len(batch_sizes) == 1, "batch size must be the same across all modalities"
            # calculate fourier encoded positions in the range of [-1, 1], for all axis

            axis_pos = list(
                map(lambda size: torch.linspace(-1.0, 1.0, steps=size, device=device), axis)
            )
            pos = torch.stack(torch.meshgrid(*axis_pos), dim=-1)
            enc_pos = fourier_encode(
                pos, modality.max_freq, modality.num_freq_bands, modality.freq_base
            )
            enc_pos = rearrange(enc_pos, "... n d -> ... (n d)")
            enc_pos = repeat(enc_pos, "... -> b ...", b=b)

            # Figure out padding for this modality, given max dimension across all modalities:
            padding_size = self.max_modality_dim - modality.input_dim - num_modalities

            padding = torch.zeros(size=data.size()[0:-1] + (padding_size,)).to(device)
            # concat to channels of data and flatten axis
            modality_encodings = modality_encoding(
                b, axis, modality_index, num_modalities, device=device
            )

            to_concat = (data, padding, enc_pos, modality_encodings)

            data = torch.cat(to_concat, dim=-1)
            data = rearrange(data, "b ... d -> b (...) d")
            linearized_data.append(data)
        b = batch_sizes.pop()
        x = repeat(self.latents, "n d -> b n d", b=b)

        # Concatenate all the modalities:
        data = torch.cat(linearized_data, dim=1)

        for cross_attn, cross_ff, latent_transformer in self.layers:
            x = cross_attn(x, context=data, mask=mask) + x
            x = cross_ff(x) + x
            x = latent_transformer(x) + x
        x = self.pool(x)

        return self.to_logits(x)

        def pool(self, x):
            """
            Perform pooling over latents.
            :param x: batch x num_latents x latent_dim
            :return: pooled x
            """
            # implement global pooling
            return x.mean(dim=-2)
