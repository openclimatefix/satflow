from perceiver_pytorch import PerceiverIO
import torch
from satflow.models.losses import get_loss, NowcastingLoss, GridCellLoss
import pytorch_lightning as pl
import torchvision
from functools import reduce
from typing import List
from satflow.models.base import register_model


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
        pass

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


from math import pi, log
from einops import rearrange, repeat


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
        fourier_encode_data: bool = True,
        input_axis: int = 3,
        num_freq_bands: int = 64,
        input_channels: int = 3,
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
            encode_time: Whether to encode the future timestep as a one-hot encded vector, iterates through all timesteps in forward.
            **kwargs:
        """
        super().__init__()
        self.fourier_encode_data = fourier_encode_data
        fourier_channels = (input_axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0
        input_dim = fourier_channels + input_channels
        self.perceiver = PerceiverIO(dim=input_dim, **kwargs)

    def decode_output(self, data):
        pass

    def forward(self, data: torch.Tensor, mask=None, queries=None):
        b, *axis, _ = data.size()
        assert len(axis) == self.input_axis, "input data must have the right number of axis"

        if self.fourier_encode_data:
            # calculate fourier encoded positions in the range of [-1, 1], for all axis

            axis_pos = list(
                map(lambda size: torch.linspace(-1.0, 1.0, steps=size).type_as(data), axis)
            )
            pos = torch.stack(torch.meshgrid(*axis_pos), dim=-1)
            enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands, base=self.freq_base)
            enc_pos = rearrange(enc_pos, "... n d -> ... (n d)")
            enc_pos = repeat(enc_pos, "... -> b ...", b=b)

            data = torch.cat((data, enc_pos), dim=-1)

        # concat to channels of data and flatten axis
        data = rearrange(data, "b ... d -> b (...) d")

        # After this is the PerceiverIO backbone, still would need to decode it back to an image though
        perceiver_output = self.perceiver.forward(data, mask, queries)

        # Have to decode back into future Sat image frames

        return NotImplementedError
