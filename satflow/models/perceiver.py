from perceiver_pytorch import PerceiverIO
import torch
import pytorch_lightning as pl
import torchvision
from typing import List, Iterable, Dict
from satflow.models.base import register_model
from math import pi, log
from einops import rearrange, repeat
from satflow.models.layers.modalities import modality_encoding, InputModality
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from satflow.models.losses import get_loss
from satflow.models.layers import ConditionTime
import torch_optimizer as optim
import logging

logger = logging.getLogger("satflow.model")
logger.setLevel(logging.WARN)


@register_model
class Perceiver(pl.LightningModule):
    def __init__(
        self,
        input_channels: int = 12,
        sat_channels: int = 12,
        forecast_steps: int = 48,
        input_size: int = 64,
        lr: float = 5e-4,
        visualize: bool = True,
        max_frequency: float = 4.0,
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
        loss="mse",
        sin_only: bool = False,
        encode_fourier: bool = True,
    ):
        super().__init__()
        self.forecast_steps = forecast_steps
        self.sat_channels = sat_channels
        self.lr = lr
        self.visualize = visualize
        self.criterion = get_loss(loss)
        # Timeseries input
        video_modality = InputModality(
            name="timeseries",
            input_channels=sat_channels,
            input_axis=3,  # number of axes, 3 for video
            num_freq_bands=input_size,  # number of freq bands, with original value (2 * K + 1)
            max_freq=max_frequency,  # maximum frequency, hyperparameter depending on how fine the data is
            sin_only=sin_only,  # Whether if sine only for Fourier encoding, TODO test more
            fourier_encode=encode_fourier,  # Whether to encode position with Fourier features
        )
        # Use image modality for latlon, elevation, other base data?
        image_modality = InputModality(
            name="base",
            input_channels=input_channels - sat_channels,
            input_axis=2,  # number of axes, 2 for images
            num_freq_bands=input_size,  # number of freq bands, with original value (2 * K + 1)
            max_freq=max_frequency,  # maximum frequency, hyperparameter depending on how fine the data is
            sin_only=sin_only,
            fourier_encode=encode_fourier,
        )
        # Sort audio for timestep one-hot encode? Or include under other modality?
        timestep_modality = InputModality(
            name="forecast_time",
            input_channels=1,  # number of channels for mono audio
            input_axis=1,  # number of axes, 2 for images
            num_freq_bands=self.forecast_steps,  # number of freq bands, with original value (2 * K + 1)
            max_freq=max_frequency,  # maximum frequency, hyperparameter depending on how fine the data is
            sin_only=sin_only,
            fourier_encode=encode_fourier,
        )
        self.model = MultiPerceiverSat(
            modalities=[video_modality, image_modality, timestep_modality],
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
            sin_only=sin_only,
            fourier_encode_data=encode_fourier,
        )

        self.ct = ConditionTime(forecast_steps, ch_dim=3, num_dims=4)

    def encode_inputs(self, x, timestep: int = 1):
        # One hot encode the inpuuts
        video_inputs = x[:, :, : self.sat_channels, :, :]
        # base_input = x[:, 0, self.sat_channels :, :, :]
        batch_size, seq_len, n_chans, width, height = video_inputs.shape
        """
        # Stack timesteps as channels (to make a large batch)
        new_batch_size = n_chans * seq_len
        # Reshuffle to put channels last
        base_input = base_input.reshape(batch_size, width, height, -1)
        #                                 0           1       2      3
        sat_data = video_inputs.reshape(batch_size, width, height, new_batch_size)
        sat_data = torch.cat(
            [sat_data, base_input], dim=-1
        )  # Only have one copy of the basemap, instead of N copies
        return sat_data
        """
        base_inputs = x[
            :, 0, self.sat_channels :, :, :
        ]  # Base maps should be the same for all timesteps in a sample
        video_inputs = video_inputs.permute(0, 1, 3, 4, 2)  # Channel last
        base_inputs = base_inputs.permute(0, 2, 3, 1)  # Channel last
        logger.debug(f"Timeseries: {video_inputs.size()} Base: {base_inputs.size()}")
        return {"timeseries": video_inputs, "base": base_inputs}

    def add_timestep(self, batch_size: int, timestep: int = 1):
        times = (torch.eye(self.forecast_steps)[timestep]).unsqueeze(-1).unsqueeze(-1)
        ones = torch.ones(1, 1, 1)
        timestep_input = times * ones
        timestep_input = timestep_input.squeeze(-1)
        timestep_input = repeat(timestep_input, "... -> b ...", b=batch_size)
        logger.debug(f"Forecast Step: {timestep_input.size()}")
        return timestep_input

    def decode_outputs(self, x):
        pass

    def training_step(self, batch, batch_idx):
        x, y = batch
        batch_size = y.size(0)
        # For each future timestep:
        predictions = []
        query = self.construct_query(x)
        x = self.encode_inputs(x)
        for i in range(self.forecast_steps):
            x["forecast_time"] = self.add_timestep(batch_size, i).type_as(y)
            # x_i = self.ct(x["timeseries"], fstep=i)
            y_hat = self(x, query=query)
            predictions.append(y_hat)
        y_hat = torch.stack(predictions, dim=1)  # Stack along the timestep dimension
        loss = self.criterion(y, y_hat)
        self.log_dict({"train/loss": loss})
        frame_loss_dict = {}
        for f in range(self.forecast_steps):
            frame_loss = self.criterion(y_hat[:, f, :, :], y[:, f, :, :]).item()
            frame_loss_dict[f"train/frame_{f}_loss"] = frame_loss
        self.log_dict(frame_loss_dict)
        return loss

    def configure_optimizers(self):
        # They use LAMB as the optimizer
        optimizer = optim.Lamb(self.parameters(), lr=self.lr, betas=(0.9, 0.999))
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=100)
        lr_dict = {
            # REQUIRED: The scheduler instance
            "scheduler": scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "step",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": None,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_dict}

    def construct_query(self, x):
        y_query = x[:, -1, 0, :, :]  # Only want sat channels, the output
        # y_query = torch.permute(y_query, (0, 2, 3, 1)) # Channel Last
        # Need to reshape to 3 dimensions, TxHxW or HxWxC
        # y_query = rearrange(y_query, "b h w d -> b (h w) d")
        logger.debug(f"Query Shape: {y_query.shape}")
        return y_query

    def validation_step(self, batch, batch_idx):
        x, y = batch
        batch_size = y.size(0)
        predictions = []
        query = self.construct_query(x)
        x = self.encode_inputs(x)
        for i in range(self.forecast_steps):
            # x_i = self.ct(x["timeseries"], fstep=i)
            x["forecast_time"] = self.add_timestep(batch_size, i).type_as(y)
            y_hat = self(x, query=query)
            predictions.append(y_hat)
        y_hat = torch.stack(predictions, dim=1)  # Stack along the timestep dimension
        loss = self.criterion(y, y_hat)
        self.log_dict({"val/loss": loss})
        frame_loss_dict = {}
        for f in range(self.forecast_steps):
            frame_loss = self.criterion(y_hat[:, f, :, :], y[:, f, :, :]).item()
            frame_loss_dict[f"val/frame_{f}_loss"] = frame_loss
        self.log_dict(frame_loss_dict)
        return loss

    def forward(self, x, mask=None, query=None):
        return self.model.forward(x, mask=mask, queries=query)

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


def fourier_encode(x, max_freq, num_bands=4, base=2, sin_only=False):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.logspace(
        0.0, log(max_freq / 2) / log(base), num_bands, base=base, device=device, dtype=dtype
    )
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = x.sin() if sin_only else torch.cat([x.sin(), x.cos()], dim=-1)
    logger.debug(f"Fourier Channels Shape: {x.size()} Sin Only: {sin_only}")
    x = torch.cat((x, orig_x), dim=-1)
    return x


class PerceiverSat(torch.nn.Module):
    def __init__(
        self,
        modalities: Iterable[InputModality],
        fourier_encode_data: bool = True,
        input_channels: int = 3,
        forecast_steps: int = 48,
        input_axis=2,
        num_freq_bands=64,
        **kwargs,
    ):
        """
                PerceiverIO made to work more specifically with timeseries images
                Not a recurrent model, so lifrom torch.nn import Embedding
        ke MetNet somewhat, can optionally give a one-hot encoded vector for the future
                timestep
                Args:
                    input_channels: Number of input channels
                    forecast_steps: Number of forecast steps to make
                    **kwargs:
        """
        super().__init__()
        self.fourier_encode_data = fourier_encode_data
        self.forecast_steps = forecast_steps
        self.input_channels = input_channels
        self.input_axis = input_axis
        self.num_freq_bands = num_freq_bands
        self.freq_base = 2
        self.max_freq = 8.0
        self.fourier_encode_data = fourier_encode_data
        fourier_channels = (input_axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0
        input_dim = fourier_channels + input_channels
        # Pop dim
        self.max_modality_dim = input_dim
        kwargs.pop("dim")
        logger.debug(f"Input dim: {input_dim}")
        self.perceiver = PerceiverIO(dim=346, **kwargs)
        self.conv = torch.nn.Conv2d(in_channels=64, out_channels=12, kernel_size=(1, 1))

    def decode_output(self, data):
        pass

    def forward(self, data: torch.Tensor, mask=None, queries=None):
        b, *axis, m = data.size()
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
        logger.debug(data.shape)
        # After this is the PerceiverIO backbone, still would need to decode it back to an image though
        perceiver_output = self.perceiver.forward(data, mask, queries)

        # For multiple modalities, they are split after this beack into different tensors
        # For Sat images, we just want the images, not the other ones, so can leave it as is?

        # Have to decode back into future Sat image frames
        # Perceiver for 'pixel' postprocessing does nothing, or undoes the space2depth from before if just image
        # If doing depth2space, should split modalities again
        logger.debug(perceiver_output.size())

        # For a 2, 4096, 334 input gives 2, 256, 512 output, which could be reshaped to 64x64x32 output -> 1x1 conv down to 12 sat channels
        image_output = perceiver_output.reshape(b, -1, *axis)
        logger.debug(image_output.size())
        # Downscale to 12 channel output
        image_output = self.conv(image_output)
        logger.debug(image_output.size())
        return image_output


class MultiPerceiverSat(torch.nn.Module):
    def __init__(
        self,
        modalities: Iterable[InputModality],
        fourier_encode_data: bool = True,
        input_channels: int = 3,
        output_channels: int = 12,
        forecast_steps: int = 48,
        sin_only: bool = False,
        **kwargs,
    ):
        """
        PerceiverIO made to work more specifically with timeseries images
        Not a recurrent model, so like MetNet somewhat, can optionally give a one-hot encoded vector for the future
        timestep
        Args:
            input_channels: Number of input channels
            forecast_steps: Number of forecast steps to make
            **kwargs:
        """
        super().__init__()
        self.fourier_encode_data = fourier_encode_data
        self.forecast_steps = forecast_steps
        self.input_channels = input_channels
        self.sin_only = sin_only
        self.output_channels = output_channels
        self.modalities = {modality.name: modality for modality in modalities}
        # we encode modality with one hot encoding, so need one dim per modality:
        modality_encoding_dim = sum([1 for _ in modalities])
        # input_dim is the maximum dimension over all input modalities:
        logger.debug(
            f"Max Modality Input Dim: {max(modality.input_dim for modality in modalities)} Encoding Dim: {modality_encoding_dim}"
        )
        input_dim = max(modality.input_dim for modality in modalities) + modality_encoding_dim
        # Pop dim
        self.max_modality_dim = input_dim
        logger.debug(f"Max Modality Dim: {self.max_modality_dim}")
        kwargs.pop("dim")
        # Want toe logit_dim to be the same as the channels * width or height
        kwargs["logits_dim"] = 32 * self.output_channels
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
            b, *axis, _ = data.size()
            logger.debug(modality_name)
            logger.debug(axis)
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
                    pos,
                    modality.max_freq,
                    modality.num_freq_bands,
                    modality.freq_base,
                    sin_only=self.sin_only,
                )
                logger.debug(f"Encoding Shape: {enc_pos.size()}")
                enc_pos = rearrange(enc_pos, "... n d -> ... (n d)")
                logger.debug(f"Encoding Shape After Rearranging: {enc_pos.size()}")
                enc_pos = repeat(enc_pos, "... -> b ...", b=b)

            # Figure out padding for this modality, given max dimension across all modalities:
            logger.debug(
                f"Max Size: {self.max_modality_dim} Input Dim: {modality.input_dim} Num Modality: {num_modalities}"
            )
            padding_size = self.max_modality_dim - modality.input_dim - num_modalities
            logger.debug(f"Padding Size: {padding_size}")

            padding = torch.zeros(size=data.size()[0:-1] + (padding_size,)).type_as(data)
            # concat to channels of data and flatten axis
            modality_encodings = modality_encoding(
                b, axis, modality_index, num_modalities
            ).type_as(data)
            to_concat = (
                (data, padding, enc_pos, modality_encodings)
                if len(enc_pos) > 0
                else (data, padding, modality_encodings)
            )
            logger.debug(
                f"Data: {data.size()} Padding: {padding.size()} Enc_pos: {enc_pos.size()} Modality: {modality_encodings.size()}"
            )
            data = torch.cat(to_concat, dim=-1)
            # concat to channels of data and flatten axis
            data = rearrange(data, "b ... d -> b (...) d")
            logger.debug(f"{modality_name} Size: {data.size()}")
            linearized_data.append(data)

        # Concatenate all the modalities:
        logger.debug([t.size() for t in linearized_data])
        data = torch.cat(linearized_data, dim=1)

        # After this is the PerceiverIO backbone, still would need to decode it back to an image though
        # Should include the query shape here for the output we want, could be learned embeddings, repeated input frames of the same shape that is desired, etc.
        perceiver_output = self.perceiver.forward(data, mask, queries)

        # For multiple modalities, they are split after this beack into different tensors
        # For Sat images, we just want the images, not the other ones, so can leave it as is?

        # Have to decode back into future Sat image frames
        # Perceiver for 'pixel' postprocessing does nothing, or undoes the space2depth from before if just image
        # If doing depth2space, should split modalities again
        logger.debug(perceiver_output.size())

        # Reshape to the correct output
        # This is how it is done in the official implementation, do a decoder query with cross attention, then just reshape the output
        # For timeseries, this is given as a query with T*H*W shape
        # For Flow Decoder, this is the same, except has a rescale factor
        perceiver_output = rearrange(
            perceiver_output, "b h (w c) -> b c h w", c=self.output_channels
        )
        logger.debug(f"Perceiver Finished Output: {perceiver_output.size()}")
        return perceiver_output
