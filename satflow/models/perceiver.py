from perceiver_pytorch import PerceiverIO, MultiPerceiver
from perceiver_pytorch.modalities import InputModality, modality_encoding
from perceiver_pytorch.utils import encode_position
from perceiver_pytorch.encoders import ImageEncoder
from perceiver_pytorch.decoders import ImageDecoder
import torch
from math import prod
from torch.distributions import uniform
from typing import Iterable, Dict, Optional, Any, Union, Tuple
from satflow.models.base import register_model, BaseModel
from einops import rearrange, repeat
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from satflow.models.losses import get_loss
import torch_optimizer as optim
import logging

logger = logging.getLogger("satflow.model")
logger.setLevel(logging.WARN)


@register_model
class Perceiver(BaseModel):
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
        decoder_ff: bool = True,
        dim: int = 32,
        logits_dim: int = 100,
        queries_dim: int = 32,
        latent_dim_heads: int = 64,
        loss="mse",
        sin_only: bool = False,
        encode_fourier: bool = True,
        preprocessor_type: Optional[str] = None,
        postprocessor_type: Optional[str] = None,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        decoder_kwargs: Optional[Dict[str, Any]] = None,
        pretrained: bool = False,
    ):
        super(BaseModel, self).__init__()
        self.forecast_steps = forecast_steps
        self.input_channels = input_channels
        self.lr = lr
        self.pretrained = pretrained
        self.visualize = visualize
        self.sat_channels = sat_channels
        self.output_channels = sat_channels
        self.criterion = get_loss(loss)

        # Warn if using frequency is smaller than Nyquist Frequency
        if max_frequency < input_size / 2:
            print(
                f"Max frequency is less than Nyquist frequency, currently set to {max_frequency} while "
                f"the Nyquist frequency for input of size {input_size} is {input_size / 2}"
            )

        # Preprocessor, if desired, on top of the other processing done
        if preprocessor_type is not None:
            if preprocessor_type not in ("conv", "patches", "pixels", "conv1x1", "metnet"):
                raise ValueError("Invalid prep_type!")
            if preprocessor_type == "metnet":
                # MetNet processing
                self.preprocessor = ImageEncoder(
                    crop_size=input_size,
                    preprocessor_type="metnet",
                )
                video_input_channels = (
                    8 * sat_channels
                )  # This is only done on the sat channel inputs
                # If doing it on the base map, then need
                image_input_channels = 4 * (input_channels - sat_channels)
            else:
                self.preprocessor = ImageEncoder(
                    input_channels=sat_channels,
                    preprocessor_type=preprocessor_type,
                    **encoder_kwargs,
                )
                video_input_channels = self.preprocessor.output_channels
                image_input_channels = self.preprocessor.output_channels
        else:
            self.preprocessor = None
            video_input_channels = sat_channels
            image_input_channels = input_channels - sat_channels

        # The preprocessor will change the number of channels in the input

        # Timeseries input
        video_modality = InputModality(
            name="timeseries",
            input_channels=video_input_channels,
            input_axis=3,  # number of axes, 3 for video
            num_freq_bands=input_size,  # number of freq bands, with original value (2 * K + 1)
            max_freq=max_frequency,  # maximum frequency, hyperparameter depending on how fine the data is, should be Nyquist frequency (i.e. 112 for 224 input image)
            sin_only=sin_only,  # Whether if sine only for Fourier encoding, TODO test more
            fourier_encode=encode_fourier,  # Whether to encode position with Fourier features
        )
        # Use image modality for latlon, elevation, other base data?
        image_modality = InputModality(
            name="base",
            input_channels=image_input_channels,
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
        self.model = MultiPerceiver(
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
            # self_per_cross_attn=self_per_cross_attention,  # number of self attention blocks per cross attention
            sine_only=sin_only,
            fourier_encode_data=encode_fourier,
            output_shape=input_size,  # Shape of output to make the correct sized logits dim, needed so reshaping works
            decoder_ff=decoder_ff,  # Optional decoder FF
        )

        if postprocessor_type is not None:
            if postprocessor_type not in ("conv", "patches", "pixels", "conv1x1"):
                raise ValueError("Invalid postprocessor_type!")
            self.postprocessor = ImageDecoder(
                postprocess_type=postprocessor_type, output_channels=sat_channels, **decoder_kwargs
            )
        else:
            self.postprocessor = None

        self.save_hyperparameters()

    def encode_inputs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        video_inputs = x[:, :, : self.sat_channels, :, :]
        base_inputs = x[
            :, 0, self.sat_channels :, :, :
        ]  # Base maps should be the same for all timesteps in a sample

        # Run the preprocessors here when encoding the inputs
        if self.preprocessor is not None:
            # Expects Channel first
            video_inputs = self.preprocessor(video_inputs)
            base_inputs = self.preprocessor(base_inputs)
        video_inputs = video_inputs.permute(0, 1, 3, 4, 2)  # Channel last
        base_inputs = base_inputs.permute(0, 2, 3, 1)  # Channel last
        logger.debug(f"Timeseries: {video_inputs.size()} Base: {base_inputs.size()}")
        return {"timeseries": video_inputs, "base": base_inputs}

    def add_timestep(self, batch_size: int, timestep: int = 1) -> torch.Tensor:
        times = (torch.eye(self.forecast_steps)[timestep]).unsqueeze(-1).unsqueeze(-1)
        ones = torch.ones(1, 1, 1)
        timestep_input = times * ones
        timestep_input = timestep_input.squeeze(-1)
        timestep_input = repeat(timestep_input, "... -> b ...", b=batch_size)
        logger.debug(f"Forecast Step: {timestep_input.size()}")
        return timestep_input

    def _train_or_validate_step(self, batch, batch_idx, is_training: bool = True):
        x, y = batch
        batch_size = y.size(0)
        # For each future timestep:
        predictions = []
        query = self.construct_query(x)
        if self.visualize:
            vis_x = x.cpu()
        x = self.encode_inputs(x)
        for i in range(self.forecast_steps):
            x["forecast_time"] = self.add_timestep(batch_size, i).type_as(y)
            # x_i = self.ct(x["timeseries"], fstep=i)
            y_hat = self(x, query=query)
            y_hat = rearrange(y_hat, "b h (w c) -> b c h w", c=self.output_channels)
            predictions.append(y_hat)
        y_hat = torch.stack(predictions, dim=1)  # Stack along the timestep dimension
        if self.postprocessor is not None:
            y_hat = self.postprocessor(y_hat)
        if self.visualize:
            # Only visualize the second batch of train/val
            if batch_idx == 1:
                self.visualize_step(
                    vis_x, y, y_hat, batch_idx, step=f"{'train' if is_training else 'val'}"
                )
        loss = self.criterion(y, y_hat)
        self.log_dict({f"{'train' if is_training else 'val'}/loss": loss})
        frame_loss_dict = {}
        for f in range(self.forecast_steps):
            frame_loss = self.criterion(y_hat[:, f, :, :, :], y[:, f, :, :, :]).item()
            frame_loss_dict[f"{'train' if is_training else 'val'}/frame_{f}_loss"] = frame_loss
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
        # key, value: B x N x K; query: B x M x K
        # Attention maps -> B x N x M
        # Output -> B x M x K
        # So want query to be B X (T*H*W) X C to reshape to B x T x C x H x W
        if self.preprocessor is not None:
            x = self.preprocessor(x)
        y_query = x[:, -1, 0, :, :]  # Only want sat channels, the output
        # y_query = torch.permute(y_query, (0, 2, 3, 1)) # Channel Last
        # Need to reshape to 3 dimensions, TxHxW or HxWxC
        # y_query = rearrange(y_query, "b h w d -> b (h w) d")
        logger.debug(f"Query Shape: {y_query.shape}")
        return y_query

    def forward(self, x, mask=None, query=None):
        return self.model.forward(x, mask=mask, queries=query)


class MultiPerceiverSat(torch.nn.Module):
    def __init__(
        self,
        use_input_as_query: bool = False,
        use_learnable_query: bool = False,
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
        super(MultiPerceiverSat, self).__init__()
        self.multi_perceiver = MultiPerceiver(**kwargs)
        if use_learnable_query:
            self.learnable_query = torch.nn.Linear(self.query_dim, self.query_dim)
            self.distribution = uniform.Uniform(low=torch.Tensor([0.0]), high=torch.Tensor([1.0]))
            self.query_dim = kwargs.get("query_dim", 32)
            self.query_future_size = prod(kwargs.get("output_shape", [24, 32, 32]))
            # Like GAN sorta, random input, learn important parts in linear layer T*H*W shape,
            # need to add Fourier features too though

    def forward(self, multi_modality_data: Dict[str, torch.Tensor], mask=None, queries=None):
        data = self.multi_perceiver.forward(multi_modality_data)
        # Create learnable query here, need to add fourier features as well
        if self.use_learnable_query:
            # Create learnable query, also adds somewhat ensemble on multiple forward passes
            # Middle is the shape of the future timesteps and such
            z = self.distribution.sample(
                (data.shape[0], self.query_future_size, self.query_dim)
            ).type_as(data)
            queries = self.learnable_query(z)
            # Add Fourier Features now to the query

        perceiver_output = self.multi_perceiver.perceiver.forward(data, mask, queries)

        logger.debug(f"Perceiver Finished Output: {perceiver_output.size()}")
        return perceiver_output
