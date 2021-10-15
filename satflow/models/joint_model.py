from perceiver_pytorch import MultiPerceiver
from perceiver_pytorch.modalities import InputModality
from perceiver_pytorch.encoders import ImageEncoder
from perceiver_pytorch.decoders import ImageDecoder
from perceiver_pytorch.queries import LearnableQuery
from perceiver_pytorch.utils import encode_position
import torch
from typing import Iterable, Dict, Optional, Any, Union, Tuple
from nowcasting_utils.models.base import register_model, BaseModel
from einops import rearrange, repeat
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from nowcasting_utils.models.loss import get_loss
import torch_optimizer as optim
import logging
from nowcasting_dataset.consts import (
    SATELLITE_DATA,
    NWP_DATA,
    TOPOGRAPHIC_DATA,
    GSP_YIELD,
    PV_YIELD,
    PV_SYSTEM_ID,
    GSP_ID,
    DEFAULT_N_GSP_PER_EXAMPLE,
    DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE,
)

logger = logging.getLogger("satflow.model")
logger.setLevel(logging.WARN)


@register_model
class JointPerceiver(BaseModel):
    def __init__(
        self,
        input_channels: int = 22,
        sat_channels: int = 12,
        nwp_channels: int = 10,
        base_channels: int = 1,
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
        gsp_loss="mse",
        sin_only: bool = False,
        encode_fourier: bool = True,
        preprocessor_type: Optional[str] = None,
        postprocessor_type: Optional[str] = None,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        decoder_kwargs: Optional[Dict[str, Any]] = None,
        pretrained: bool = False,
        predict_timesteps_together: bool = False,
        nwp_modality: bool = False,
        use_learnable_query: bool = True,
        generate_fourier_features: bool = True,
        temporally_consistent_fourier_features: bool = False,
        use_gsp_data: bool = False,
        use_pv_data: bool = False,
    ):
        super(BaseModel, self).__init__()
        self.forecast_steps = forecast_steps
        self.input_channels = input_channels
        self.lr = lr
        self.pretrained = pretrained
        self.visualize = visualize
        self.sat_channels = sat_channels
        self.nwp_channels = nwp_channels
        self.output_channels = sat_channels
        self.criterion = get_loss(loss)
        self.gsp_criterion = get_loss(gsp_loss)
        self.input_size = input_size
        self.predict_timesteps_together = predict_timesteps_together
        self.use_learnable_query = use_learnable_query
        self.max_frequency = max_frequency
        self.temporally_consistent_fourier_features = temporally_consistent_fourier_features
        self.use_gsp_data = use_gsp_data
        self.use_pv_data = use_pv_data

        if use_learnable_query:
            self.query = LearnableQuery(
                channel_dim=queries_dim,
                query_shape=(self.forecast_steps, self.input_size, self.input_size)
                if predict_timesteps_together
                else (self.input_size, self.input_size),
                conv_layer="3d",
                max_frequency=max_frequency,
                num_frequency_bands=input_size,
                sine_only=sin_only,
                generate_fourier_features=generate_fourier_features,
            )
            # Need second query as well
            self.pv_query = LearnableQuery(
                channel_dim=queries_dim,
                query_shape=(self.forecast_steps, 1)  # Only need one number for GSP
                if predict_timesteps_together
                else (1, 1),
                conv_layer="3d",
                max_frequency=max_frequency,
                num_frequency_bands=input_size,
                sine_only=sin_only,
                generate_fourier_features=generate_fourier_features,
            )
        else:
            self.pv_query = None
            self.query = None

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
                    prep_type="metnet",
                )
                video_input_channels = (
                    8 * sat_channels
                )  # This is only done on the sat channel inputs
                nwp_input_channels = 8 * nwp_channels
                # If doing it on the base map, then need
                image_input_channels = 4 * base_channels
            else:
                self.preprocessor = ImageEncoder(
                    input_channels=sat_channels,
                    prep_type=preprocessor_type,
                    **encoder_kwargs,
                )
                nwp_input_channels = self.preprocessor.output_channels
                video_input_channels = self.preprocessor.output_channels
                image_input_channels = self.preprocessor.output_channels
        else:
            self.preprocessor = None
            nwp_input_channels = nwp_channels
            video_input_channels = sat_channels
            image_input_channels = base_channels

        # The preprocessor will change the number of channels in the input
        modalities = []
        # Timeseries input
        sat_modality = InputModality(
            name=SATELLITE_DATA,
            input_channels=video_input_channels,
            input_axis=3,  # number of axes, 3 for video
            num_freq_bands=input_size,  # number of freq bands, with original value (2 * K + 1)
            max_freq=max_frequency,  # maximum frequency, hyperparameter depending on how fine the data is, should be Nyquist frequency (i.e. 112 for 224 input image)
            sin_only=sin_only,  # Whether if sine only for Fourier encoding, TODO test more
            fourier_encode=encode_fourier,  # Whether to encode position with Fourier features
        )
        modalities.append(sat_modality)
        if nwp_modality:
            nwp_modality = InputModality(
                name=NWP_DATA,
                input_channels=nwp_input_channels,
                input_axis=3,  # number of axes, 3 for video
                num_freq_bands=input_size,  # number of freq bands, with original value (2 * K + 1)
                max_freq=max_frequency,  # maximum frequency, hyperparameter depending on how fine the data is, should be Nyquist frequency (i.e. 112 for 224 input image)
                sin_only=sin_only,  # Whether if sine only for Fourier encoding, TODO test more
                fourier_encode=encode_fourier,  # Whether to encode position with Fourier features
            )
            modalities.append(nwp_modality)
        # Use image modality for latlon, elevation, other base data?
        image_modality = InputModality(
            name=TOPOGRAPHIC_DATA,
            input_channels=image_input_channels,
            input_axis=2,  # number of axes, 2 for images
            num_freq_bands=input_size,  # number of freq bands, with original value (2 * K + 1)
            max_freq=max_frequency,  # maximum frequency, hyperparameter depending on how fine the data is
            sin_only=sin_only,
            fourier_encode=encode_fourier,
        )
        modalities.append(image_modality)

        if use_gsp_data:
            # Sort audio for timestep one-hot encode? Or include under other modality?
            gsp_modality = InputModality(
                name=GSP_YIELD,
                input_channels=1,  # number of channels for mono audio
                input_axis=1,  # number of axes, 2 for images
                num_freq_bands=self.forecast_steps,  # number of freq bands, with original value (2 * K + 1)
                max_freq=max_frequency,  # maximum frequency, hyperparameter depending on how fine the data is
                sin_only=sin_only,
                fourier_encode=encode_fourier,
            )
            modalities.append(gsp_modality)
            gsp_id_modality = InputModality(
                name=GSP_ID,
                input_channels=1,  # number of channels for mono audio
                input_axis=1,  # number of axes, 2 for images
                num_freq_bands=DEFAULT_N_GSP_PER_EXAMPLE,  # number of freq bands, with original value (2 * K + 1)
                max_freq=max_frequency,  # maximum frequency, hyperparameter depending on how fine the data is
                sin_only=sin_only,
                fourier_encode=encode_fourier,
            )
            modalities.append(gsp_id_modality)

        if use_pv_data:
            # Sort audio for timestep one-hot encode? Or include under other modality?
            pv_modality = InputModality(
                name=PV_YIELD,
                input_channels=1,  # number of channels for mono audio
                input_axis=1,  # number of axes, 2 for images
                num_freq_bands=self.forecast_steps,  # number of freq bands, with original value (2 * K + 1)
                max_freq=max_frequency,  # maximum frequency, hyperparameter depending on how fine the data is
                sin_only=sin_only,
                fourier_encode=encode_fourier,
            )
            modalities.append(pv_modality)
            pv_id_modality = InputModality(
                name=PV_SYSTEM_ID,
                input_channels=1,  # number of channels for mono audio
                input_axis=1,  # number of axes, 2 for images
                num_freq_bands=DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE,  # number of freq bands, with original value (2 * K + 1)
                max_freq=max_frequency,  # maximum frequency, hyperparameter depending on how fine the data is
                sin_only=sin_only,
                fourier_encode=encode_fourier,
            )
            modalities.append(pv_id_modality)

        self.model = MultiPerceiver(
            modalities=modalities,
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

    def encode_inputs(self, x: dict) -> Dict[str, torch.Tensor]:
        video_inputs = x[SATELLITE_DATA]
        nwp_inputs = x.get(NWP_DATA, [])
        base_inputs = x.get(
            TOPOGRAPHIC_DATA, []
        )  # Base maps should be the same for all timesteps in a sample

        # Run the preprocessors here when encoding the inputs
        if self.preprocessor is not None:
            # Expects Channel first
            video_inputs = self.preprocessor(video_inputs)
            base_inputs = self.preprocessor(base_inputs)
            if nwp_inputs:
                nwp_inputs = self.preprocessor(nwp_inputs)
        video_inputs = video_inputs.permute(0, 1, 3, 4, 2)  # Channel last
        if nwp_inputs:
            nwp_inputs = nwp_inputs.permute(0, 1, 3, 4, 2)  # Channel last
            x[NWP_DATA] = nwp_inputs
        base_inputs = base_inputs.permute(0, 2, 3, 1)  # Channel last
        logger.debug(f"Timeseries: {video_inputs.size()} Base: {base_inputs.size()}")
        x[SATELLITE_DATA] = video_inputs
        x[TOPOGRAPHIC_DATA] = base_inputs
        return x

    def _train_or_validate_step(self, batch, batch_idx, is_training: bool = True):
        x, y = batch
        sat_query, gsp_query = self.construct_query(x)
        x = self.encode_inputs(x)
        # Predicting all future ones at once
        sat_y_hat = self(x, query=sat_query)
        gsp_y_hat = self(x, query=gsp_query)
        sat_y_hat = rearrange(
            sat_y_hat,
            "b (t h w) c -> b c t h w",
            t=self.forecast_steps,
            h=self.input_size,
            w=self.input_size,
        )
        if self.postprocessor is not None:
            sat_y_hat = self.postprocessor(sat_y_hat)
        if self.visualize:
            self.visualize_step(x, y, sat_y_hat, batch_idx, step="train" if is_training else "val")

        # Satellite Loss
        loss = self.criterion(y[SATELLITE_DATA], sat_y_hat)
        self.log_dict({f"{'train' if is_training else 'val'}/sat_loss": loss})
        frame_loss_dict = {}
        for f in range(self.forecast_steps):
            frame_loss = self.criterion(
                sat_y_hat[:, f, :, :, :], y[SATELLITE_DATA][:, f, :, :, :]
            ).item()
            frame_loss_dict[
                f"{'train' if is_training else 'val'}/sat_timestep_{f}_loss"
            ] = frame_loss
        # GSP Loss
        gsp_loss = self.gsp_criterion(y[GSP_YIELD], gsp_y_hat)
        self.log_dict({f"{'train' if is_training else 'val'}/gsp_loss": gsp_loss})
        for f in range(self.forecast_gsp_steps):
            frame_loss = self.gsp_criterion(gsp_y_hat[:, f, :], y[GSP_YIELD][:, f, :]).item()
            frame_loss_dict[
                f"{'train' if is_training else 'val'}/gsp_timestep_{f}_loss"
            ] = frame_loss
        self.log_dict(frame_loss_dict)
        return loss + gsp_loss

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

    def construct_query(self, x: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.query(x), self.pv_query(x)

    def forward(self, x, mask=None, query=None):
        return self.model.forward(x, mask=mask, queries=query)
