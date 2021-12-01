import logging
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import einops
import torch
import torch_optimizer as optim
import torch.nn.functional as F
from einops import rearrange, repeat
from nowcasting_dataset.consts import (
    DEFAULT_N_GSP_PER_EXAMPLE,
    DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE,
    GSP_ID,
    GSP_YIELD,
    GSP_DATETIME_INDEX,
    NWP_DATA,
    PV_SYSTEM_ID,
    PV_YIELD,
    SATELLITE_DATA,
    TOPOGRAPHIC_DATA,
)
from nowcasting_utils.models.base import BaseModel, register_model
from nowcasting_utils.models.loss import get_loss
from perceiver_pytorch import MultiPerceiver
from perceiver_pytorch.decoders import ImageDecoder
from perceiver_pytorch.encoders import ImageEncoder
from perceiver_pytorch.modalities import InputModality
from perceiver_pytorch.queries import LearnableQuery
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from nowcasting_utils.visualization.visualization import plot_example
from nowcasting_utils.metrics.validation import make_validation_results, save_validation_results_to_logger
from nowcasting_dataloader.batch import BatchML
from nowcasting_utils.visualization.line import plot_batch_results
import pandas as pd

logger = logging.getLogger("satflow.model")
logger.setLevel(logging.WARN)

HRV_KEY = "hrv_" + SATELLITE_DATA


@register_model
class JointPerceiver(BaseModel):
    def __init__(
        self,
        input_channels: int = 22,
        sat_channels: int = 12,
        nwp_channels: int = 10,
        base_channels: int = 1,
        forecast_steps: int = 48,
        gsp_forecast_steps: int = 12,
        sat_input_size: int = 24,
        hrv_sat_input_size: int = 64,
        nwp_input_size: int = 64,
        topo_input_size: int = 64,
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
        sine_only: bool = False,
        encode_fourier: bool = False,
        preprocessor_type: Optional[str] = None,
        postprocessor_type: Optional[str] = None,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        decoder_kwargs: Optional[Dict[str, Any]] = None,
        pretrained: bool = False,
        predict_timesteps_together: bool = False,
        nwp_modality: bool = False,
        sat_modality: bool = True,
        hrv_sat_modality: bool = True,
        topographic_modality: bool = False,
        use_learnable_query: bool = True,
        generate_fourier_features: bool = True,
        pv_modality: bool = False,
        predict_satellite: bool = False,
        predict_hrv_satellite: bool = False,
        number_fourier_bands: int = 16,
    ):
        """
        Joint Satellite Image + GSP PV Output prediction model

        Args:
            input_channels: Total number of input channels to the model
            sat_channels: Number of satellite channels
            nwp_channels: Number of NWP channels
            base_channels: Number of channels in the base map (i.e. Topographic data)
            forecast_steps: Number of satellite forecast steps
            gsp_forecast_steps: Number of forecast steps for GSPs
            sat_input_size: Input size in pixels for satellite/NWP/Basemap images
            lr: Learning rate
            visualize: Whether to visualize the output
            max_frequency: Max frequency for the Fourier Features
            depth: Depth of the PerceiverIO
            num_latents: Number of latents
            cross_heads: Number of cross-attention heads
            latent_heads: Number of latent heads
            cross_dim_heads: Dimension of the cross-attention heads
            latent_dim: Dimension of the latent space
            weight_tie_layers: Whether to weight tie layers
            decoder_ff: Whether to have a feedforward at the end of the decoder in PerceiverIO
            dim: Dimension
            logits_dim: Dimension of the logits in PerceiverIO
            queries_dim: Query dimension in PerceiverIO
            latent_dim_heads: Number of latent
            loss: Satellite image loss function
            gsp_loss: GSP PV output loss function
            sine_only: Whether to use sin-only for the Fourier Features or not
            encode_fourier: Whether to encode the inputs with fourier features
            preprocessor_type: Type of preprocessor for the image inputs
            postprocessor_type: Type of postprocessor for the image outputs
            encoder_kwargs: Preprocessor encoder kwargs
            decoder_kwargs: Preprocessor decoder kwargs
            pretrained: Whether to download a pre-trained model from HuggingFace, default False
            predict_timesteps_together: Whether to predict all future timesteps at once or individually
            nwp_modality: Whether NWPs are being included
            use_learnable_query: Whether to use the LearnableQuery
            generate_fourier_features: Whether to generate Fourier Features in the LearnableQuery
            pv_modality: Whether to use PV data
            predict_satellite: Whether to predict non-HRV satellite imagery
            predict_hrv_satellite: Whether to predict HRV satellite imagery
            number_fourier_bands: Number of Fourier bands given to the dataloader for the inputs
        """
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
        self.sat_input_size = sat_input_size
        self.nwp_input_size = nwp_input_size
        self.hrv_sat_input_size = hrv_sat_input_size
        self.topo_input_size = topo_input_size
        self.predict_timesteps_together = predict_timesteps_together
        self.use_learnable_query = use_learnable_query
        self.max_frequency = max_frequency
        self.use_pv_data = pv_modality
        self.sat_modality = sat_modality
        self.hrv_sat_modality = hrv_sat_modality
        self.predict_satellite = predict_satellite
        self.predict_hrv_satellite = predict_hrv_satellite
        self.gsp_forecast_steps = gsp_forecast_steps

        if use_learnable_query:
            self.gsp_query = LearnableQuery(
                channel_dim=queries_dim,
                query_shape=(self.forecast_steps, 1)  # Only need one number for GSP
                if predict_timesteps_together
                else (1, 1),
                conv_layer="2d",
                max_frequency=max_frequency,
                num_frequency_bands=sat_input_size,
                sine_only=sine_only,
                generate_fourier_features=generate_fourier_features,
            )
            if self.predict_satellite:
                self.sat_query = LearnableQuery(
                    channel_dim=queries_dim,
                    query_shape=(self.forecast_steps, self.sat_input_size, self.sat_input_size)
                    if predict_timesteps_together
                    else (self.sat_input_size, self.sat_input_size),
                    conv_layer="3d",
                    max_frequency=max_frequency,
                    num_frequency_bands=sat_input_size,
                    sine_only=sine_only,
                    generate_fourier_features=generate_fourier_features,
                )
            if self.predict_hrv_satellite:
                self.hrv_sat_query = LearnableQuery(
                    channel_dim=queries_dim,
                    query_shape=(
                        self.forecast_steps,
                        self.hrv_sat_input_size,
                        self.hrv_sat_input_size,
                    )
                    if predict_timesteps_together
                    else (self.hrv_sat_input_size, self.hrv_sat_input_size),
                    conv_layer="3d",
                    max_frequency=max_frequency,
                    num_frequency_bands=sat_input_size,
                    sine_only=sine_only,
                    generate_fourier_features=generate_fourier_features,
                )
        else:
            self.gsp_query = None
            self.sat_query = None
            self.hrv_sat_query = None

        # Warn if using frequency is smaller than Nyquist Frequency
        if max_frequency < sat_input_size / 2:
            print(
                f"Max frequency is less than Nyquist frequency, currently set to {max_frequency}"
                f" while the Nyquist frequency for input of size {sat_input_size} is {sat_input_size / 2}"
            )

        # Preprocessor, if desired, on top of the other processing done
        if preprocessor_type is not None:
            if preprocessor_type not in ("conv", "patches", "pixels", "conv1x1", "metnet"):
                raise ValueError("Invalid prep_type!")
            if preprocessor_type == "metnet":
                # MetNet processing
                self.preprocessor = ImageEncoder(
                    crop_size=sat_input_size,
                    prep_type="metnet",
                )
            else:
                self.preprocessor = ImageEncoder(
                    input_channels=sat_channels,
                    prep_type=preprocessor_type,
                    **encoder_kwargs
                )
        else:
            self.preprocessor = None

        # The preprocessor will change the number of channels in the input
        modalities = []
        if sat_modality:
            # TODO Get the actual input channels for all the inputs and put that in directly
            # Timeseries input
            sat_modality = InputModality(
                name=SATELLITE_DATA,
                input_channels=number_fourier_bands * 4
                + 2
                + 13
                + 13
                + 11,  # Spatial features + Datetime + Datetime + 11 Sat channels
                input_axis=3,  # number of axes, 3 for video
                num_freq_bands=2 * sat_input_size
                + 1,  # number of freq bands, with original value (2 * K + 1)
                max_freq=max_frequency,  # maximum frequency, hyperparameter depending on how fine the data is, should be Nyquist frequency (i.e. 112 for 224 input image)
                sin_only=sine_only,  # Whether if sine only for Fourier encoding, TODO test more
                fourier_encode=False,  # Whether to encode position with Fourier features
            )
            modalities.append(sat_modality)
        if hrv_sat_modality:
            hrv_sat_modality = InputModality(
                name=HRV_KEY,
                input_channels=36 + 32,  # DAtetime+Spatial ++ 1 HRV and patches 4x4x2 = 32 total
                input_axis=3,  # number of axes, 3 for video
                num_freq_bands=2 * hrv_sat_input_size
                + 1,  # number of freq bands, with original value (2 * K + 1)
                max_freq=max_frequency,  # maximum frequency, hyperparameter depending on how fine the data is, should be Nyquist frequency (i.e. 112 for 224 input image)
                sin_only=sine_only,  # Whether if sine only for Fourier encoding, TODO test more
                fourier_encode=False,  # Whether to encode position with Fourier features
            )
            modalities.append(hrv_sat_modality)

        if nwp_modality:
            nwp_modality = InputModality(
                name=NWP_DATA,
                input_channels=36 + 320,  # Spatial features + Datetime + Datetime + 10 * patchsize (4x4x2 = 32 * 10 = 320) NWP channels,
                input_axis=3,  # number of axes, 3 for video
                num_freq_bands=2 * nwp_input_size
                + 1,  # number of freq bands, with original value (2 * K + 1)
                max_freq=max_frequency,  # maximum frequency, hyperparameter depending on how fine the data is, should be Nyquist frequency (i.e. 112 for 224 input image)
                sin_only=sine_only,  # Whether if sine only for Fourier encoding, TODO test more
                fourier_encode=False,  # Whether to encode position with Fourier features
            )
            modalities.append(nwp_modality)

        if topographic_modality:
            # Use image modality for latlon, elevation, other base data?
            image_modality = InputModality(
                name=TOPOGRAPHIC_DATA,
                input_channels=number_fourier_bands * 4
                + 2
                + 1,  # Spatial features + 1 Topo channel
                input_axis=2,  # number of axes, 2 for images
                num_freq_bands=2 * topo_input_size
                + 1,  # number of freq bands, with original value (2 * K + 1)
                max_freq=max_frequency,  # maximum frequency, hyperparameter depending on how fine the data is
                sin_only=sine_only,
                fourier_encode=False,
            )
            modalities.append(image_modality)

        gsp_id_modality = InputModality(
            name=GSP_ID,
            input_channels=1,
            input_axis=1,
            num_freq_bands=DEFAULT_N_GSP_PER_EXAMPLE,  # number of freq bands, with original value (2 * K + 1)
            max_freq=2 * DEFAULT_N_GSP_PER_EXAMPLE
            + 1,  # maximum frequency, hyperparameter depending on how fine the data is
            sin_only=sine_only,
            fourier_encode=False,
        )
        modalities.append(gsp_id_modality)

        if pv_modality:
            # Sort audio for timestep one-hot encode? Or include under other modality?
            pv_modality = InputModality(
                name=PV_YIELD,
                input_channels=37,  # number of channels for mono audio
                input_axis=2,  # number of axes, 2 for images, or in this case System*timestemp
                num_freq_bands=self.forecast_steps,  # number of freq bands, with original value (2 * K + 1)
                max_freq=max_frequency,  # maximum frequency, hyperparameter depending on how fine the data is
                sin_only=sine_only,
                fourier_encode=False,
            )
            modalities.append(pv_modality)
            pv_id_modality = InputModality(
                name=PV_SYSTEM_ID,
                input_channels=1,  # number of channels for mono audio
                input_axis=1,  # number of axes, 2 for images
                num_freq_bands=DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE,  # number of freq bands, with original value (2 * K + 1)
                max_freq=2 * DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE
                + 1,  # maximum frequency, hyperparameter depending on how fine the data is
                sin_only=sine_only,
                fourier_encode=False,  # IDs have no spatial area, so just normal fourier encoding
            )
            modalities.append(pv_id_modality)
        self.modalities = modalities
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
            sine_only=sine_only,
            fourier_encode_data=encode_fourier,
            output_shape=sat_input_size,  # TODO Change Shape of output to make the correct sized logits dim, needed so reshaping works
            decoder_ff=decoder_ff,  # Optional decoder FF
        )

        self.model = self.model.float()
        self.gsp_linear = torch.nn.Linear(144 * 288, self.gsp_forecast_steps).float()
        if postprocessor_type is not None:
            if postprocessor_type not in ("conv", "patches", "pixels", "conv1x1"):
                raise ValueError("Invalid postprocessor_type!")
            self.postprocessor = ImageDecoder(
                postprocess_type=postprocessor_type, output_channels=sat_channels, **decoder_kwargs
            )
        else:
            self.postprocessor = None
        self.num_sat_timesteps = 2
        self.save_hyperparameters()

    def encode_inputs(self, x: dict) -> Dict[str, torch.Tensor]:
        x = self.remove_non_modalities(x)
        for key in [SATELLITE_DATA, HRV_KEY, NWP_DATA]:
            if len(x.get(key, [])) > 0:
                if key in [SATELLITE_DATA, HRV_KEY]:
                    # Subselect the last two frames for simpler OF style
                    x[key] = x[key][:,:,-self.num_sat_timesteps:]
                    # Split out position encoding from data values
                    x[key] = x[key].permute(0, 2, 1, 3, 4)  # Channels last
                    sat_data = x[key][:,:,:1]
                    sat_pos_encoding = x[key][
                                             :,
                                             :: 2,
                                             1:,
                                             :: 4,
                                             :: 4,
                                             ]
                    sat_data = self.run_preprocessor(sat_data)
                    #sat_data = torch.unsqueeze(sat_data, dim=1)
                    x[key] = torch.cat([sat_data, sat_pos_encoding], dim=2)
                    x[key] = x[key].permute(0, 1, 3, 4, 2)  # Channels last
                if key in [NWP_DATA]:
                    x[key] = x[key][:,:,-self.num_sat_timesteps:]
                    # Split out position encoding from data values
                    x[key] = x[key].permute(0, 2, 1, 3, 4)  # Channels last
                    # NWP has 10 channels in current one
                    print(x[key].shape)
                    sat_data = x[key][:,:,:self.nwp_channels]
                    sat_pos_encoding = x[key][
                                       :,
                                       :: 2,
                                       self.nwp_channels:,
                                       :: 4,
                                       :: 4,
                                       ]
                    sat_data = self.run_preprocessor(sat_data)
                    print(sat_data.shape)
                    #sat_data = torch.unsqueeze(sat_data, dim=1)
                    x[key] = torch.cat([sat_data, sat_pos_encoding], dim=2)
                    x[key] = x[key].permute(0, 1, 3, 4, 2)  # Channels last

        for key in [GSP_ID, PV_SYSTEM_ID]:
            if len(x.get(key, [])) > 0:
                x[key] = torch.unsqueeze(x[key], dim=2)
        for key in [TOPOGRAPHIC_DATA]:
            if len(x.get(key, [])) > 0:
                x[key] = torch.squeeze(x[key], dim=2).permute(0, 2, 3, 1)
        for key in [PV_YIELD]:
            if len(x.get(key, [])) > 0:
                x[key] = x[key].permute(0, 2, 3, 1) # Channels last
        return x

    def run_preprocessor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Run the processing step for a tensor

        Args:
            tensor: Teh tensor to transform

        Returns:
            The preprocessed tensor
        """
        if self.preprocessor is not None:
            tensor = self.preprocessor(tensor)
        return tensor

    def predict_satellite_imagery(
        self, x: dict, query: torch.Tensor, output_size: int
    ) -> torch.Tensor:
        """
        Run the predictions for satellite imagery, and optionally postprocesses them

        Args:
            x: Input data
            query: Query to use
            output_size: Size of the image output

        Returns:
            The reshaped output from the query, optioanlly postprocessed
        """
        y_hat = self(x, query=query)
        y_hat = rearrange(
            y_hat,
            "b (t h w) c -> b c t h w",
            t=self.forecast_steps,
            h=output_size,
            w=output_size,
        )
        if self.postprocessor is not None:
            y_hat = self.postprocessor(y_hat)

        return y_hat

    def compute_per_timestep_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor, key: str, is_training: bool
    ) -> Tuple[torch.Tensor, dict]:
        """
        Computes the per timestep loss for predicted imagery

        Args:
            predictions: Predictions Tensor
            targets: Ground truth Tensor
            is_training: Key for saving the loss

        Returns:
            Dictionary of the loss for the per-timestep and the overall loss
        """

        # Satellite Loss
        loss = self.criterion(targets, predictions)
        self.log_dict({f"{'train' if is_training else 'val'}/{key}_loss": loss})
        frame_loss_dict = {}
        for f in range(self.forecast_steps):
            frame_loss = self.criterion(predictions[:, f, :, :, :], targets[:, f, :, :, :]).item()
            frame_loss_dict[
                f"{'train' if is_training else 'val'}/{key}_timestep_{f}_loss"
            ] = frame_loss
        return loss, frame_loss_dict

    def remove_non_modalities(self, x: dict) -> dict[str, torch.Tensor]:
        """
        Remove keys that are not modalities
        Args:
            x: Dictionary of inputs

        Returns:
            Cleaned dictionary without the keys that are not modalities
        """
        keys_to_keep = []
        for modality in self.modalities:
            keys_to_keep.append(modality.name)
        keys_to_remove = []
        for k in x.keys():
            if k not in keys_to_keep:
                keys_to_remove.append(k)
        for k in keys_to_remove:
            x.pop(k, None)
        return x

    def _train_or_validate_step(self, batch, batch_idx, is_training: bool = True):
        x, y = batch
        gsp_query, sat_query, hrv_sat_query = self.construct_query(x)
        x = self.encode_inputs(x)
        # Now run predictions for all the queries
        # Predicting all future ones at once
        frame_loss_dict = {}
        losses = []
        if self.predict_satellite:
            sat_y_hat = self.predict_satellite_imagery(x, sat_query.float(), self.sat_input_size)
            # Satellite losses
            sat_loss, sat_frame_loss = self.compute_per_timestep_loss(
                predictions=sat_y_hat, targets=y[SATELLITE_DATA], key="sat", is_training=is_training
            )
            losses.append(sat_loss)
            frame_loss_dict.update(sat_frame_loss)
        if self.predict_hrv_satellite:
            hrv_sat_y_hat = self.predict_satellite_imagery(
                x, hrv_sat_query.float(), self.hrv_sat_input_size
            )
            # HRV Satellite losses
            hrv_sat_loss, sat_frame_loss = self.compute_per_timestep_loss(
                predictions=hrv_sat_y_hat,
                targets=y[HRV_KEY],
                key="hrv_sat",
                is_training=is_training,
            )
            losses.append(hrv_sat_loss)
            frame_loss_dict.update(sat_frame_loss)

        gsp_y_hat = self(x, query=gsp_query.float())
        # GSP Loss
        # Final linear layer from query shape down to GSP shape?
        gsp_y_hat = einops.rearrange(gsp_y_hat, "b c t -> b (c t)")
        gsp_y_hat = self.gsp_linear(gsp_y_hat)
        loss = self.gsp_criterion(gsp_y_hat, y[GSP_YIELD])
        self.log_dict({f"{'train' if is_training else 'val'}/gsp_loss": loss, f"{'train' if is_training else 'val'}/gsp_mae": F.l1_loss(gsp_y_hat, y[GSP_YIELD])})
        for f in range(gsp_y_hat.shape[1]):
            frame_loss = self.gsp_criterion(gsp_y_hat[:, f], y[GSP_YIELD][:, f]).item()
            frame_loss_dict[
                f"{'train' if is_training else 'val'}/gsp_timestep_{f}_loss"
            ] = frame_loss
            frame_loss_dict[
                f"{'train' if is_training else 'val'}/gsp_timestep_{f}_mae"
            ] = F.l1_loss(gsp_y_hat[:, f], y[GSP_YIELD][:, f])
        self.log_dict(frame_loss_dict)

        # Plot output
        if not is_training:
            name = f"val/plot/epoch_{self.current_epoch}_{batch_idx}"
            if batch_idx in [0, 1, 2, 3, 4]:

                # 2. plot summary batch of predictions and results
                # make x,y data
                y_hat = gsp_y_hat.cpu().numpy()
                timestamps = y[GSP_DATETIME_INDEX][:,1:].cpu().numpy()
                time = [pd.to_datetime(t) for t in timestamps]
                # plot and save to logger
                fig = plot_batch_results(model_name="joint_model", y=y[GSP_YIELD].cpu().numpy(), y_hat=y_hat, x=time, x_hat=time)
                fig.write_html(f"temp_{batch_idx}.html")
                try:
                    self.logger.experiment[-1][name].upload(f"temp_{batch_idx}.html")
                except:
                    pass
        for sat_loss in losses:
            loss += sat_loss
        self.log_dict({f"{'train' if is_training else 'val'}/loss": loss})
        if is_training:
            return loss
        else:
            # Return the model outputs as well
            return loss, gsp_y_hat

    def validation_step(self, batch, batch_idx):

        loss, model_output = self._train_or_validate_step(batch, batch_idx, is_training = False)

        # save validation results
        capacities = batch[1]["gsp_capacity"].cpu().numpy()
        predictions = model_output.cpu().numpy() * capacities
        truths = batch[1][GSP_YIELD].cpu().numpy() * capacities
        t0_times = batch[1][GSP_DATETIME_INDEX][:,0].cpu().numpy()
        t0_times = pd.to_datetime(t0_times)
        results = make_validation_results(truths_mw=truths,
                                          predictions_mw=predictions,
                                          capacity_mwp=capacities,
                                          gsp_ids=batch[1][GSP_ID].cpu().numpy(),
                                          batch_idx=batch_idx,
                                          t0_datetimes_utc=t0_times)

        # append so in 'validation_epoch_end' the file is saved
        if batch_idx == 0:
            self.results_dfs = []
        self.results_dfs.append(results)

        return loss


    def validation_epoch_end(self, outputs):
        save_validation_results_to_logger(results_dfs=self.results_dfs,
                                          results_file_name="prediction_results",
                                          current_epoch=self.current_epoch,
                                          logger=self.logger)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def construct_query(self, x: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Constructs teh queries for the output

        Args:
            x: Model inputs

        Returns:
            The queries for the output of the perceiver model
        """
        if self.predict_satellite:
            sat_query = x[SATELLITE_DATA + "_query"]
            if self.use_learnable_query:
                sat_query = self.sat_query(x[SATELLITE_DATA], sat_query)
            else:
                # concat to channels of data and flatten axis
                sat_query = einops.rearrange(sat_query, "b ... d -> b (...) d")
        else:
            sat_query = None
        if self.predict_hrv_satellite:
            hrv_sat_query = x[HRV_KEY + "_query"]
            if self.use_learnable_query:
                hrv_sat_query = self.hrv_sat_query(x[HRV_KEY], hrv_sat_query)
            else:
                # concat to channels of data and flatten axis
                hrv_sat_query = einops.rearrange(hrv_sat_query, "b ... d -> b (...) d")
        else:
            hrv_sat_query = None
        gsp_query = x[GSP_YIELD + "_query"]
        if self.use_learnable_query:
            gsp_query = self.gsp_query(x[GSP_ID], gsp_query)
        else:
            # concat to channels of data and flatten axis
            gsp_query = einops.rearrange(gsp_query, "b ... d -> b (...) d")
        return gsp_query, sat_query, hrv_sat_query

    def forward(self, x, mask=None, query=None):
        # TODO Remove enventuallyA
        for key in self.modalities:
            x[key.name] = x[key.name].float()
        return self.model.forward(x, mask=mask, queries=query)
