import logging
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import einops
import pandas as pd
import torch
import torch.nn.functional as F
import torch_optimizer as optim
from einops import rearrange, repeat
from nowcasting_dataloader.batch import BatchML
from nowcasting_dataset.consts import (
    DEFAULT_N_GSP_PER_EXAMPLE,
    DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE,
    GSP_DATETIME_INDEX,
    GSP_ID,
    GSP_YIELD,
    NWP_DATA,
    PV_SYSTEM_ID,
    PV_YIELD,
    SATELLITE_DATA,
    TOPOGRAPHIC_DATA,
)
from nowcasting_utils.metrics.validation import (
    make_validation_results,
    save_validation_results_to_logger,
)
from nowcasting_utils.models.base import BaseModel, register_model
from nowcasting_utils.models.loss import get_loss
from nowcasting_utils.visualization.line import plot_batch_results
from nowcasting_utils.visualization.visualization import plot_example
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from transformers import (
    PerceiverConfig,
    PerceiverForImageClassificationLearned,
    PerceiverForMultimodalAutoencoding,
    PerceiverForOpticalFlow,
    PerceiverModel,
)

logger = logging.getLogger("satflow.model")
logger.setLevel(logging.WARN)

HRV_KEY = "hrv_" + SATELLITE_DATA


class HuggingFacePerceiver(BaseModel):
    def __init__(self, input_size: int = 224):
        self.model = PerceiverForOpticalFlow.from_pretrained(
            "deepmind/optical-flow-perceiver",
            ignore_mismatched_sizes=True,
            train_size=[input_size, input_size],
        )

        self.channel_change = torch.nn.Conv2d(in_channels=2, out_channels=11)
        self.predict_satellite = False
        self.predict_hrv_satellite = True
        self.hrv_channel_change = torch.nn.Conv2d(in_channels=2, out_channels=1)

    def forward(self, x, **kwargs) -> Any:
        return model(inputs=x)

    def _train_or_validate_step(self, batch, batch_idx, is_training: bool = True):
        x, y = batch
        # Now run predictions for all the queries
        # Predicting all future ones at once
        losses = []
        if self.predict_satellite:
            sat_y_hat = self.model(inputs=x)
            sat_y_hat = self.channel_change(sat_y_hat)
            # Satellite losses
            sat_loss, sat_frame_loss = self.mse(hrv_sat_y_hat, y[SATELLITE_DATA])
            losses.append(sat_loss)
        if self.predict_hrv_satellite:
            hrv_sat_y_hat = self.model(inputs=x)
            hrv_sat_y_hat = self.hrv_channel_change(hrv_sat_y_hat)
            # HRV Satellite losses
            hrv_sat_loss, sat_frame_loss = self.mse(hrv_sat_y_hat, y[HRV_KEY])
            losses.append(hrv_sat_loss)
        loss = losses[0]
        for sat_loss in losses[1:]:
            loss += sat_loss
        self.log_dict({f"{'train' if is_training else 'val'}/loss": loss})
        if is_training:
            return loss
        else:
            # Return the model outputs as well
            return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
