from transformers import PerceiverForOpticalFlow, PerceiverForMultimodalAutoencoding, PerceiverModel, PerceiverConfig, PerceiverForImageClassificationLearned
import torch
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

class HuggingFacePerceiver(BaseModel):

    def __init__(self, input_size: int = 224):
        self.model = PerceiverForOpticalFlow.from_pretrained('deepmind/optical-flow-perceiver', ignore_mismatched_sizes=True, train_size=[input_size, input_size])

    def forward(self, x, **kwargs) -> Any:
        return model(inputs=x)
