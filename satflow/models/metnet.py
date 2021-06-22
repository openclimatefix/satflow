from typing import Any, Dict

import pytorch_lightning as pl
import torch

from satflow.models.base import Model, register_model


@register_model
class MetNet(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        return MetNet(config=config)
