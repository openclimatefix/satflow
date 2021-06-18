import torch
from satflow.models.base import register_model, Model
from typing import Dict, Any
import pytorch_lightning as pl


@register_model
class MetNet(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        return MetNet(config=config)
