import torch
import torch.nn.functional as F
from typing import Union
from satflow.models.losses import FocalLoss, get_loss
import numpy as np
import pytorch_lightning as pl

from satflow.models.base import register_model
from satflow.models.gan.common import LatentConditioningStack, ContextConditioningStack
from satflow.models.gan.generators import NowcastingSampler
from satflow.models.gan.discriminators import (
    NowcastingSpatialDiscriminator,
    NowcastingTemporalDiscriminator,
)


@register_model
class NowcastingGAN(pl.LightningModule):
    def __init__(
        self,
        forecast_steps: int,
        input_channels: int = 3,
        num_layers: int = 5,
        hidden_dim: int = 64,
        bilinear: bool = False,
        lr: float = 0.001,
        make_vis: bool = False,
        loss: Union[str, torch.nn.Module] = "mse",
        pretrained: bool = False,
    ):
        """
        Nowcasting GAN is an attempt to recreate DeepMind's Skillful Nowcasting GAN from https://arxiv.org/abs/2104.00954
        but slightly modified for multiple satellite channels
        Args:
            forecast_steps:
            input_channels:
            num_layers:
            hidden_dim:
            bilinear:
            lr:
            make_vis:
            loss:
            pretrained:
        """
        super(NowcastingGAN, self).__init__()
        self.lr = lr
        self.criterion = get_loss(loss)
        self.make_vis = make_vis
        self.input_channels = input_channels
        self.model = NowcastingGAN(
            forecast_steps, input_channels, num_layers, hidden_dim, bilinear
        )
        self.save_hyperparameters()

    @classmethod
    def from_config(cls, config):
        return NowcastingGAN(
            forecast_steps=config.get("forecast_steps", 12),
            input_channels=config.get("in_channels", 12),
            hidden_dim=config.get("features", 64),
            num_layers=config.get("num_layers", 5),
            bilinear=config.get("bilinear", False),
            lr=config.get("lr", 0.001),
        )

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        # DeepSpeedCPUAdam provides 5x to 7x speedup over torch.optim.adam(w)
        # optimizer = torch.optim.adam()
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        if self.make_vis:
            if np.random.random() < 0.001:
                self.visualize(x, y, y_hat, batch_idx)
        # Generally only care about the center x crop, so the model can take into account the clouds in the area without
        # being penalized for that, but for now, just do general MSE loss, also only care about first 12 channels
        loss = self.criterion(y_hat, y)
        self.log("train/loss", loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.criterion(y_hat, y)
        self.log("val/loss", val_loss, on_step=True, on_epoch=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss
