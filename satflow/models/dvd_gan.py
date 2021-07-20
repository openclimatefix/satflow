import torch
import torch.nn.functional as F
from typing import Union
from satflow.models.losses import FocalLoss
import numpy as np
import pytorch_lightning as pl

from satflow.models.base import register_model


@register_model
class DVDGan(pl.LightningModule):
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
        super(DVDGan, self).__init__()
        self.lr = lr
        assert loss in ["mse", "bce", "binary_crossentropy", "crossentropy", "focal"]
        if loss == "mse":
            self.criterion = F.mse_loss
        elif loss in ["bce", "binary_crossentropy", "crossentropy"]:
            self.criterion = F.nll_loss
        elif loss in ["focal"]:
            self.criterion = FocalLoss()
        else:
            raise ValueError(f"loss {loss} not recognized")
        self.make_vis = make_vis
        self.input_channels = input_channels
        self.model = DVDGan(forecast_steps, input_channels, num_layers, hidden_dim, bilinear)
        self.save_hyperparameters()

    @classmethod
    def from_config(cls, config):
        return DVDGan(
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
