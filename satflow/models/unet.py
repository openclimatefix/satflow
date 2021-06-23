import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from satflow.models.base import register_model
from pl_bolts.models.vision import UNet
import numpy as np

@register_model
class Unet(pl.LightningModule):
    def __init__(
        self,
        forecast_steps: int,
        input_channels: int = 3,
        num_layers: int = 5,
        features_start: int = 64,
        bilinear: bool = False,
        learning_rate: float = 0.001,
    ):
        super(Unet, self).__init__()
        self.lr = learning_rate
        self.model = UNet(forecast_steps, input_channels, num_layers, features_start, bilinear)

    @classmethod
    def from_config(cls, config):
        return Unet(
            forecast_steps=config.get("forecast_steps", 12),
            input_channels=config.get("in_channels", 12),
            features_start=config.get("features", 64),
            num_layers=config.get("num_layers", 5),
            bilinear=config.get("bilinear", False),
            learning_rate=config.get("learning_rate", 0.001),
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
        # the logger you used (in this case tensorboard)
        tensorboard = self.logger.experiment
        # Add all the different timesteps for a single prediction, 0.1% of the time
        if np.random.random() < 0.001:
            in_image = x[0] # Input image stack
            for i, in_slice in enumerate(in_image):
                for j, in_channel in enumerate(in_slice):
                    tensorboard.add_image(f"Input_Image_{i}_Channel_{j}", in_channel, global_step=batch_idx) # Each Channel
            out_image = y_hat[0]
            for i, out_slice in enumerate(out_image):
                for j, out_channel in enumerate(out_slice):
                    tensorboard.add_image(f"Output_Image_{i}_Channel_{j}", out_channel, global_step=batch_idx) # Each Channel

        # Generally only care about the center x crop, so the model can take into account the clouds in the area without
        # being penalized for that, but for now, just do general MSE loss, also only care about first 12 channels
        loss = F.mse_loss(y_hat, y)
        self.log("train/loss", loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.mse_loss(y_hat, y)
        self.log("val/loss", val_loss, on_step=True, on_epoch=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x, self.forecast_steps)
        loss = F.mse_loss(y_hat, y)
        return loss
