from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from nowcasting_utils.models.base import register_model
from nowcasting_utils.models.loss import get_loss
from pl_bolts.models.vision import UNet


@register_model
class Unet(pl.LightningModule):
    def __init__(
        self,
        forecast_steps: int,
        input_channels: int = 3,
        num_layers: int = 5,
        hidden_dim: int = 64,
        bilinear: bool = False,
        lr: float = 0.001,
        visualize: bool = False,
        loss: Union[str, torch.nn.Module] = "mse",
        pretrained: bool = False,
    ):
        super(Unet, self).__init__()
        self.lr = lr
        self.input_channels = input_channels
        self.forecast_steps = forecast_steps
        self.criterion = get_loss(loss=loss)
        self.visualize = visualize
        self.model = UNet(forecast_steps, input_channels, num_layers, hidden_dim, bilinear)
        self.save_hyperparameters()

    @classmethod
    def from_config(cls, config):
        return Unet(
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
        x = x.float()
        y_hat = self(x)

        if self.visualize:
            if np.random.random() < 0.01:
                self.visualize_step(x, y, y_hat, batch_idx)
        # Generally only care about the center x crop, so the model can take into account the clouds in the area without
        # being penalized for that, but for now, just do general MSE loss, also only care about first 12 channels
        loss = self.criterion(y_hat, y)
        self.log("train/loss", loss, on_step=True)
        frame_loss_dict = {}
        for f in range(self.forecast_steps):
            frame_loss = self.criterion(y_hat[:, f, :, :], y[:, f, :, :]).item()
            frame_loss_dict[f"train/frame_{f}_loss"] = frame_loss
        self.log_dict(frame_loss_dict)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y_hat = self(x)
        val_loss = self.criterion(y_hat, y)
        self.log("val/loss", val_loss)
        # Save out loss per frame as well
        frame_loss_dict = {}
        for f in range(self.forecast_steps):
            frame_loss = self.criterion(y_hat[:, f, :, :], y[:, f, :, :]).item()
            frame_loss_dict[f"val/frame_{f}_loss"] = frame_loss
        self.log_dict(frame_loss_dict)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss

    def visualize_step(self, x, y, y_hat, batch_idx, step="train"):
        tensorboard = self.logger.experiment[0]
        # Add all the different timesteps for a single prediction, 0.1% of the time
        images = x[0].cpu().detach()
        images = [torch.unsqueeze(img, dim=0) for img in images]
        image_grid = torchvision.utils.make_grid(images, nrow=self.channels_per_timestep)
        tensorboard.add_image(f"{step}/Input_Image_Stack", image_grid, global_step=batch_idx)
        images = y[0].cpu().detach()
        images = [torch.unsqueeze(img, dim=0) for img in images]
        image_grid = torchvision.utils.make_grid(images, nrow=12)
        tensorboard.add_image(f"{step}/Target_Image_Stack", image_grid, global_step=batch_idx)
        images = y_hat[0].cpu().detach()
        images = [torch.unsqueeze(img, dim=0) for img in images]
        image_grid = torchvision.utils.make_grid(images, nrow=12)
        tensorboard.add_image(f"{step}/Generated_Image_Stack", image_grid, global_step=batch_idx)
