from typing import Union

import antialiased_cnns
import numpy as np
import pytorch_lightning as pl
import torchvision
from nowcasting_utils.models.base import register_model
from nowcasting_utils.models.loss import get_loss

from satflow.models.layers.RUnetLayers import *


@register_model
class RUnet(pl.LightningModule):
    def __init__(
        self,
        input_channels: int = 12,
        forecast_steps: int = 48,
        recurrent_steps: int = 2,
        loss: Union[str, torch.nn.Module] = "mse",
        lr: float = 0.001,
        visualize: bool = False,
        conv_type: str = "standard",
        pretrained: bool = False,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.forecast_steps = forecast_steps
        self.module = R2U_Net(
            img_ch=input_channels, output_ch=forecast_steps, t=recurrent_steps, conv_type=conv_type
        )
        self.lr = lr
        self.input_channels = input_channels
        self.forecast_steps = forecast_steps
        self.criterion = get_loss(loss=loss)
        self.visualize = visualize
        self.save_hyperparameters()

    @classmethod
    def from_config(cls, config):
        return RUnet(
            forecast_steps=config.get("forecast_steps", 12),
            input_channels=config.get("in_channels", 12),
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


class R2U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, t=2, conv_type: str = "standard"):
        super(R2U_Net, self).__init__()
        if conv_type == "antialiased":
            self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=1)
            self.antialiased = True
        else:
            self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.antialiased = False

        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64, t=t, conv_type=conv_type)
        self.Blur1 = antialiased_cnns.BlurPool(64, stride=2) if self.antialiased else nn.Identity()
        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t, conv_type=conv_type)
        self.Blur2 = antialiased_cnns.BlurPool(128, stride=2) if self.antialiased else nn.Identity()

        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t, conv_type=conv_type)
        self.Blur3 = antialiased_cnns.BlurPool(256, stride=2) if self.antialiased else nn.Identity()

        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t, conv_type=conv_type)
        self.Blur4 = antialiased_cnns.BlurPool(512, stride=2) if self.antialiased else nn.Identity()

        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t, conv_type=conv_type)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t, conv_type=conv_type)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t, conv_type=conv_type)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t, conv_type=conv_type)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t, conv_type=conv_type)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Blur1(x2)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Blur2(x3)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Blur3(x4)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Blur4(x5)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
