from typing import Union

import numpy as np
import pytorch_lightning as pl
import torchvision
from nowcasting_utils.models.base import register_model
from nowcasting_utils.models.loss import get_loss
from nowcasting_utils.models.losses.FocalLoss import FocalLoss

from satflow.models.layers.RUnetLayers import *


@register_model
class AttentionUnet(pl.LightningModule):
    def __init__(
        self,
        input_channels: int = 12,
        forecast_steps: int = 12,
        loss: Union[str, torch.nn.Module] = "mse",
        lr: float = 0.001,
        visualize: bool = False,
        conv_type: str = "standard",
        pretrained: bool = False,
    ):
        super().__init__()
        self.lr = lr
        self.visualize = visualize
        self.input_channels = input_channels
        self.forecast_steps = forecast_steps
        self.channels_per_timestep = 12
        self.model = AttU_Net(
            input_channels=input_channels, output_channels=forecast_steps, conv_type=conv_type
        )
        self.criterion = get_loss(loss)

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
                self.visualize_step(x, y, y_hat, batch_idx, "train")
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

    def visualize_step(self, x, y, y_hat, batch_idx, step):
        # the logger you used (in this case tensorboard)
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


@register_model
class AttentionRUnet(pl.LightningModule):
    def __init__(
        self,
        input_channels: int = 12,
        forecast_steps: int = 12,
        recurrent_blocks: int = 2,
        visualize: bool = False,
        loss: Union[str, torch.nn.Module] = "mse",
        lr: float = 0.001,
        pretrained: bool = False,
    ):
        super().__init__()
        self.lr = lr
        self.input_channels = input_channels
        self.forecast_steps = forecast_steps
        self.channels_per_timestep = 12
        self.model = R2AttU_Net(
            input_channels=input_channels, output_channels=forecast_steps, t=recurrent_blocks
        )
        self.visualize = visualize
        self.criterion = get_loss(loss)

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
                self.visualize_step(x, y, y_hat, batch_idx, "train")
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

    def visualize_step(self, x, y, y_hat, batch_idx, step):
        # the logger you used (in this case tensorboard)
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


class AttU_Net(nn.Module):
    def __init__(self, input_channels=3, output_channels=1, conv_type: str = "standard"):
        super(AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=input_channels, ch_out=64, conv_type=conv_type)
        self.Conv2 = conv_block(ch_in=64, ch_out=128, conv_type=conv_type)
        self.Conv3 = conv_block(ch_in=128, ch_out=256, conv_type=conv_type)
        self.Conv4 = conv_block(ch_in=256, ch_out=512, conv_type=conv_type)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024, conv_type=conv_type)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256, conv_type=conv_type)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512, conv_type=conv_type)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128, conv_type=conv_type)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256, conv_type=conv_type)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64, conv_type=conv_type)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128, conv_type=conv_type)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32, conv_type=conv_type)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64, conv_type=conv_type)

        self.Conv_1x1 = nn.Conv2d(64, output_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class R2AttU_Net(nn.Module):
    def __init__(self, input_channels=3, output_channels=1, t=2, conv_type: str = "standard"):
        super(R2AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=input_channels, ch_out=64, t=t, conv_type=conv_type)

        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t, conv_type=conv_type)

        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t, conv_type=conv_type)

        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t, conv_type=conv_type)

        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t, conv_type=conv_type)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256, conv_type=conv_type)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t, conv_type=conv_type)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128, conv_type=conv_type)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t, conv_type=conv_type)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64, conv_type=conv_type)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t, conv_type=conv_type)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32, conv_type=conv_type)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t, conv_type=conv_type)

        self.Conv_1x1 = nn.Conv2d(64, output_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
