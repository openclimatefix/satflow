from typing import Any, Dict, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from nowcasting_utils.models.base import register_model
from nowcasting_utils.models.loss import get_loss

from satflow.models.layers.ConvLSTM import ConvLSTMCell


@register_model
class EncoderDecoderConvLSTM(pl.LightningModule):
    def __init__(
        self,
        hidden_dim: int = 64,
        input_channels: int = 12,
        out_channels: int = 1,
        forecast_steps: int = 48,
        lr: float = 0.001,
        visualize: bool = False,
        loss: Union[str, torch.nn.Module] = "mse",
        pretrained: bool = False,
        conv_type: str = "standard",
    ):
        super(EncoderDecoderConvLSTM, self).__init__()
        self.forecast_steps = forecast_steps
        self.criterion = get_loss(loss)
        self.lr = lr
        self.visualize = visualize
        self.model = ConvLSTM(input_channels, hidden_dim, out_channels, conv_type=conv_type)
        self.save_hyperparameters()

    @classmethod
    def from_config(cls, config):
        return EncoderDecoderConvLSTM(
            hidden_dim=config.get("num_hidden", 64),
            input_channels=config.get("in_channels", 12),
            out_channels=config.get("out_channels", 1),
            forecast_steps=config.get("forecast_steps", 1),
            lr=config.get("lr", 0.001),
        )

    def forward(self, x, future_seq=0, hidden_state=None):
        return self.model.forward(x, future_seq, hidden_state)

    def configure_optimizers(self):
        # DeepSpeedCPUAdam provides 5x to 7x speedup over torch.optim.adam(w)
        # optimizer = torch.optim.adam()
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x, self.forecast_steps)
        y_hat = torch.permute(y_hat, dims=(0, 2, 1, 3, 4))
        # Generally only care about the center x crop, so the model can take into account the clouds in the area without
        # being penalized for that, but for now, just do general MSE loss, also only care about first 12 channels
        # the logger you used (in this case tensorboard)
        # if self.visualize:
        #    if np.random.random() < 0.01:
        #        self.visualize_step(x, y, y_hat, batch_idx)
        loss = self.criterion(y_hat, y)
        self.log("train/loss", loss, on_step=True)
        frame_loss_dict = {}
        for f in range(self.forecast_steps):
            frame_loss = self.criterion(y_hat[:, f, :, :, :], y[:, f, :, :, :]).item()
            frame_loss_dict[f"train/frame_{f}_loss"] = frame_loss
        self.log_dict(frame_loss_dict, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x, self.forecast_steps)
        y_hat = torch.permute(y_hat, dims=(0, 2, 1, 3, 4))
        val_loss = self.criterion(y_hat, y)
        # Save out loss per frame as well
        frame_loss_dict = {}
        # y_hat = torch.moveaxis(y_hat, 2, 1)
        for f in range(self.forecast_steps):
            frame_loss = self.criterion(y_hat[:, f, :, :, :], y[:, f, :, :, :]).item()
            frame_loss_dict[f"val/frame_{f}_loss"] = frame_loss
        self.log("val/loss", val_loss, on_step=True, on_epoch=True)
        self.log_dict(frame_loss_dict, on_step=False, on_epoch=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x, self.forecast_steps)
        loss = self.criterion(y_hat, y)
        return loss

    def visualize_step(self, x, y, y_hat, batch_idx, step="train"):
        tensorboard = self.logger.experiment[0]
        # Add all the different timesteps for a single prediction, 0.1% of the time
        if len(x.shape) == 5:
            # Timesteps per channel
            images = x[0].cpu().detach()
            for i, t in enumerate(images):  # Now would be (C, H, W)
                t = [torch.unsqueeze(img, dim=0) for img in t]
                image_grid = torchvision.utils.make_grid(t, nrow=self.input_channels)
                tensorboard.add_image(
                    f"{step}/Input_Image_Stack_Frame_{i}", image_grid, global_step=batch_idx
                )
            images = y[0].cpu().detach()
            for i, t in enumerate(images):  # Now would be (C, H, W)
                t = [torch.unsqueeze(img, dim=0) for img in t]
                image_grid = torchvision.utils.make_grid(t, nrow=self.output_channels)
                tensorboard.add_image(
                    f"{step}/Target_Image_Stack_Frame_{i}", image_grid, global_step=batch_idx
                )
            images = y_hat[0].cpu().detach()
            for i, t in enumerate(images):  # Now would be (C, H, W)
                t = [torch.unsqueeze(img, dim=0) for img in t]
                image_grid = torchvision.utils.make_grid(t, nrow=self.output_channels)
                tensorboard.add_image(
                    f"{step}/Generated_Stack_Frame_{i}", image_grid, global_step=batch_idx
                )


class ConvLSTM(torch.nn.Module):
    def __init__(self, input_channels, hidden_dim, out_channels, conv_type: str = "standard"):
        super().__init__()
        """ ARCHITECTURE

        # Encoder (ConvLSTM)
        # Encoder Vector (final hidden state of encoder)
        # Decoder (ConvLSTM) - takes Encoder Vector as input
        # Decoder (3D CNN) - produces regression predictions for our model

        """
        self.encoder_1_convlstm = ConvLSTMCell(
            input_dim=input_channels,
            hidden_dim=hidden_dim,
            kernel_size=(3, 3),
            bias=True,
            conv_type=conv_type,
        )

        self.encoder_2_convlstm = ConvLSTMCell(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            kernel_size=(3, 3),
            bias=True,
            conv_type=conv_type,
        )

        self.decoder_1_convlstm = ConvLSTMCell(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            kernel_size=(3, 3),
            bias=True,  # nf + 1
            conv_type=conv_type,
        )

        self.decoder_2_convlstm = ConvLSTMCell(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            kernel_size=(3, 3),
            bias=True,
            conv_type=conv_type,
        )

        self.decoder_CNN = nn.Conv3d(
            in_channels=hidden_dim,
            out_channels=out_channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
        )

    def autoencoder(self, x, seq_len, future_step, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4):

        outputs = []

        # encoder
        for t in range(seq_len):
            h_t, c_t = self.encoder_1_convlstm(
                input_tensor=x[:, t, :, :], cur_state=[h_t, c_t]
            )  # we could concat to provide skip conn here
            h_t2, c_t2 = self.encoder_2_convlstm(
                input_tensor=h_t, cur_state=[h_t2, c_t2]
            )  # we could concat to provide skip conn here

        # encoder_vector
        encoder_vector = h_t2

        # decoder
        for t in range(future_step):
            h_t3, c_t3 = self.decoder_1_convlstm(
                input_tensor=encoder_vector, cur_state=[h_t3, c_t3]
            )  # we could concat to provide skip conn here
            h_t4, c_t4 = self.decoder_2_convlstm(
                input_tensor=h_t3, cur_state=[h_t4, c_t4]
            )  # we could concat to provide skip conn here
            encoder_vector = h_t4
            outputs += [h_t4]  # predictions

        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.decoder_CNN(outputs)
        outputs = torch.nn.Sigmoid()(outputs)

        return outputs

    def forward(self, x, forecast_steps=0, hidden_state=None):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """

        # find size of different input dimensions
        b, seq_len, _, h, w = x.size()

        # initialize hidden states
        h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t3, c_t3 = self.decoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t4, c_t4 = self.decoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))

        # autoencoder forward
        outputs = self.autoencoder(
            x, seq_len, forecast_steps, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4
        )

        return outputs
