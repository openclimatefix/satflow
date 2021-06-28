from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from satflow.models.base import register_model
from satflow.models.layers.ConvLSTM import ConvLSTMCell


@register_model
class EncoderDecoderConvLSTM(pl.LightningModule):
    def __init__(
        self, num_hidden, in_channels, out_channels, forecast_steps, learning_rate, make_vis=False
    ):
        super(EncoderDecoderConvLSTM, self).__init__()
        self.forecast_steps = forecast_steps
        self.lr = learning_rate
        self.make_vis = make_vis
        """ ARCHITECTURE

        # Encoder (ConvLSTM)
        # Encoder Vector (final hidden state of encoder)
        # Decoder (ConvLSTM) - takes Encoder Vector as input
        # Decoder (3D CNN) - produces regression predictions for our model

        """
        self.encoder_1_convlstm = ConvLSTMCell(
            input_dim=in_channels, hidden_dim=num_hidden, kernel_size=(3, 3), bias=True
        )

        self.encoder_2_convlstm = ConvLSTMCell(
            input_dim=num_hidden, hidden_dim=num_hidden, kernel_size=(3, 3), bias=True
        )

        self.decoder_1_convlstm = ConvLSTMCell(
            input_dim=num_hidden, hidden_dim=num_hidden, kernel_size=(3, 3), bias=True  # nf + 1
        )

        self.decoder_2_convlstm = ConvLSTMCell(
            input_dim=num_hidden, hidden_dim=num_hidden, kernel_size=(3, 3), bias=True
        )

        self.decoder_CNN = nn.Conv3d(
            in_channels=num_hidden,
            out_channels=out_channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
        )
        self.save_hyperparameters()

    @classmethod
    def from_config(cls, config):
        return EncoderDecoderConvLSTM(
            num_hidden=config.get("num_hidden", 64),
            in_channels=config.get("in_channels", 12),
            out_channels=config.get("out_channels", 1),
            forecast_steps=config.get("forecast_steps", 1),
            learning_rate=config.get("learning_rate", 0.001),
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

    def forward(self, x, future_seq=0, hidden_state=None):

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
            x, seq_len, future_seq, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4
        )

        return outputs

    def configure_optimizers(self):
        # DeepSpeedCPUAdam provides 5x to 7x speedup over torch.optim.adam(w)
        # optimizer = torch.optim.adam()
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x, self.forecast_steps)
        # Generally only care about the center x crop, so the model can take into account the clouds in the area without
        # being penalized for that, but for now, just do general MSE loss, also only care about first 12 channels
        # the logger you used (in this case tensorboard)
        if self.make_vis:
            self.visualize(x, y, y_hat, batch_idx)
        loss = F.mse_loss(y_hat, y)
        self.log("train/loss", loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x, self.forecast_steps)
        val_loss = F.mse_loss(y_hat, y)
        self.log("val/loss", val_loss, on_step=True, on_epoch=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x, self.forecast_steps)
        loss = F.mse_loss(y_hat, y)
        return loss

    def visualize(self, x, y, y_hat, batch_idx):
        tensorboard = self.logger.experiment[0]
        # print(tensorboard)
        # Add all the different timesteps for a single prediction, 0.1% of the time
        in_image = x[0].cpu().detach().numpy()  # Input image stack
        for i, in_slice in enumerate(in_image):
            for j, in_channel in enumerate(in_slice):
                tensorboard.add_image(
                    f"Input_Image_{i}_Channel_{j}",
                    np.expand_dims(in_channel, axis=0),
                    global_step=batch_idx,
                )  # Each Channel
        out_image = y_hat[0].cpu().detach().numpy()
        for i, out_slice in enumerate(out_image):
            for j, out_channel in enumerate(out_slice):
                tensorboard.add_image(
                    f"Output_Image_{i}_Channel_{j}",
                    np.expand_dims(out_channel, axis=0),
                    global_step=batch_idx,
                )  # Each Channel
        out_image = y[0].cpu().detach().numpy()
        for i, out_slice in enumerate(out_image):
            for j, out_channel in enumerate(out_slice):
                tensorboard.add_image(
                    f"Target_{i}_Channel_{j}",
                    np.expand_dims(out_channel, axis=0),
                    global_step=batch_idx,
                )  # Each Channel
