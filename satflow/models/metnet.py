from typing import Any, Dict, Optional
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from satflow.models.base import register_model
from satflow.models.layers import ConvGRU, TimeDistributed
from axial_attention import AxialAttention


class DownSampler(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.module = nn.Sequential(
            nn.Conv2d(in_channels, 160, 3, padding=1),
            nn.MaxPool2d((2, 2), stride=2),
            nn.BatchNorm2d(160),
            nn.Conv2d(160, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.MaxPool2d((2, 2), stride=2),
        )

    def forward(self, x):
        return self.module.forward(x)


@register_model
class MetNet(pl.LightningModule):
    def __init__(
        self,
        image_encoder: str = "downsampler",
        input_channels: int = 12,
        hidden_dim: int = 64,
        kernel_size: int = 3,
        num_layers: int = 1,
        num_att_layers: int = 1,
        head: nn.Module = nn.Identity(),
        forecast_steps: int = 48,
        temporal_dropout: float = 0.2,
        lr: float = 0.001,
        pretrained: bool = False,
    ):
        super().__init__()

        self.horizon = forecast_steps
        self.lr = lr
        self.drop = nn.Dropout(temporal_dropout)
        if image_encoder in ["downsampler", "default"]:
            image_encoder = DownSampler(input_channels + forecast_steps)
        else:
            raise ValueError(f"Image_encoder {image_encoder} is not recognized")
        nf = 256  # from the simple image encoder
        self.image_encoder = TimeDistributed(image_encoder)
        self.ct = ConditionTime(forecast_steps)
        self.temporal_enc = TemporalEncoder(nf, hidden_dim, ks=kernel_size, n_layers=num_layers)
        self.temporal_agg = nn.Sequential(
            *[
                AxialAttention(dim=hidden_dim, dim_index=1, heads=8, num_dimensions=2)
                for _ in range(num_att_layers)
            ]
        )

        self.head = head
        self.head = nn.Conv2d(hidden_dim, 1, kernel_size=(1, 1))  # Reduces to mask
        # self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), )

    def encode_timestep(self, x, fstep=1):

        # Condition Time
        x = self.ct(x, fstep)

        ##CNN
        x = self.image_encoder(x)

        # Temporal Encoder
        _, state = self.temporal_enc(self.drop(x))
        return self.temporal_agg(state)

    def forward(self, imgs):
        """It takes a rank 5 tensor
        - imgs [bs, seq_len, channels, h, w]
        """
        # stack feature as images
        # TODO Could leave this out? Do this in dataloader?
        # - feats [bs, n_feats, seq_len]
        # if self.n_feats > 0:
        # feats = feat2image(feats, target_size=imgs.shape[-2:])
        # imgs = torch.cat([imgs, feats], dim=2)
        # Compute all timesteps, probably can be parallelized
        res = []
        for i in range(self.horizon):
            x_i = self.encode_timestep(imgs, i)
            out = self.head(x_i)
            res.append(out)
        res = torch.stack(res, dim=1).squeeze()
        return res

    def configure_optimizers(self):
        # DeepSpeedCPUAdam provides 5x to 7x speedup over torch.optim.adam(w)
        # optimizer = torch.optim.adam()
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = torch.squeeze(y)

        # if self.make_vis:
        #    if np.random.random() < 0.01:
        #        self.visualize(x, y, y_hat, batch_idx)
        # Generally only care about the center x crop, so the model can take into account the clouds in the area without
        # being penalized for that, but for now, just do general MSE loss, also only care about first 12 channels
        loss = F.mse_loss(y_hat, y)
        self.log("train/loss", loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = torch.squeeze(y)
        val_loss = F.mse_loss(y_hat, y)
        self.log("val/loss", val_loss, on_step=True, on_epoch=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = torch.squeeze(y)
        loss = F.mse_loss(y_hat, y)
        return loss


class TemporalEncoder(nn.Module):
    def __init__(self, in_channels, out_channels=384, ks=3, n_layers=1):
        super().__init__()
        self.rnn = ConvGRU(in_channels, out_channels, (ks, ks), n_layers, batch_first=True)

    def forward(self, x):
        x, h = self.rnn(x)
        return (x, h[-1])


def condition_time(x, i=0, size=(12, 16), seq_len=15):
    "create one hot encoded time image-layers, i in [1, seq_len]"
    assert i < seq_len
    times = (torch.eye(seq_len, dtype=x.dtype, device=x.device)[i]).unsqueeze(-1).unsqueeze(-1)
    ones = torch.ones(1, *size, dtype=x.dtype, device=x.device)
    return times * ones


class ConditionTime(nn.Module):
    "Condition Time on a stack of images, adds `horizon` channels to image"

    def __init__(self, horizon, ch_dim=2):
        super().__init__()
        self.horizon = horizon
        self.ch_dim = ch_dim

    def forward(self, x, fstep=0):
        "x stack of images, fsteps"
        bs, seq_len, ch, h, w = x.shape
        ct = condition_time(x, fstep, (h, w), seq_len=self.horizon).repeat(bs, seq_len, 1, 1, 1)
        x = torch.cat([x, ct], dim=self.ch_dim)
        assert x.shape[self.ch_dim] == (ch + self.horizon)  # check if it makes sense
        return x


def feat2image(x, target_size=(128, 128)):
    "This idea comes from MetNet"
    x = x.transpose(1, 2)
    return x.unsqueeze(-1).unsqueeze(-1) * x.new_ones(1, 1, 1, *target_size)
