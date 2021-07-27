import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from satflow.models.base import register_model
from satflow.models.utils import get_conv_layer
from satflow.models.losses import get_loss
from satflow.models.layers import ConvGRU, TimeDistributed, ConditionTime
from axial_attention import AxialAttention
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import numpy as np
import torchvision
import antialiased_cnns


class DownSampler(nn.Module):
    def __init__(self, in_channels, conv_type: str = "standard"):
        super().__init__()
        conv2d = get_conv_layer(conv_type=conv_type)
        if conv_type == "antialiased":
            antialiased = True
        else:
            antialiased = False

        self.module = nn.Sequential(
            conv2d(in_channels, 160, 3, padding=1),
            nn.MaxPool2d((2, 2), stride=1 if antialiased else 2),
            antialiased_cnns.BlurPool(160, stride=2) if antialiased else nn.Identity(),
            nn.BatchNorm2d(160),
            conv2d(160, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            conv2d(256, 256, 3, padding=1),
            nn.MaxPool2d((2, 2), stride=1 if antialiased else 2),
            antialiased_cnns.BlurPool(256, stride=2) if antialiased else nn.Identity(),
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
        visualize: bool = False,
        loss: str = "mse",
        output_channels: int = 12,
    ):
        super().__init__()

        self.forecast_steps = forecast_steps
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.criterion = get_loss(loss)
        self.lr = lr
        self.visualize = visualize
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
        self.head = nn.Conv2d(hidden_dim, output_channels, kernel_size=(1, 1))  # Reduces to mask
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
        for i in range(self.forecast_steps):
            x_i = self.encode_timestep(imgs, i)
            out = self.head(x_i)
            res.append(out)
        res = torch.stack(res, dim=1).squeeze()
        return res

    def configure_optimizers(self):
        # DeepSpeedCPUAdam provides 5x to 7x speedup over torch.optim.adam(w)
        # optimizer = torch.optim.adam()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=100)
        lr_dict = {
            # REQUIRED: The scheduler instance
            "scheduler": scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "step",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": None,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_dict}

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y_hat = self(x)
        # y = torch.squeeze(y)

        if self.visualize:
            if np.random.random() < 0.01:
                self.visualize_step(x, y, y_hat, batch_idx)
        # Generally only care about the center x crop, so the model can take into account the clouds in the area without
        # being penalized for that, but for now, just do general MSE loss, also only care about first 12 channels
        loss = self.criterion(y_hat, y)
        self.log("train/loss", loss)
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
        y_hat = self(x)
        y = torch.squeeze(y)
        loss = F.mse_loss(y_hat, y)
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
        else:
            images = x[0].cpu().detach()
            images = [torch.unsqueeze(img, dim=0) for img in images]
            image_grid = torchvision.utils.make_grid(images, nrow=self.input_channels)
            tensorboard.add_image(f"{step}/Input_Image_Stack", image_grid, global_step=batch_idx)


class TemporalEncoder(nn.Module):
    def __init__(self, in_channels, out_channels=384, ks=3, n_layers=1):
        super().__init__()
        self.rnn = ConvGRU(in_channels, out_channels, (ks, ks), n_layers, batch_first=True)

    def forward(self, x):
        x, h = self.rnn(x)
        return (x, h[-1])


def feat2image(x, target_size=(128, 128)):
    "This idea comes from MetNet"
    x = x.transpose(1, 2)
    return x.unsqueeze(-1).unsqueeze(-1) * x.new_ones(1, 1, 1, *target_size)
