import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from typing import Union
from collections import OrderedDict
from satflow.models.losses import get_loss, NowcastingLoss, GridCellLoss
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
        output_shape: int = 256,
        gen_lr: float = 0.00005,
        disc_lr: float = 0.0002,
        make_vis: bool = False,
        loss: Union[str, torch.nn.Module] = "mse",
        pretrained: bool = False,
        conv_type: str = "standard",
        num_samples: int = 6,
        grid_lambda: float = 20.0,
        beta1: float = 0.0,
        beta2: float = 0.999,
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
        self.gen_lr = gen_lr
        self.disc_lr = disc_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.criterion = get_loss(loss)
        self.discriminator_loss = NowcastingLoss()
        self.grid_regularizer = GridCellLoss()
        self.grid_lambda = grid_lambda
        self.num_samples = num_samples
        self.make_vis = make_vis
        self.input_channels = input_channels
        self.conditioning_stack = ContextConditioningStack(
            input_channels=input_channels, conv_type=conv_type
        )
        self.latent_stack = LatentConditioningStack(
            shape=(output_shape // 32, output_shape // 32, 8 * self.input_channels)
        )
        self.sampler = NowcastingSampler(forecast_steps=forecast_steps, input_channels=768)
        self.temporal_discriminator = NowcastingTemporalDiscriminator(
            input_channels=input_channels, crop_size=output_shape // 4
        )
        self.spatial_discriminator = NowcastingSpatialDiscriminator(
            input_channels=input_channels, num_timesteps=8
        )
        self.save_hyperparameters()

        self.global_iteration = 0

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

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
        conditioning_states = self.conditioning_stack(x)
        latent_dim = self.latent_stack()
        x = self.sampler(conditioning_states, latent_dim)
        return x

    def training_step(self, batch, batch_idx):
        images, future_images = batch
        self.global_iteration += 1
        g_opt, d_opt_s, d_opt_t = self.optimizers()
        ##########################
        # Optimize Discriminator #
        ##########################
        # Two discriminator steps per generator step
        # Measure discriminator's ability to classify real from generated samples

        # First get the 6 samples to mean?
        # TODO Make sure this is what the paper actually means, or is it run it 6 times then average output?
        mean_prediction = []
        for _ in range(self.num_samples):
            mean_prediction.append(self(images))
        mean_prediction = torch.mean(torch.cat(mean_prediction, dim=0), dim=0)

        # Get Spatial Loss
        spatial_real = self.spatial_discriminator(torch.cat((images, future_images), 1))
        spatial_fake = self.spatial_discriminator(torch.cat((images, mean_prediction), 1))
        spatial_loss = self.discriminator_loss(spatial_real, spatial_fake)

        # Get Temporal Loss
        temporal_real = self.temporal_discriminator(torch.cat((images, future_images), 1))
        temporal_fake = self.temporal_discriminator(torch.cat((images, mean_prediction), 1))
        temporal_loss = self.discriminator_loss(temporal_real, temporal_fake)

        # discriminator loss is the average of these
        d_loss = spatial_loss + temporal_loss
        d_opt_t.zero_grad()
        self.manual_backward(temporal_loss)
        d_opt_t.step()
        d_opt_s.zero_grad()
        self.manual_backward(spatial_loss)
        d_opt_s.step()

        ######################
        # Optimize Generator #
        ######################

        # Get Spatial Loss
        spatial_real = self.spatial_discriminator(torch.cat((images, future_images), 1))
        spatial_fake = self.spatial_discriminator(torch.cat((images, mean_prediction), 1))
        spatial_loss = self.discriminator_loss(spatial_real, spatial_fake)

        # Get Temporal Loss
        temporal_real = self.temporal_discriminator(torch.cat((images, future_images), 1))
        temporal_fake = self.temporal_discriminator(torch.cat((images, mean_prediction), 1))
        temporal_loss = self.discriminator_loss(temporal_real, temporal_fake)

        # Grid Cell Loss
        grid_loss = self.grid_regularizer(mean_prediction, future_images)

        g_loss = spatial_loss + temporal_loss - (self.grid_lambda * grid_loss)

        g_opt.zero_grad()
        self.manual_backward(g_loss)
        g_opt.step()

        self.log_dict(
            {
                "train/d_loss": d_loss,
                "train/temporal_loss": temporal_loss,
                "train/spatial_loss": spatial_loss,
                "train/g_loss": g_loss,
                "train/grid_loss": grid_loss,
            },
            prog_bar=True,
        )

        # generate images
        generated_images = self(images)
        # log sampled images
        self.visualize_step(
            images, future_images, generated_images, self.global_iteration, step="train"
        )

    def validation_step(self, batch, batch_idx):
        images, future_images = batch

        # First get the 6 samples to mean?
        # TODO Make sure this is what the paper actually means, or is it run it 6 times then average output?
        mean_prediction = []
        for _ in range(self.num_samples):
            mean_prediction.append(self(images))
        mean_prediction = torch.mean(torch.cat(mean_prediction, dim=0), dim=0)

        # Get Spatial Loss
        spatial_real = self.spatial_discriminator(torch.cat((images, future_images), 1))
        spatial_fake = self.spatial_discriminator(torch.cat((images, mean_prediction), 1))
        spatial_loss = self.discriminator_loss(spatial_real, spatial_fake)

        # Get Temporal Loss
        temporal_real = self.temporal_discriminator(torch.cat((images, future_images), 1))
        temporal_fake = self.temporal_discriminator(torch.cat((images, mean_prediction), 1))
        temporal_loss = self.discriminator_loss(temporal_real, temporal_fake)

        # Grid Cell Loss
        grid_loss = self.grid_regularizer(mean_prediction, future_images)

        g_loss = spatial_loss + temporal_loss - (self.grid_lambda * grid_loss)

        self.log_dict(
            {
                "val/d_loss": temporal_loss + spatial_loss,
                "val/temporal_loss": temporal_loss,
                "val/spatial_loss": spatial_loss,
                "val/g_loss": g_loss,
                "val/grid_loss": grid_loss,
            },
            prog_bar=True,
        )

    def configure_optimizers(self):
        b1 = self.b1
        b2 = self.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.gen_lr, betas=(b1, b2))
        opt_d_s = torch.optim.Adam(
            self.spatial_discriminator.parameters(), lr=self.disc_lr, betas=(b1, b2)
        )
        opt_d_t = torch.optim.Adam(
            self.temporal_discriminator.parameters(), lr=self.disc_lr, betas=(b1, b2)
        )

        return [opt_g, opt_d_s, opt_d_t], []
