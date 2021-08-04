import torch
from satflow.models.losses import get_loss, NowcastingLoss, GridCellLoss
import pytorch_lightning as pl
import torchvision
from functools import reduce
from typing import List
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
        forecast_steps: int = 18,
        input_channels: int = 1,
        output_shape: int = 256,
        gen_lr: float = 0.00005,
        disc_lr: float = 0.0002,
        visualize: bool = False,
        pretrained: bool = False,
        conv_type: str = "standard",
        num_samples: int = 6,
        grid_lambda: float = 20.0,
        beta1: float = 0.0,
        beta2: float = 0.999,
        latent_channels: int = 768,
        context_channels: int = 384,
    ):
        """
        Nowcasting GAN is an attempt to recreate DeepMind's Skillful Nowcasting GAN from https://arxiv.org/abs/2104.00954
        but slightly modified for multiple satellite channels
        Args:
            forecast_steps: Number of steps to predict in the future
            input_channels: Number of input channels per image
            visualize: Whether to visualize output during training
            gen_lr: Learning rate for the generator
            disc_lr: Learning rate for the discriminators, shared for both temporal and spatial discriminator
            conv_type: Type of 2d convolution to use, see satflow/models/utils.py for options
            beta1: Beta1 for Adam optimizer
            beta2: Beta2 for Adam optimizer
            num_samples: Number of samples of the latent space to sample for training/validation
            grid_lambda: Lambda for the grid regularization loss
            output_shape: Shape of the output predictions, generally should be same as the input shape
            latent_channels: Number of channels that the latent space should be reshaped to,
                input dimension into ConvGRU, also affects the number of channels for other linked inputs/outputs
            pretrained:
        """
        super(NowcastingGAN, self).__init__()
        self.gen_lr = gen_lr
        self.disc_lr = disc_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.discriminator_loss = NowcastingLoss()
        self.grid_regularizer = GridCellLoss()
        self.grid_lambda = grid_lambda
        self.num_samples = num_samples
        self.visualize = visualize
        self.latent_channels = latent_channels
        self.context_channels = context_channels
        self.input_channels = input_channels
        self.conditioning_stack = ContextConditioningStack(
            input_channels=input_channels,
            conv_type=conv_type,
            output_channels=self.context_channels,
        )
        self.latent_stack = LatentConditioningStack(
            shape=(8 * self.input_channels, output_shape // 32, output_shape // 32),
            output_channels=self.latent_channels,
        )
        self.sampler = NowcastingSampler(
            forecast_steps=forecast_steps,
            latent_channels=self.latent_channels,
            context_channels=self.context_channels,
        )
        self.generator = NowcastingGenerator(
            self.conditioning_stack, self.latent_stack, self.sampler
        )
        self.temporal_discriminator = NowcastingTemporalDiscriminator(
            input_channels=input_channels, crop_size=output_shape // 2, conv_type=conv_type
        )
        self.spatial_discriminator = NowcastingSpatialDiscriminator(
            input_channels=input_channels, num_timesteps=8, conv_type=conv_type
        )
        self.save_hyperparameters()

        self.global_iteration = 0

        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        torch.autograd.set_detect_anomaly(True)

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
        x = self.generator(x)
        return x

    def training_step(self, batch, batch_idx):
        images, future_images = batch
        self.global_iteration += 1
        g_opt, d_opt_s, d_opt_t = self.optimizers()
        ##########################
        # Optimize Discriminator #
        ##########################
        # Two discriminator steps per generator step
        # for _ in range(2):
        # TODO Make sure this is meant to be the mean predictions, or to run it 6 times and then take mean?
        mean_prediction = []
        for _ in range(self.num_samples):
            mean_prediction.append(self(images))
        mean_prediction = self.average_tensors(mean_prediction)
        # mean_prediction = self(images)
        # Get Spatial Loss
        # x should be the chosen 8 or so
        # idxs = torch.randint(low=0, high=mean_prediction.size()[1], size=(8,))
        # mean_s_prediction = mean_prediction[:,idxs,:,:,:]
        # future_s_images = future_images[:,idxs,:,:,:]
        spatial_real = self.spatial_discriminator(torch.cat((images, future_images), 1))
        spatial_fake = self.spatial_discriminator(torch.cat((images, mean_prediction), 1))
        spatial_loss = self.discriminator_loss(spatial_real, True) + self.discriminator_loss(
            spatial_fake, False
        )
        print(spatial_loss)
        d_opt_s.zero_grad()
        self.manual_backward(spatial_loss)
        d_opt_s.step()
        # Get Temporal Loss
        temporal_real = self.temporal_discriminator(torch.cat((images, future_images), 1))
        temporal_fake = self.temporal_discriminator(torch.cat((images, mean_prediction), 1))
        temporal_loss = self.discriminator_loss(temporal_real, True) + self.discriminator_loss(
            temporal_fake, False
        )

        # discriminator loss is the average of these
        d_loss = spatial_loss + temporal_loss
        print(temporal_loss)
        d_opt_t.zero_grad()
        self.manual_backward(temporal_loss)
        d_opt_t.step()

        ######################
        # Optimize Generator #
        ######################

        # Get Spatial Loss
        spatial_fake = self.spatial_discriminator(torch.cat((images, mean_prediction), 1))
        spatial_loss = self.discriminator_loss(spatial_fake, True)

        # Get Temporal Loss
        temporal_fake = self.temporal_discriminator(torch.cat((images, mean_prediction), 1))
        temporal_loss = self.discriminator_loss(temporal_fake, True)

        # Grid Cell Loss
        grid_loss = self.grid_regularizer(mean_prediction, future_images)

        g_loss = spatial_loss + temporal_loss - (self.grid_lambda * grid_loss)
        print(grid_loss.shape)
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
        if self.visualize:
            self.visualize_step(
                images, future_images, generated_images, self.global_iteration, step="train"
            )

    def average_tensors(self, x: List[torch.Tensor]):
        summed_tensor = x[0]
        for tensor in x[1:]:
            summed_tensor = torch.add(summed_tensor, tensor)
        summed_tensor = torch.div(summed_tensor, len(x))
        return summed_tensor

    def validation_step(self, batch, batch_idx):
        images, future_images = batch

        # First get the 6 samples to mean?
        # TODO Make sure this is what the paper actually means, or is it run it 6 times then average output?
        mean_prediction = []
        for _ in range(self.num_samples):
            mean_prediction.append(self(images))
        # It is a list of tensors of forecasts
        mean_prediction = self.average_tensors(mean_prediction)

        # Get Spatial Loss
        spatial_real = self.spatial_discriminator(torch.cat((images, future_images), 1))
        spatial_fake = self.spatial_discriminator(torch.cat((images, mean_prediction), 1))
        spatial_loss = self.discriminator_loss(spatial_real, True) + self.discriminator_loss(
            spatial_fake, False
        )

        # Get Temporal Loss
        temporal_real = self.temporal_discriminator(torch.cat((images, future_images), 1))
        temporal_fake = self.temporal_discriminator(torch.cat((images, mean_prediction), 1))
        temporal_loss = self.discriminator_loss(temporal_real, True) + self.discriminator_loss(
            temporal_fake, False
        )

        # Grid Cell Loss
        grid_loss = self.grid_regularizer(mean_prediction, future_images)

        # Generator Loss
        g_s = self.discriminator_loss(spatial_fake, True)
        g_t = self.discriminator_loss(temporal_fake, True)
        g_loss = g_s + g_t - (self.grid_lambda * grid_loss)

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
        b1 = self.beta1
        b2 = self.beta2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.gen_lr, betas=(b1, b2))
        opt_d_s = torch.optim.Adam(
            self.spatial_discriminator.parameters(), lr=self.disc_lr, betas=(b1, b2)
        )
        opt_d_t = torch.optim.Adam(
            self.temporal_discriminator.parameters(), lr=self.disc_lr, betas=(b1, b2)
        )

        return [opt_g, opt_d_s, opt_d_t], []

    def visualize_step(
        self, x: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor, batch_idx: int, step: str
    ) -> None:
        # the logger you used (in this case tensorboard)
        tensorboard = self.logger.experiment[0]
        # Timesteps per channel
        images = x[0].cpu().detach()
        future_images = y[0].cpu().detach()
        generated_images = y_hat[0].cpu().detach()
        for i, t in enumerate(images):  # Now would be (C, H, W)
            t = [torch.unsqueeze(img, dim=0) for img in t]
            image_grid = torchvision.utils.make_grid(t, nrow=self.input_channels)
            tensorboard.add_image(
                f"{step}/Input_Image_Stack_Frame_{i}", image_grid, global_step=batch_idx
            )
            t = [torch.unsqueeze(img, dim=0) for img in future_images[i]]
            image_grid = torchvision.utils.make_grid(t, nrow=self.input_channels)
            tensorboard.add_image(
                f"{step}/Target_Image_Frame_{i}", image_grid, global_step=batch_idx
            )
            t = [torch.unsqueeze(img, dim=0) for img in generated_images[i]]
            image_grid = torchvision.utils.make_grid(t, nrow=self.input_channels)
            tensorboard.add_image(
                f"{step}/Generated_Image_Frame_{i}", image_grid, global_step=batch_idx
            )


class NowcastingGenerator(torch.nn.Module):
    def __init__(
        self,
        conditioning_stack: torch.nn.Module,
        latent_stack: torch.nn.Module,
        sampler: torch.nn.Module,
    ):
        """
        Wraps the three parts of the generator for simpler calling
        Args:
            conditioning_stack:
            latent_stack:
            sampler:
        """
        super().__init__()
        self.conditioning_stack = conditioning_stack
        self.latent_stack = latent_stack
        self.sampler = sampler

    def forward(self, x):
        conditioning_states = self.conditioning_stack(x)
        latent_dim = self.latent_stack(x)
        x = self.sampler(conditioning_states, latent_dim)
        return x
