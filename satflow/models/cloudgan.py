from collections import OrderedDict

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from nowcasting_utils.models.loss import get_loss
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim import lr_scheduler

from satflow.models import ConvLSTM, R2U_Net
from satflow.models.gan import GANLoss, define_discriminator, define_generator
from satflow.models.layers import ConditionTime


class CloudGAN(pl.LightningModule):
    def __init__(
        self,
        forecast_steps: int = 48,
        input_channels: int = 12,
        lr: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.999,
        num_filters: int = 64,
        generator_model: str = "runet",
        norm: str = "batch",
        use_dropout: bool = False,
        discriminator_model: str = "enhanced",
        discriminator_layers: int = 0,
        loss: str = "vanilla",
        scheduler: str = "plateau",
        lr_epochs: int = 10,
        lambda_l1: float = 100.0,
        l1_loss: str = "l1",
        channels_per_timestep: int = 12,
        condition_time: bool = False,
        pretrained: bool = False,
    ):
        """
        Creates CloudGAN, based off of https://www.climatechange.ai/papers/icml2021/54
        Changes include allowing outputs for all timesteps, optionally conditioning on time
        for single timestep output

        Args:
            forecast_steps: Number of timesteps to forecast
            input_channels: Number of input channels
            lr: Learning Rate
            beta1: optimizer beta1
            beta2: optimizer beta2 value
            num_filters: Number of filters in generator
            generator_model: Generator name
            norm: Norm type
            use_dropout: Whether to use dropout
            discriminator_model: model for discriminator, one of options in define_discriminator
            discriminator_layers: Number of layers in discriminator, only for NLayerDiscriminator
            loss: Loss function, described in GANLoss
            scheduler: LR scheduler name
            lr_epochs: Epochs for LR scheduler
            lambda_l1: Lambda for L1 loss, from slides recommended between 5-200
            l1_loss: Loss to use for the L1 in the slides, default is L1, also SSIM is available
            channels_per_timestep: Channels per input timestep
            condition_time: Whether to condition on a future timestep, similar to MetNet
        """
        super().__init__()
        self.lr = lr
        self.b1 = beta1
        self.b2 = beta2
        self.loss = loss
        self.lambda_l1 = lambda_l1
        self.lr_epochs = lr_epochs
        self.lr_method = scheduler
        self.forecast_steps = forecast_steps
        self.input_channels = input_channels
        self.output_channels = forecast_steps * channels_per_timestep
        self.channels_per_timestep = channels_per_timestep
        self.condition_time = condition_time
        if condition_time:
            self.ct = ConditionTime(forecast_steps)
        # define networks (both generator and discriminator)
        gen_input_channels = (
            input_channels  # + forecast_steps if condition_time else input_channels
        )
        self.recurrent = (
            False  # Does the generator generate all timesteps at once, or a single one at a time?
        )
        if generator_model == "runet":
            generator_model = R2U_Net(gen_input_channels, self.output_channels, t=3)
        elif generator_model == "convlstm":
            self.recurrent = True  # ConvLSTM makes a list of output timesteps
            generator_model = ConvLSTM(
                gen_input_channels, hidden_dim=num_filters, out_channels=self.channels_per_timestep
            )
        self.generator = define_generator(
            gen_input_channels,
            self.output_channels,
            num_filters,
            generator_model,
            norm,
            use_dropout,
        )
        if generator_model == "convlstm":
            # Timestep x C x H x W inputs/outputs, need to flatten for discriminator
            # TODO Add Discriminator that can use timesteps
            self.flatten_generator = True
        else:
            self.flatten_generator = False

        self.discriminator = define_discriminator(
            self.channels_per_timestep if condition_time else self.output_channels,
            num_filters,
            discriminator_model,
            discriminator_layers,
            norm,
        )

        # define loss functions
        self.criterionGAN = GANLoss(loss)
        self.criterionL1 = get_loss(l1_loss, channels=self.channels_per_timestep)
        self.save_hyperparameters()

    def train_per_timestep(
        self, images: torch.Tensor, future_images: torch.Tensor, optimizer_idx: int, batch_idx: int
    ):
        """
        For training with conditioning on time, so when the model is giving a single output

        This goes through every timestep in forecast_steps and runs the training
        Args:
            images: (Batch, Timestep, Channels, Width, Height)
            future_images: (Batch, Timestep, Channels, Width, Height)
            optimizer_idx: int, the optiimizer to use

        Returns:

        """

        if optimizer_idx == 0:
            # generate images
            total_loss = 0
            vis_step = True if np.random.random() < 0.01 else False
            generated_images = self(
                images, forecast_steps=self.forecast_steps
            )  # (Batch, Channel, Width, Height)
            for i in range(self.forecast_steps):
                # x = self.ct.forward(images, i)  # Condition on future timestep
                # fake = self(x, forecast_steps=i + 1)  # (Batch, Channel, Width, Height)
                fake = generated_images[:, :, i, :, :]  # Only take the one at the end
                if vis_step:
                    self.visualize_step(
                        images, future_images[:, i, :, :], fake, batch_idx, step=f"train_frame_{i}"
                    )
                # adversarial loss is binary cross-entropy
                gan_loss = self.criterionGAN(self.discriminator(fake), True)
                # Only L1 loss on the given timestep
                l1_loss = self.criterionL1(fake, future_images[:, i, :, :]) * self.lambda_l1
                self.log(f"train/frame_{i}_l1_loss", l1_loss)
                g_loss = gan_loss + l1_loss
                total_loss += g_loss
            g_loss = total_loss / self.forecast_steps  # Get the mean loss over all timesteps
            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict({"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            self.log_dict({"train/g_loss": g_loss})
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples
            # generate images
            total_loss = 0
            generated_images = self(
                images, forecast_steps=self.forecast_steps
            )  # (Batch, Channel, Width, Height)
            for i in range(self.forecast_steps):
                # x = self.ct.forward(images, i)  # Condition on future timestep
                # fake = self(x, forecast_steps=i + 1)  # (Batch, Channel, Width, Height)
                fake = generated_images[:, :, i, :, :]  # Only take the one at the end
                real_loss = self.criterionGAN(self.discriminator(future_images[:, i, :, :]), True)
                # adversarial loss is binary cross-entropy
                fake_loss = self.criterionGAN(self.discriminator(fake), False)
                # Only L1 loss on the given timestep
                # discriminator loss is the average of these
                d_loss = (real_loss + fake_loss) / 2
                self.log(f"train/frame_{i}_d_loss", d_loss)
                total_loss += d_loss
            d_loss = total_loss / self.forecast_steps  # Average of the per-timestep loss
            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict({"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            self.log_dict({"train/d_loss": d_loss})
            return output

    def train_all_timestep(
        self, images: torch.Tensor, future_images: torch.Tensor, optimizer_idx: int, batch_idx: int
    ):
        """
        Train on all timesteps, instead of single timestep at a time. No conditioning on future timestep
        Args:
            images:
            future_images:
            optimizer_idx:
            batch_idx:

        Returns:

        """
        if optimizer_idx == 0:
            # generate images
            generated_images = self(images)
            fake = torch.cat((images, generated_images), 1)
            # log sampled images
            if np.random.random() < 0.01:
                self.visualize_step(
                    images, future_images, generated_images, batch_idx, step="train"
                )

            # adversarial loss is binary cross-entropy
            gan_loss = self.criterionGAN(self.discriminator(fake), True)
            l1_loss = self.criterionL1(generated_images, future_images) * self.lambda_l1
            g_loss = gan_loss + l1_loss
            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict({"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            self.log_dict({"train/g_loss": g_loss})
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            real = torch.cat((images, future_images), 1)
            real_loss = self.criterionGAN(self.discriminator(real), True)

            # how well can it label as fake?
            gen_output = self(images)
            fake = torch.cat((images, gen_output), 1)
            fake_loss = self.criterionGAN(self.discriminator(fake), False)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict({"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            self.log_dict({"train/d_loss": d_loss})
            return output

    def training_step(self, batch, batch_idx, optimizer_idx):
        images, future_images = batch
        if self.condition_time:
            return self.train_per_timestep(images, future_images, optimizer_idx, batch_idx)
        else:
            return self.train_all_timestep(images, future_images, optimizer_idx, batch_idx)

    def val_all_timestep(self, images, future_images, batch_idx):
        # generate images
        generated_images = self(images)
        fake = torch.cat((images, generated_images), 1)
        # log sampled images
        if np.random.random() < 0.01:
            self.visualize_step(images, future_images, generated_images, batch_idx, step="val")
        # adversarial loss is binary cross-entropy
        gan_loss = self.criterionGAN(self.discriminator(fake), True)
        l1_loss = self.criterionL1(generated_images, future_images) * self.lambda_l1
        g_loss = gan_loss + l1_loss
        # how well can it label as real?
        real = torch.cat((images, future_images), 1)
        real_loss = self.criterionGAN(self.discriminator(real), True)

        # how well can it label as fake?
        fake_loss = self.criterionGAN(self.discriminator(fake), True)

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2
        tqdm_dict = {"d_loss": d_loss}
        output = OrderedDict(
            {
                "val/discriminator_loss": d_loss,
                "val/generator_loss": g_loss,
                "progress_bar": tqdm_dict,
                "log": tqdm_dict,
            }
        )
        self.log_dict({"val/d_loss": d_loss, "val/g_loss": g_loss, "val/loss": d_loss + g_loss})
        return output

    def val_per_timestep(self, images, future_images, batch_idx):
        total_g_loss = 0
        total_d_loss = 0
        vis_step = True if np.random.random() < 0.01 else False
        generated_images = self(
            images, forecast_steps=self.forecast_steps
        )  # (Batch, Channel, Width, Height)
        for i in range(self.forecast_steps):
            # x = self.ct.forward(images, i)  # Condition on future timestep
            fake = generated_images[:, :, i, :, :]  # Only take the one at the end
            if vis_step:
                self.visualize_step(
                    images, future_images[:, i, :, :], fake, batch_idx, step=f"val_frame_{i}"
                )
            # adversarial loss is binary cross-entropy
            gan_loss = self.criterionGAN(self.discriminator(fake), True)
            # Only L1 loss on the given timestep
            l1_loss = self.criterionL1(fake, future_images[:, i, :, :]) * self.lambda_l1
            real_loss = self.criterionGAN(self.discriminator(future_images[:, i, :, :]), True)
            # adversarial loss is binary cross-entropy
            fake_loss = self.criterionGAN(self.discriminator(fake), False)
            # Only L1 loss on the given timestep
            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            self.log(f"val/frame_{i}_d_loss", d_loss)
            total_d_loss += d_loss
            self.log(f"val/frame_{i}_l1_loss", l1_loss)
            g_loss = gan_loss + l1_loss
            total_g_loss += g_loss
        g_loss = total_g_loss / self.forecast_steps
        d_loss = total_d_loss / self.forecast_steps
        loss = g_loss + d_loss
        tqdm_dict = {"loss": loss}
        output = OrderedDict(
            {
                "val/discriminator_loss": d_loss,
                "val/generator_loss": g_loss,
                "progress_bar": tqdm_dict,
                "log": tqdm_dict,
            }
        )
        self.log_dict({"val/d_loss": d_loss, "val/g_loss": g_loss, "val/loss": d_loss + g_loss})
        return output

    def validation_step(self, batch, batch_idx):
        images, future_images = batch
        if self.condition_time:
            return self.val_per_timestep(images, future_images, batch_idx)
        else:
            return self.val_all_timestep(images, future_images, batch_idx)

    def forward(self, x, **kwargs):
        return self.generator.forward(x, **kwargs)

    def configure_optimizers(self):
        lr = self.lr
        b1 = self.b1
        b2 = self.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        if self.lr_method == "plateau":
            g_scheduler = lr_scheduler.ReduceLROnPlateau(
                opt_g, mode="min", factor=0.2, threshold=0.01, patience=10
            )
            d_scheduler = lr_scheduler.ReduceLROnPlateau(
                opt_d, mode="min", factor=0.2, threshold=0.01, patience=10
            )
        elif self.lr_method == "cosine":
            g_scheduler = lr_scheduler.CosineAnnealingLR(opt_g, T_max=self.lr_epochs, eta_min=0)
            d_scheduler = lr_scheduler.CosineAnnealingLR(opt_d, T_max=self.lr_epochs, eta_min=0)
        elif self.lr_method == "warmup":
            g_scheduler = LinearWarmupCosineAnnealingLR(
                opt_g, warmup_epochs=self.lr_epochs, max_epochs=100
            )
            d_scheduler = LinearWarmupCosineAnnealingLR(
                opt_d, warmup_epochs=self.lr_epochs, max_epochs=100
            )
        else:
            return NotImplementedError("learning rate policy is not implemented")

        return [opt_g, opt_d], [g_scheduler, d_scheduler]

    def visualize_step(
        self, x: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor, batch_idx: int, step: str
    ):
        # the logger you used (in this case tensorboard)
        tensorboard = self.logger.experiment[0]
        # Image input is either (B, C, H, W) or (B, T, C, H, W)
        if len(x.shape) == 5:
            # Timesteps per channel
            images = x[0].cpu().detach()
            for i, t in enumerate(images):  # Now would be (C, H, W)
                t = [torch.unsqueeze(img, dim=0) for img in t]
                image_grid = torchvision.utils.make_grid(t, nrow=self.channels_per_timestep)
                tensorboard.add_image(
                    f"{step}/Input_Image_Stack_Frame_{i}", image_grid, global_step=batch_idx
                )
        else:
            images = x[0].cpu().detach()
            images = [torch.unsqueeze(img, dim=0) for img in images]
            image_grid = torchvision.utils.make_grid(images, nrow=self.channels_per_timestep)
            tensorboard.add_image(f"{step}/Input_Image_Stack", image_grid, global_step=batch_idx)
        # In all cases, the output target and image are in (B, C, H, W) format
        images = y[0].cpu().detach()
        images = [torch.unsqueeze(img, dim=0) for img in images]
        image_grid = torchvision.utils.make_grid(images, nrow=12)
        tensorboard.add_image(f"{step}/Target_Image_Stack", image_grid, global_step=batch_idx)
        images = y_hat[0].cpu().detach()
        images = [torch.unsqueeze(img, dim=0) for img in images]
        image_grid = torchvision.utils.make_grid(images, nrow=12)
        tensorboard.add_image(f"{step}/Generated_Image_Stack", image_grid, global_step=batch_idx)
