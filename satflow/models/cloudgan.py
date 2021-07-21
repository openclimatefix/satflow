import pytorch_lightning as pl
import torch
from torch.optim import lr_scheduler
import torchvision
from collections import OrderedDict
from satflow.models import R2U_Net, ConvLSTM
from satflow.models.gan import GANLoss, define_generator, define_discriminator
from satflow.models.layers import ConditionTime
import numpy as np


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
        channels_per_timestep: int = 12,
        condition_time: bool = False,
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
        self.output_channels = forecast_steps * 12
        self.channels_per_timestep = channels_per_timestep
        self.condition_time = condition_time
        if condition_time:
            self.ct = ConditionTime(forecast_steps)
        # define networks (both generator and discriminator)
        if generator_model == "runet":
            generator_model = R2U_Net(input_channels, self.output_channels, t=3)
        elif generator_model == "convlstm":
            generator_model = ConvLSTM(
                input_channels, hidden_dim=num_filters, out_channels=self.output_channels
            )
        self.generator = define_generator(
            input_channels, self.output_channels, num_filters, generator_model, norm, use_dropout
        )
        if generator_model == "convlstm":
            # Timestep x C x H x W inputs/outputs, need to flatten for discriminator
            # TODO Add Discriminator that can use timesteps
            self.flatten_generator = True
        else:
            self.flatten_generator = False

        self.discriminator = define_discriminator(
            input_channels + self.output_channels,
            num_filters,
            discriminator_model,
            discriminator_layers,
            norm,
        )

        # define loss functions
        self.criterionGAN = GANLoss(loss)
        self.criterionL1 = torch.nn.L1Loss()
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx, optimizer_idx):
        images, future_images, future_masks = batch
        # train generator
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
            fake_loss = self.criterionGAN(self.discriminator(fake), True)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict({"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            self.log_dict({"train/d_loss": d_loss})
            return output

    def validation_step(self, batch, batch_idx):
        images, future_images, future_masks = batch
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

    def forward(self, x, **kwargs):
        if self.condition_time:
            res = []
            for i in range(self.forecast_steps):
                x_i = self.ct.forward(x, i)
                out = self.generator.forward(x_i, **kwargs)
                res.append(out)
            res = torch.stack(res, dim=1).squeeze()
            return res
        else:
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
            d_scheduler = lr_scheduler.CosineAnnealingLR(opt_g, T_max=self.lr_epochs, eta_min=0)
        else:
            return NotImplementedError("learning rate policy is not implemented")

        return [opt_g, opt_d], [g_scheduler, d_scheduler]

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
