"""A fully convolutional network"""
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from nowcasting_utils.models.base import register_model
from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101
import numpy as np
from typing import Union
from nowcasting_utils.models.losses.FocalLoss import FocalLoss


@register_model
class FCN(pl.LightningModule):
    """A fully convolutional network"""
    def __init__(
        self,
        forecast_steps: int = 48,
        input_channels: int = 12,
        lr: float = 0.001,
        make_vis: bool = False,
        loss: Union[str, torch.nn.Module] = "mse",
        backbone: str = "resnet50",
        pretrained: bool = False,
    ):
        """
        Initialize the model

        Args:
            forecast_steps: number of timesteps to forecast. default is 48.
            input_channels: default is 12
            lr: learning rate. default is 0.001
            make_vis: whether to add a visualization step. default is False.
            loss: name of the loss function or torch.nn.Module. Default is "mse"
            backbone: the name of the backbone model. default is "resnet50".
            pretrained: Whether to use a model pre-trained on other data, Default is False
        """
        super(FCN, self).__init__()
        self.lr = lr
        assert loss in ["mse", "bce", "binary_crossentropy", "crossentropy", "focal"]
        if loss == "mse":
            self.criterion = F.mse_loss
        elif loss in ["bce", "binary_crossentropy", "crossentropy"]:
            self.criterion = F.nll_loss
        elif loss in ["focal"]:
            self.criterion = FocalLoss()
        else:
            raise ValueError(f"loss {loss} not recognized")
        self.make_vis = make_vis
        if backbone in ["r101", "resnet101"]:
            self.model = fcn_resnet101(pretrained=pretrained, num_classes=forecast_steps)
        else:
            self.model = fcn_resnet50(pretrained=pretrained, num_classes=forecast_steps)

        if input_channels != 3:
            self.model.backbone.conv1 = torch.nn.Conv2d(
                input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )
        self.save_hyperparameters()

    @classmethod
    def from_config(cls, config):
        """Initialize model from configuration values"""
        return DeeplabV3(
            forecast_steps=config.get("forecast_steps", 12),
            input_channels=config.get("in_channels", 12),
            hidden_dim=config.get("features", 64),
            num_layers=config.get("num_layers", 5),
            bilinear=config.get("bilinear", False),
            lr=config.get("lr", 0.001),
        )

    def forward(self, x):
        """A forward step of the model"""
        return self.model.forward(x)

    def configure_optimizers(self):
        """Get the optimizer with the initialized parameters"""
        # DeepSpeedCPUAdam provides 5x to 7x speedup over torch.optim.adam(w)
        # optimizer = torch.optim.adam()
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        """
        Perform a training step of the model

        Args:
            batch: tuple of (x, y)
            batch_idx: used to visualize the results of the training step

        Returns:
            The loss for the training step
        """
        x, y = batch
        y_hat = self(x)

        if self.make_vis:
            if np.random.random() < 0.01:
                self.visualize(x, y, y_hat, batch_idx)
        # Generally only care about the center x crop, so the model can take into account the clouds in the area without
        # being penalized for that, but for now, just do general MSE loss, also only care about first 12 channels
        loss = self.criterion(y_hat, y)
        self.log("train/loss", loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a validation step of the model

        Args:
            batch: tuple of (x, y)
            batch_idx: not implemented

        Returns:
            The loss for the validation step
        """
        x, y = batch
        y_hat = self(x)
        val_loss = self.criterion(y_hat, y)
        self.log("val/loss", val_loss, on_step=True, on_epoch=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        """
        Perform a testing step of the model

        Args:
            batch: tuple of (x, y)
            batch_idx: not implemented

        Returns:
            The loss for the testing step
        """
        x, y = batch
        y_hat = self(x, self.forecast_steps)
        loss = self.criterion(y_hat, y)
        return loss

    def visualize(self, x, y, y_hat, batch_idx):
        """
        Visualize the results of a step of the model

        Args:
            x: input data
            y: output
            y_hat: prediction
            batch_idx: (int) the global step to record for this batch
        """
        # the logger you used (in this case tensorboard)
        tensorboard = self.logger.experiment
        # Add all the different timesteps for a single prediction, 0.1% of the time
        in_image = (
            x[0].cpu().detach().numpy()
        )  # Input image stack, Unet takes everything in channels, so no time dimension
        for i, in_slice in enumerate(in_image):
            j = 0
            if i % self.input_channels == 0:  # First one
                j += 1
                tensorboard.add_image(
                    f"Input_Image_{j}_Channel_{i}", in_slice, global_step=batch_idx
                )  # Each Channel
        out_image = y_hat[0].cpu().detach().numpy()
        for i, out_slice in enumerate(out_image):
            tensorboard.add_image(
                f"Output_Image_{i}", out_slice, global_step=batch_idx
            )  # Each Channel
        out_image = y[0].cpu().detach().numpy()
        for i, out_slice in enumerate(out_image):
            tensorboard.add_image(
                f"Target_Image_{i}", out_slice, global_step=batch_idx
            )  # Each Channel
