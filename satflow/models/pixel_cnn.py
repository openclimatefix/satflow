"""A PixelCNN model: https://arxiv.org/pdf/1905.09272.pdf"""
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from nowcasting_utils.models.base import register_model
from pl_bolts.models.vision import PixelCNN as Pixcnn


@register_model
class PixelCNN(pl.LightningModule):
    """A Pixel CNN model: https://arxiv.org/pdf/1905.09272.pdf"""
    def __init__(
        self,
        future_timesteps: int,
        input_channels: int = 3,
        num_layers: int = 5,
        num_hidden: int = 64,
        pretrained: bool = False,
        lr: float = 0.001,
    ):
        """
        Initialize the model

        Args:
            future_timesteps: not implemented
            input_channels: default is 3
            num_layers: default is 5
            num_hidden: default is 64
            pretrained: not implemented. default is False
            lr: learning rate. default is 0.001
        """
        super(PixelCNN, self).__init__()
        self.lr = lr
        self.model = Pixcnn(
            input_channels=input_channels, hidden_channels=num_hidden, num_blocks=num_layers
        )

    @classmethod
    def from_config(cls, config):
        """Initialize PixelCNN model from configuration values"""
        return PixelCNN(
            future_timesteps=config.get("future_timesteps", 12),
            input_channels=config.get("in_channels", 12),
            features_start=config.get("features", 64),
            num_layers=config.get("num_layers", 5),
            bilinear=config.get("bilinear", False),
            lr=config.get("lr", 0.001),
        )

    def forward(self, x):
        """A forward step of the model"""
        self.model.forward(x)

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
            batch_idx: not implemented

        Returns:
            The loss for the training step
        """
        x, y = batch
        y_hat = self(x)
        # Generally only care about the center x crop, so the model can take into account the clouds in the area without
        # being penalized for that, but for now, just do general MSE loss, also only care about first 12 channels
        loss = F.mse_loss(y_hat, y)
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
        val_loss = F.mse_loss(y_hat, y)
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
        loss = F.mse_loss(y_hat, y)
        return loss
