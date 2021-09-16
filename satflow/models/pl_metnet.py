import torch
import torch.nn as nn
from typing import Any

from nowcasting_utils.models.base import register_model, BaseModel
from nowcasting_utils.models.loss import get_loss
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from metnet import MetNet

head_to_module = {"identity": nn.Identity()}


@register_model
class LitMetNet(BaseModel):
    def __init__(
        self,
        image_encoder: str = "downsampler",
        input_channels: int = 12,
        sat_channels: int = 12,
        input_size: int = 256,
        output_channels: int = 12,
        hidden_dim: int = 64,
        kernel_size: int = 3,
        num_layers: int = 1,
        num_att_layers: int = 1,
        head: str = "identity",
        forecast_steps: int = 48,
        temporal_dropout: float = 0.2,
        lr: float = 0.001,
        pretrained: bool = False,
        visualize: bool = False,
        loss: str = "mse",
    ):
        super(BaseModel, self).__init__()
        self.forecast_steps = forecast_steps
        self.input_channels = input_channels
        self.lr = lr
        self.pretrained = pretrained
        self.visualize = visualize
        self.output_channels = output_channels
        self.criterion = get_loss(
            loss, channel=output_channels, nonnegative_ssim=True, convert_range=True
        )
        self.model = MetNet(
            image_encoder=image_encoder,
            input_channels=input_channels,
            sat_channels=sat_channels,
            input_size=input_size,
            output_channels=output_channels,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=num_layers,
            num_att_layers=num_att_layers,
            head=head_to_module[head],
            forecast_steps=forecast_steps,
            temporal_dropout=temporal_dropout,
        )
        # TODO: Would be nice to have this automatically applied to all classes
        # that inherit from BaseModel
        self.save_hyperparameters()

    def forward(self, imgs, **kwargs) -> Any:
        return self.model(imgs)

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

    def _train_or_validate_step(self, batch, batch_idx, is_training: bool = True):
        x, y = batch
        x = x.float()
        y_hat = self(x)

        if self.visualize:
            if batch_idx == 1:
                self.visualize_step(x, y, y_hat, batch_idx, step="train" if is_training else "val")
        loss = self.criterion(y_hat, y)
        self.log(f"{'train' if is_training else 'val'}/loss", loss, prog_bar=True)
        frame_loss_dict = {}
        for f in range(self.forecast_steps):
            frame_loss = self.criterion(y_hat[:, f, :, :], y[:, f, :, :]).item()
            frame_loss_dict[f"{'train' if is_training else 'val'}/frame_{f}_loss"] = frame_loss
        self.log_dict(frame_loss_dict)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss
