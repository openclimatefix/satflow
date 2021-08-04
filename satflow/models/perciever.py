from perceiver_pytorch import PerceiverIO
import torch
from satflow.models.losses import get_loss, NowcastingLoss, GridCellLoss
import pytorch_lightning as pl
import torchvision
from functools import reduce
from typing import List
from satflow.models.base import register_model


class Perciever(pl.LightningModule):
    def __init__(
        self,
        input_channels: int = 12,
        forecast_steps: int = 48,
        depth: int = 6,
        num_latents: int = 256,
        cross_heads: int = 1,
        latent_heads: int = 8,
        cross_dim_heads: int = 8,
        latent_dim: int = 512,
        weight_tie_layers: bool = False,
        self_per_cross_attention: int = 2,
        dim: int = 32,
        logits_dim: int = 100,
        queries_dim: int = 32,
        latent_dim_heads: int = 64,
    ):
        super().__init__()
        self.model = PerceiverIO(
            dim=dim,  # dimension of sequence to be encoded
            queries_dim=queries_dim,  # dimension of decoder queries
            logits_dim=logits_dim,  # dimension of final logits
            depth=depth,  # depth of net
            num_latents=num_latents,  # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim=latent_dim,  # latent dimension
            cross_heads=cross_heads,  # number of heads for cross attention. paper said 1
            latent_heads=latent_heads,  # number of heads for latent self attention, 8
            cross_dim_head=cross_dim_heads,  # number of dimensions per cross attention head
            latent_dim_head=latent_dim_heads,  # number of dimensions per latent self attention head
            weight_tie_layers=weight_tie_layers,  # whether to weight tie layers (optional, as indicated in the diagram)
            self_per_cross_attn=self_per_cross_attention,  # number of self attention blocks per cross attention
        )

    def training_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def forward(self, x):
        pass

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
