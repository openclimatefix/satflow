from typing import Any, Dict, Type
import torch.nn
import pytorch_lightning as pl
import torchvision

REGISTERED_MODELS = {}


def register_model(cls: Type[pl.LightningModule]):
    global REGISTERED_MODELS
    name = cls.__name__
    assert name not in REGISTERED_MODELS, f"exists class: {REGISTERED_MODELS}"
    REGISTERED_MODELS[name] = cls
    return cls


def get_model(name: str) -> Type[pl.LightningModule]:
    global REGISTERED_MODELS
    assert name in REGISTERED_MODELS, f"available class: {REGISTERED_MODELS}"
    return REGISTERED_MODELS[name]


def split_model_name(model_name):
    model_split = model_name.split(":", 1)
    if len(model_split) == 1:
        return "", model_split[0]
    else:
        source_name, model_name = model_split
        assert source_name in ("timm", "hf_hub")
        return source_name, model_name


def safe_model_name(model_name, remove_source=True):
    def make_safe(name):
        return "".join(c if c.isalnum() else "_" for c in name).rstrip("_")

    if remove_source:
        model_name = split_model_name(model_name)[-1]
    return make_safe(model_name)


def create_model(model_name, pretrained=False, checkpoint_path="", **kwargs):
    """Create a model

    Almost entirely taken from timm https://github.com/rwightman/pytorch-image-models

    Args:
        model_name (str): name of model to instantiate
        pretrained (bool): load pretrained ImageNet-1k weights if true
        checkpoint_path (str): path of checkpoint to load after model is initialized

    Keyword Args:
        drop_rate (float): dropout rate for training (default: 0.0)
        global_pool (str): global pool type (default: 'avg')
        input_channels (int): number of input channels (default: 12)
        forecast_steps (int): number of steps to forecast (default: 48)
        lr (float): learning rate (default: 0.001)
        **: other kwargs are model specific
    """
    source_name, model_name = split_model_name(model_name)

    # Parameters that aren't supported by all models or are intended to only override model defaults if set
    # should default to None in command line args/cfg. Remove them if they are present and not set so that
    # non-supporting models don't break and default args remain in effect.
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if model_name in REGISTERED_MODELS:
        model = get_model(model_name)(pretrained=pretrained, **kwargs)
    else:
        raise RuntimeError("Unknown model (%s)" % model_name)

    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)

    return model


class BaseModel(pl.LightningModule):
    def __init__(
        self,
        pretrained: bool = False,
        forecast_steps: int = 48,
        input_channels: int = 12,
        output_channels: int = 12,
        lr: float = 0.001,
        visualize: bool = False,
    ):
        super(BaseModel, self).__init__()
        self.forecast_steps = forecast_steps
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.lr = lr
        self.pretrained = pretrained
        self.visualize = visualize

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError

    def _train_or_validate_step(self, batch, batch_idx, is_training: bool = True):
        pass

    def training_step(self, batch, batch_idx):
        return self._train_or_validate_step(batch, batch_idx, is_training=True)

    def validation_step(self, batch, batch_idx):
        return self._train_or_validate_step(batch, batch_idx, is_training=False)

    def forward(self, x, **kwargs) -> Any:
        return self.model.forward(x, **kwargs)

    def visualize_step(
        self, x: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor, batch_idx: int, step: str
    ) -> None:
        # the logger you used (in this case tensorboard)
        tensorboard = self.logger.experiment
        # Timesteps per channel
        images = x[0].cpu().detach()  # T, C, H, W
        future_images = y[0].cpu().detach()
        generated_images = y_hat[0].cpu().detach()
        for i, t in enumerate(images):  # Now would be (C, H, W)
            t = [torch.unsqueeze(img, dim=0) for img in t]
            image_grid = torchvision.utils.make_grid(t, nrow=self.input_channels)
            tensorboard.log(f"{step}/Input_Frame_{i}", image_grid, global_step=batch_idx)
            t = [torch.unsqueeze(img, dim=0) for img in future_images[i]]
            image_grid = torchvision.utils.make_grid(t, nrow=self.output_channels)
            tensorboard.log(f"{step}/Target_Frame_{i}", image_grid, global_step=batch_idx)
            t = [torch.unsqueeze(img, dim=0) for img in generated_images[i]]
            image_grid = torchvision.utils.make_grid(t, nrow=self.output_channels)
            tensorboard.log(f"{step}/Predicted_Frame_{i}", image_grid, global_step=batch_idx)
