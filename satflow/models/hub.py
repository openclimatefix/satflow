"""
Originally Taken from https://github.com/rwightman/pytorch-image-models/blob/acd6c687fd1c0507128f0ce091829b233c8560b9/timm/models/hub.py
"""

import json
import logging
import os
from functools import partial
from typing import Optional, Union

import pytorch_lightning
import torch

try:
    from torch.hub import get_dir
except ImportError:
    from torch.hub import _get_torch_home as get_dir

from satflow import __version__

try:
    from huggingface_hub import cached_download, hf_hub_url

    cached_download = partial(cached_download, library_name="satflow", library_version=__version__)
except ImportError:
    hf_hub_url = None
    cached_download = None

from huggingface_hub import CONFIG_NAME, PYTORCH_WEIGHTS_NAME, ModelHubMixin, hf_hub_download

MODEL_CARD_MARKDOWN = """---
license: mit
tags:
- satflow
- forecasting
- timeseries
- remote-sensing
---

# {model_name}

## Model description

[More information needed]

## Intended uses & limitations

[More information needed]

## How to use

[More information needed]

## Limitations and bias

[More information needed]

## Training data

[More information needed]

## Training procedure

[More information needed]

## Evaluation results

[More information needed]

"""

_logger = logging.getLogger(__name__)


def get_cache_dir(child_dir=""):
    """
    Returns the location of the directory where models are cached (and creates it if necessary).
    """
    hub_dir = get_dir()
    child_dir = () if not child_dir else (child_dir,)
    model_dir = os.path.join(hub_dir, "checkpoints", *child_dir)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def has_hf_hub(necessary=False):
    if hf_hub_url is None and necessary:
        # if no HF Hub module installed and it is necessary to continue, raise error
        raise RuntimeError(
            "Hugging Face hub model specified but package not installed. Run `pip install huggingface_hub`."
        )
    return hf_hub_url is not None


def hf_split(hf_id):
    rev_split = hf_id.split("@")
    assert (
        0 < len(rev_split) <= 2
    ), "hf_hub id should only contain one @ character to identify revision."
    hf_model_id = rev_split[0]
    hf_revision = rev_split[-1] if len(rev_split) > 1 else None
    return hf_model_id, hf_revision


def load_cfg_from_json(json_file: Union[str, os.PathLike]):
    with open(json_file, "r", encoding="utf-8") as reader:
        text = reader.read()
    return json.loads(text)


def _download_from_hf(model_id: str, filename: str):
    hf_model_id, hf_revision = hf_split(model_id)
    url = hf_hub_url(hf_model_id, filename, revision=hf_revision)
    return cached_download(url, cache_dir=get_cache_dir("hf"))


def load_model_config_from_hf(model_id: str):
    assert has_hf_hub(True)
    cached_file = _download_from_hf(model_id, "config.json")
    default_cfg = load_cfg_from_json(cached_file)
    default_cfg[
        "hf_hub"
    ] = model_id  # insert hf_hub id for pretrained weight load during model creation
    model_name = default_cfg.get("architecture")
    return default_cfg, model_name


def load_state_dict_from_hf(model_id: str):
    assert has_hf_hub(True)
    cached_file = _download_from_hf(model_id, "pytorch_model.pth")
    state_dict = torch.load(cached_file, map_location="cpu")
    return state_dict


def cache_file_from_hf(model_id: str):
    assert has_hf_hub(True)
    cached_file = _download_from_hf(model_id, "pytorch_model.pth")
    return cached_file


def load_pretrained(
    model,
    default_cfg: Optional[dict] = None,
    in_chans: int = 12,
    strict: bool = True,
) -> Union[torch.nn.Module, pytorch_lightning.LightningModule]:
    """Load pretrained checkpoint

    Taken from https://github.com/rwightman/pytorch-image-models/blob/acd6c687fd1c0507128f0ce091829b233c8560b9/timm/models/helpers.py

    Args:
        model (nn.Module) : PyTorch model module, or LightningModule
        default_cfg (Optional[Dict]): default configuration for pretrained weights / target dataset
        in_chans (int): in_chans for model
        strict (bool): strict load of checkpoint
    """
    is_lightning_module = issubclass(model, pytorch_lightning.LightningModule)
    default_cfg = default_cfg or getattr(model, "default_cfg", None) or {}
    pretrained_path = default_cfg.pop("checkpoint_path", None)
    hf_hub_id = default_cfg.pop("hf_hub", None)
    if in_chans != default_cfg.get("input_channels", None):
        strict = False
        _logger.warning(
            f"Unable to convert pretrained weights because of mismatch in input channels, using random init for first layer."
        )
    if not is_lightning_module:
        # The model is passed uninitialized, so if not having to do the PL thing, should initialize here
        model = model(**default_cfg)
    if not pretrained_path and not hf_hub_id:
        _logger.warning("No pretrained weights exist for this model. Using random initialization.")
        return model
    if hf_hub_id and has_hf_hub(necessary=not pretrained_path):
        _logger.info(f"Loading pretrained weights from Hugging Face hub ({hf_hub_id})")
        if is_lightning_module:
            checkpoint = cache_file_from_hf(hf_hub_id)
            model.load_from_checkpoint(checkpoint, strict=strict, **default_cfg)
            return model
        state_dict = load_state_dict_from_hf(hf_hub_id)
    else:
        if is_lightning_module:
            model.load_from_checkpoint(pretrained_path, strict=strict, **default_cfg)
            return model
        state_dict = torch.load(pretrained_path, map_location="cpu")

    model.load_state_dict(state_dict, strict=strict)
    return model


class SatFlowModelHubMixin(ModelHubMixin):
    def __init__(self, *args, **kwargs):
        """
        Mix this class with your pl.LightningModule class to easily push / download the model via the Hugging Face Hub

        Example::

            >>> from satflow.models.hub import SatFlowModelHubMixin

            >>> class MyModel(nn.Module, SatFlowModelHubMixin):
            ...    def __init__(self, **kwargs):
            ...        super().__init__()
            ...        self.layer = ...
            ...    def forward(self, ...)
            ...        return ...

            >>> model = MyModel()
            >>> model.push_to_hub("mymodel") # Pushing model-weights to hf-hub

            >>> # Downloading weights from hf-hub & model will be initialized from those weights
            >>> model = MyModel.from_pretrained("username/mymodel")
        """

    def _create_model_card(self, path):
        model_card = MODEL_CARD_MARKDOWN.format(model_name=type(self).__name__)
        with open(os.path.join(path, "README.md"), "w") as f:
            f.write(model_card)

    def _save_config(self, module, save_directory):
        config = dict(module.hparams)
        path = os.path.join(save_directory, CONFIG_NAME)
        with open(path, "w") as f:
            json.dump(config, f)

    def _save_pretrained(self, save_directory: str, save_config: bool = True):
        # Save model weights
        path = os.path.join(save_directory, PYTORCH_WEIGHTS_NAME)
        model_to_save = self.module if hasattr(self, "module") else self
        torch.save(model_to_save.state_dict(), path)
        # Save model config
        if save_config and model_to_save.hparams:
            self._save_config(model_to_save, save_directory)
        # Save model card
        self._create_model_card(save_directory)

    @classmethod
    def _from_pretrained(
        cls,
        model_id,
        revision,
        cache_dir,
        force_download,
        proxies,
        resume_download,
        local_files_only,
        use_auth_token,
        map_location="cpu",
        strict=False,
        **model_kwargs,
    ):
        map_location = torch.device(map_location)

        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, PYTORCH_WEIGHTS_NAME)
        else:
            model_file = hf_hub_download(
                repo_id=model_id,
                filename=PYTORCH_WEIGHTS_NAME,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                use_auth_token=use_auth_token,
                local_files_only=local_files_only,
            )
        model = cls(**model_kwargs["config"])

        state_dict = torch.load(model_file, map_location=map_location)
        model.load_state_dict(state_dict, strict=strict)
        model.eval()

        return model
