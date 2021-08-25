"""
Originally Taken from https://github.com/rwightman/pytorch-image-models/blob/acd6c687fd1c0507128f0ce091829b233c8560b9/timm/models/hub.py
"""

import json
import logging
import os
from functools import partial
from typing import Union

import pytorch_lightning
import torch

try:
    from torch.hub import get_dir
except ImportError:
    from torch.hub import _get_torch_home as get_dir

from satflow import __version__

try:
    from huggingface_hub import hf_hub_url
    from huggingface_hub import cached_download

    cached_download = partial(cached_download, library_name="satflow", library_version=__version__)
except ImportError:
    hf_hub_url = None
    cached_download = None

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


def load_pretrained(model, default_cfg=None, in_chans=12, strict=True):
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
    if not pretrained_path and not hf_hub_id:
        _logger.warning("No pretrained weights exist for this model. Using random initialization.")
        return model
    if hf_hub_id and has_hf_hub(necessary=not pretrained_path):
        _logger.info(f"Loading pretrained weights from Hugging Face hub ({hf_hub_id})")
        if is_lightning_module:
            checkpoint = cache_file_from_hf(hf_hub_id)
            model.load_from_checkpoint(checkpoint, **default_cfg)
            return model
        state_dict = load_state_dict_from_hf(hf_hub_id)
    else:
        if is_lightning_module:
            model.load_from_checkpoint(pretrained_path, **default_cfg)
            return model
        state_dict = torch.load(pretrained_path, map_location="cpu")
    input_convs = default_cfg.get("first_conv", None)
    if input_convs is not None and in_chans != default_cfg.get("input_channels", None):
        strict = False
        _logger.warning(
            f"Unable to convert pretrained weights because of mismatch in input channels, using random init for first layer."
        )

    model.load_state_dict(state_dict, strict=strict)
    return model
