from abc import ABC, abstractmethod
from typing import Any, Dict, Type
import torch.nn

REGISTERED_MODELS = {}


def register_model(cls: Type[torch.nn.Module]):
    global REGISTERED_MODELS
    name = cls.__name__
    assert name not in REGISTERED_MODELS, f"exists class: {REGISTERED_MODELS}"
    REGISTERED_MODELS[name] = cls
    return cls


def get_model(name: str) -> Type[torch.nn.Module]:
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
        model = get_model(model_name).from_config(pretrained=pretrained, **kwargs)
    else:
        raise RuntimeError("Unknown model (%s)" % model_name)

    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)

    return model
