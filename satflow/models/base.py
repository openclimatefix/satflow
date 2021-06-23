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


class Model(torch.nn.Module, ABC):
    @property
    def name(self) -> str:
        return type(self).__name__

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]):
        return NotImplementedError
