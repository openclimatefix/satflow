import os
import tempfile

from huggingface_hub import CONFIG_NAME, PYTORCH_WEIGHTS_NAME

from satflow.models.base import BaseModel
from satflow.models.hub import SatFlowModelHubMixin


class DummyModel(BaseModel, SatFlowModelHubMixin):
    # Define hyperparameters of various types to test serialisation / deserialisation
    # This follows the pattern adopted in LitMetNet and Perceiver
    def __init__(
        self, loss: str = "mse", forecast_steps: int = 42, pretrained: bool = True, lr: float = 0.05
    ):
        super(BaseModel, self).__init__()
        self.loss = loss
        self.forecast_steps = forecast_steps
        self.pretrained = pretrained
        self.lr = lr
        self.save_hyperparameters()


def test_satflow_mixin():
    # Override some of the hyperparameters
    config = {"loss": "rmse", "forecast_steps": 123}
    model = DummyModel(**config)
    with tempfile.TemporaryDirectory() as storage_folder:
        # Save model weights, config, and model card
        model.save_pretrained(storage_folder)
        folder_contents = os.listdir(storage_folder)
        assert len(folder_contents) == 3
        assert "README.md" in folder_contents
        assert PYTORCH_WEIGHTS_NAME in folder_contents
        assert CONFIG_NAME in folder_contents
        # Load new model and compare hyperparameters
        new_model = DummyModel.from_pretrained(storage_folder)
        assert new_model.hparams == model.hparams
