from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import os


class NeptuneModelLogger(Callback):
    """
    Saves out the last and best models after each validation epoch. If the files don't exists, does nothing.

    Example::
        from pl_bolts.callbacks import NeptuneModelLogger
        trainer = Trainer(callbacks=[NeptuneModelLogger()])
    """

    def __init__(self) -> None:
        super().__init__()

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        try:
            trainer.logger.experiment[0]["model_checkpoints/last.ckpt"].upload(
                os.path.join(trainer.default_root_dir, "checkpoints", "last.ckpt")
            )
        except:
            print(
                f"No file to upload at {os.path.join(trainer.default_root_dir, 'checkpoints', 'last.ckpt')}"
            )
            pass

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        try:
            trainer.logger.experiment[0]["model_checkpoints/best.ckpt"].upload(
                os.path.join(trainer.default_root_dir, "checkpoints", "best.ckpt"),
            )
        except:
            print(
                f"No file to upload at {os.path.join(trainer.default_root_dir, 'checkpoints', 'best.ckpt')}"
            )
            pass
