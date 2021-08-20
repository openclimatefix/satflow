from pytorch_lightning import Callback, LightningModule, Trainer
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
            trainer.logger.experiment.log_artifact(
                os.path.join(trainer.log_dir, "checkpoints", "best.ckpt"), "models/best.ckpt"
            )
        except:
            pass
        try:
            trainer.logger.experiment.log_artifact(
                os.path.join(trainer.log_dir, "checkpoints", "last.ckpt"), "models/last.ckpt"
            )
        except:
            pass
