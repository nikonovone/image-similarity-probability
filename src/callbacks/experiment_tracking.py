import os
from typing import Dict, Optional

from clearml import OutputModel, Task
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from src.utils.config import ExperimentConfig
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class ClearMLTracking(Callback):
    def __init__(
        self,
        cfg: ExperimentConfig,
        label_enumeration: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.label_enumeration = label_enumeration
        self.task: Optional[Task] = None
        self.output_model: Optional[OutputModel] = None

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._setup_task()

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        final_checkpoint = select_checkpoint_for_export(trainer)
        log.info('Uploading checkpoint "%s" to ClearML', final_checkpoint)
        if self.output_model is not None:
            self.output_model.update_weights(
                weights_filename=final_checkpoint,
                auto_delete_file=True,
            )

    def _setup_task(self):
        Task.force_requirements_env_freeze()  # or use task.set_packages() for more granular control.
        self.task = Task.init(
            project_name=self.cfg.project_name,
            task_name=self.cfg.experiment_name,
            # If `output_uri=True` uses default ClearML output URI,
            # can use string value to specify custom storage URI like S3.
            output_uri=True,
        )
        self.task.connect_configuration(configuration=self.cfg.model_dump())
        self.output_model = OutputModel(
            task=self.task,
            label_enumeration=self.label_enumeration,
        )


def select_checkpoint_for_export(trainer: Trainer) -> str:
    checkpoint_cb: Optional[ModelCheckpoint] = trainer.checkpoint_callback
    if checkpoint_cb is not None:
        checkpoint_path = checkpoint_cb.best_model_path
        if os.path.isfile(checkpoint_path):
            log.info("Selected best checkpoint: %s", checkpoint_path)
            return checkpoint_path
        else:
            log.warning(
                "Couldn't find the best checkpoint, probably callback haven't been called yet.",
            )

    checkpoint_path = os.path.join(trainer.log_dir, "checkpoint-from-trainer.pth")
    trainer.save_checkpoint(checkpoint_path)
    log.info("Saved checkpoint: %s.", checkpoint_path)
    return checkpoint_path
