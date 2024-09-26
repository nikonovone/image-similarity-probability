import os

import lightning
from lightning import Trainer
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from torch import set_float32_matmul_precision
import sys

sys.path.append("D:/Code/Test_tasks/image-similarity-probability")
from src.callbacks.debug import LogModelSummary, VisualizeBatch
from src.callbacks.experiment_tracking import (
    ClearMLTracking,
)
from src.callbacks.freeze import FeatureExtractorFreezeUnfreeze
from src.data import DefaultDataModule
from src.models import CompareModel
from src.utils import PROJECT_ROOT, ExperimentConfig
from lightning.pytorch import loggers as pl_loggers


def train(cfg: ExperimentConfig):
    lightning.seed_everything(cfg.data_config.seed)
    set_float32_matmul_precision("medium")

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")
    datamodule = DefaultDataModule(cfg=cfg.data_config)

    callbacks = [
        LogModelSummary(),
        RichProgressBar(),
        VisualizeBatch(every_n_epochs=5),
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            save_top_k=3,
            monitor="valid_retrieval_precision",
            mode="max",
            every_n_epochs=1,
        ),
        FeatureExtractorFreezeUnfreeze(unfreeze_at_epoch=0),
    ]

    if cfg.track_in_clearml:
        tracking_cb = ClearMLTracking(cfg)
        callbacks += [
            tracking_cb,
        ]
    model = CompareModel(
        cfg.name_model,
        optimizer_params=cfg.optimizer_config,
        scheduler_params=cfg.scheduler_config,
    )

    trainer = Trainer(**dict(cfg.trainer_config), callbacks=callbacks, logger=tb_logger)
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    cfg_path = os.getenv("TRAIN_CFG_PATH", PROJECT_ROOT / "configs" / "train.yaml")
    cfg = ExperimentConfig.from_yaml(cfg_path)
    train(cfg=ExperimentConfig.from_yaml(cfg_path))
