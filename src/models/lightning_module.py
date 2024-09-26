from typing import Dict, List, Optional

import timm
from lightning import LightningModule
from torch import Tensor
from torch.optim import SGD, Adam, AdamW
from torchmetrics import MeanMetric

from src.utils.config import OptimizerConfig, SchedulerConfig
from src.utils.metrics import get_metrics
from src.utils.schedulers import get_cosine_schedule_with_warmup
from pytorch_metric_learning import losses
import torch


class CompareModel(LightningModule):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        optimizer_params: Optional[OptimizerConfig] = None,
        scheduler_params: Optional[SchedulerConfig] = None,
        margin: float = 1.0,
    ):
        super().__init__()

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
        )
        self.triplet_loss = losses.TripletMarginLoss(
            margin=margin,
        )

        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params

        self._train_loss = MeanMetric()
        self._valid_loss = MeanMetric()

        metrics = get_metrics()
        self._valid_metrics = metrics.clone(prefix="valid_")
        self._test_metrics = metrics.clone(prefix="test_")

        self.save_hyperparameters()

    def forward(self, images: Tensor) -> Tensor:
        return self.model(images)

    def training_step(self, batch: List[Tensor]) -> Dict[str, Tensor]:
        anchors, positives, negatives = batch
        anchor_embeddings = self(anchors).squeeze()
        positive_embeddings = self(positives).squeeze()
        negative_embeddings = self(negatives).squeeze()

        # Check if the number of embeddings matches the number of labels
        embeddings = torch.cat(
            [anchor_embeddings, positive_embeddings, negative_embeddings],
            dim=0,
        )
        labels = torch.cat(
            [
                torch.arange(len(anchors), device=self.device),
                torch.arange(len(positives), device=self.device),
                torch.arange(len(negatives), device=self.device),
            ],
        )

        # Create indices_tuple for triplet selection
        indices_tuple = (
            torch.arange(len(anchors)),
            torch.arange(len(anchors), len(anchors) + len(positives)),
            torch.arange(len(anchors) + len(positives), len(embeddings)),
        )

        loss = self.triplet_loss(embeddings, labels, indices_tuple, embeddings, labels)

        # Track loss
        self._train_loss(loss)
        self.log("step_loss", loss, on_step=True, prog_bar=True, logger=True)

        return {"loss": loss}

    def on_train_epoch_end(self) -> None:
        """Log the average training loss at the end of the epoch."""
        self.log(
            "mean_train_loss",
            self._train_loss.compute(),
            on_step=False,
            prog_bar=True,
            on_epoch=True,
        )
        self._train_loss.reset()

    def validation_step(
        self,
        batch: List[Tensor],
    ) -> None:
        """Validation step with triplet loss."""
        anchors, positives, negatives = batch
        anchor_embeddings = self(anchors).squeeze()
        positive_embeddings = self(positives).squeeze()
        negative_embeddings = self(negatives).squeeze()

        embeddings = torch.cat(
            [anchor_embeddings, positive_embeddings, negative_embeddings],
            dim=0,
        )
        labels = torch.cat(
            [
                torch.arange(len(anchors), device=self.device),
                torch.arange(len(positives), device=self.device),
                torch.arange(len(negatives), device=self.device),
            ],
        )

        # Create indices_tuple for triplet selection
        indices_tuple = (
            torch.arange(len(anchors)),
            torch.arange(len(anchors), len(anchors) + len(positives)),
            torch.arange(len(anchors) + len(positives), len(embeddings)),
        )
        # Compute triplet loss
        loss = self.triplet_loss(embeddings, labels, indices_tuple, embeddings, labels)
        # Track validation loss and metrics
        self._valid_loss(loss)

        preds = torch.cat(
            [anchor_embeddings, positive_embeddings, negative_embeddings],
            dim=0,
        ).norm(dim=1)
        target = torch.cat(
            [
                torch.zeros(len(anchors), dtype=torch.bool, device=self.device),
                torch.ones(len(positives), dtype=torch.bool, device=self.device),
                torch.zeros(len(negatives), dtype=torch.bool, device=self.device),
            ],
        )
        indexes = torch.cat(
            [
                torch.zeros(len(anchors)),
                torch.ones(len(positives)),
                torch.ones(len(negatives)),
            ],
        ).to(self.device, dtype=torch.long)

        # Update retrieval metrics - using predictions, targets, and indexes
        self._valid_metrics(preds, target, indexes=indexes)

    def on_validation_epoch_end(self) -> None:
        """Log the validation loss and metrics at the end of the epoch."""
        self.log(
            "mean_valid_loss",
            self._valid_loss.compute(),
            on_step=False,
            prog_bar=True,
            on_epoch=True,
        )
        self._valid_loss.reset()

        self.log_dict(self._valid_metrics.compute(), prog_bar=True, on_epoch=True)
        self._valid_metrics.reset()

    def test_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        """Test step for triplet loss."""
        anchors, positives, negatives = batch
        anchor_embeddings = self(anchors).squeeze()
        positive_embeddings = self(positives).squeeze()
        negative_embeddings = self(negatives).squeeze()

        embeddings = torch.cat(
            [anchor_embeddings, positive_embeddings, negative_embeddings],
            dim=0,
        )
        labels = torch.cat(
            [
                torch.arange(len(anchors), device=self.device),
                torch.arange(len(positives), device=self.device),
                torch.arange(len(negatives), device=self.device),
            ],
        )

        # Calculate Precision at 1 or other metrics
        self._test_metrics(embeddings, labels)

        return embeddings

    def on_test_epoch_end(self) -> None:
        """Log the test metrics at the end of the test epoch."""
        self.log_dict(self._test_metrics.compute(), on_epoch=True, prog_bar=True)
        self._test_metrics.reset()

    def configure_optimizers(self):
        opt_name = self.optimizer_params.name
        optimizer_params = self.optimizer_params.dict()
        optimizer_params.pop("name")

        # Filter parameters that require gradients
        params = filter(lambda p: p.requires_grad, self.parameters())

        # Select the optimizer
        if opt_name == "AdamW":
            optimizer = AdamW(params, **optimizer_params)
        elif opt_name == "Adam":
            optimizer = Adam(params, **optimizer_params)
        elif opt_name == "SGD":
            optimizer = SGD(params, **optimizer_params)
        else:
            raise ValueError(f'Unknown optimizer: "{opt_name}"')

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.scheduler_params.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
            num_cycles=self.scheduler_params.num_cycles,
        )

        return [optimizer], [
            {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        ]
