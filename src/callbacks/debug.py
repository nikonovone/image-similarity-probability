import random
from typing import Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from lightning import Callback, LightningModule, Trainer
from torchinfo import summary
from torchvision.utils import make_grid

from src.models import CompareModel
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def add_label_to_image(img: torch.Tensor, label: str) -> torch.Tensor:
    """Add a label to an image using OpenCV."""
    pil_img = F.to_pil_image(img)
    img_array = np.array(pil_img)

    img_with_label = cv2.putText(
        img_array,
        label,
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 255, 0),
        2,
    )

    return torch.from_numpy(img_with_label).permute(2, 0, 1).to("cpu", torch.uint8)


def create_triplet_visualization(
    triplet_batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Create a visualization grid for a batch of triplets."""
    anchor, positive, negative = triplet_batch
    labels = ["Anchor", "Positive", "Negative"]
    visualizations = []

    for i in range(len(anchor)):
        for img, label in zip([anchor[i], positive[i], negative[i]], labels):
            labeled_img = add_label_to_image(img, label)
            visualizations.append(labeled_img)

    return make_grid(visualizations, nrow=3, normalize=False)


class LogModelSummary(Callback):
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        images = next(iter(trainer.train_dataloader))[0]

        images = images.to(pl_module.device)
        pl_module: CompareModel = trainer.model
        summary(pl_module.model, input_data=images)


class VisualizeBatch(Callback):
    def __init__(self, every_n_epochs: int):
        super().__init__()
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_start(self, trainer: Trainer, pl_module: "CompareModel"):
        if trainer.current_epoch % self.every_n_epochs == 0:
            triplet_batch = self.get_random_batch(trainer.train_dataloader)
            grid = create_triplet_visualization(triplet_batch)

            trainer.logger.experiment.add_image(
                "Triplet Batch Preview",
                img_tensor=grid,
                global_step=trainer.global_step,
            )

    @staticmethod
    def get_random_batch(dataloader) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a random batch from the dataloader."""
        random_index = random.randint(0, len(dataloader) - 1)
        for idx, batch in enumerate(dataloader):
            if idx == random_index:
                return batch
        raise IndexError("Failed to retrieve a random batch from the dataloader.")
