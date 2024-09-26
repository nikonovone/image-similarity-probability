import random

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

    def on_train_epoch_start(
        self,
        trainer: Trainer,
        pl_module: CompareModel,
    ):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Select a random index to sample a batch
            random_index = random.randint(0, len(trainer.train_dataloader) - 1)

            # Retrieve the sample batch from the dataloader
            for idx, triplet_batch in enumerate(trainer.train_dataloader):
                if idx == random_index:
                    anchor, positive, negative = triplet_batch
                    break

            visualizations = []
            labels = ["Anchor", "Positive", "Negative"]
            for img, label in zip([anchor, positive, negative], labels):
                img = img.mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)

                # Convert tensor image to PIL image
                pil_img = F.to_pil_image(img[0])
                img_with_label = cv2.putText(
                    np.array(pil_img),
                    label,
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 255, 0),
                    2,
                )
                # Convert back to tensor image
                img_with_label = torch.from_numpy(img_with_label)
                img_with_label = img_with_label.permute(2, 0, 1).to(
                    pl_module.device,
                    torch.uint8,
                )
                visualizations.append(img_with_label)

            # Create a grid of the visualized triplets
            grid = make_grid(visualizations, nrow=3, normalize=False)

            # Log the grid of images
            trainer.logger.experiment.add_image(
                "Triplet Batch Preview",
                img_tensor=grid,
                global_step=trainer.global_step,
            )
