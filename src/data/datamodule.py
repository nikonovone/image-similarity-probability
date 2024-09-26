import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from lightning import LightningDataModule
from torch import Generator
from torch.utils.data import DataLoader, random_split

from src.utils import PROJECT_ROOT, DataConfig

from .dataset import PairComparisonDataset
from .transform import get_train_transforms, get_valid_transforms


class DefaultDataModule(LightningDataModule):  # noqa: WPS214
    def __init__(
        self,
        cfg: DataConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self._train_transforms = get_train_transforms(*cfg.img_size)
        self._valid_transforms = get_valid_transforms(*cfg.img_size)

        # Prevent hyperparameters from being stored in checkpoints.
        self.save_hyperparameters(logger=False)

    def _generate_triplets(self) -> List[Tuple[Path, Path, Path]]:
        """Precompute all possible triplets (anchor, positive, negative)."""
        triplets = []
        all_classes = list(self.image_groups.keys())

        for anchor_class in all_classes:
            positive_images = self.image_groups[anchor_class]

            # Ensure at least two images exist for triplet creation
            if len(positive_images) < 2:
                continue

            for i in range(len(positive_images)):
                for j in range(i + 1, len(positive_images)):
                    anchor = positive_images[i]

                    positive = positive_images[j]

                    # Sample a negative class different from the anchor class
                    negative_class = random.choice(
                        [cls for cls in all_classes if cls != anchor_class],
                    )
                    negative = random.choice(self.image_groups[negative_class])

                    # Add the triplet
                    triplets.append((anchor, positive, negative))

        return triplets

    def _group_images_by_class(self) -> Dict[str, List[Path]]:
        """Group images by their base class name."""
        dataset_dir = Path(PROJECT_ROOT / self.cfg.dataset_dir)
        all_images = list(dataset_dir.glob("*"))
        image_groups: Dict[str, List[Path]] = {}

        for img in all_images:
            name_parts = re.match(
                r"(.+)_(\d+)\..*",
                img.name,
            )  # Match base name and number
            if name_parts:
                base_name, _ = name_parts.groups()  # Group by base name
                if base_name not in image_groups:
                    image_groups[base_name] = []
                image_groups[base_name].append(img)

        return image_groups

    def prepare_data(self):
        self.image_groups = self._group_images_by_class()
        self.all_triplets = self._generate_triplets()

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets based on the current stage (train/validate/test)."""
        # Initialize the full dataset (could be any dataset, e.g., TripletDataset)
        train_size, val_size, test_size = self.cfg.data_split

        generator = Generator().manual_seed(self.cfg.seed)
        # Split the full dataset into train, val, and test sets
        self.train_data, self.val_data, self.test_data = random_split(
            self.all_triplets,
            [train_size, val_size, test_size],
            generator,
        )

        if stage == "fit" or stage is None:
            self.train_dataset = PairComparisonDataset(
                self.train_data,
                self._train_transforms,
            )
            self.val_dataset = PairComparisonDataset(
                self.val_data,
                self._valid_transforms,
            )
        elif stage == "test":
            self.test_dataset = PairComparisonDataset(
                self.test_data,
                self._valid_transforms,
            )

    def train_dataloader(self) -> DataLoader:
        """Return the DataLoader for the training dataset."""
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=True,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the DataLoader for the validation dataset."""
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=True,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the DataLoader for the test dataset."""
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=True,
            shuffle=False,
        )
