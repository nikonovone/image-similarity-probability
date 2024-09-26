from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.utils import read_image_vips


class PairComparisonDataset(Dataset):
    def __init__(self, triplets: List, transform=None):
        self.triplets = triplets
        self.transform = transform or transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ],
        )

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fetch precomputed triplet (anchor, positive, negative)."""
        anchor_path, positive_path, negative_path = self.triplets[idx]

        # Read the images
        anchor_img = read_image_vips(anchor_path)
        positive_img = read_image_vips(positive_path)
        negative_img = read_image_vips(negative_path)

        # Apply transformations
        if self.transform:
            anchor_img = self.transform(image=anchor_img)["image"]
            positive_img = self.transform(image=positive_img)["image"]
            negative_img = self.transform(image=negative_img)["image"]

        return anchor_img, positive_img, negative_img
