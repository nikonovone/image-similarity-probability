import albumentations as albu
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_width: int, img_height: int) -> albu.Compose:
    return albu.Compose(
        [
            albu.Resize(height=img_height, width=img_width),
            albu.Normalize(normalization="min_max"),
            ToTensorV2(),
        ],
    )


def get_valid_transforms(img_width: int, img_height: int) -> albu.Compose:
    return albu.Compose(
        [
            albu.Resize(height=img_height, width=img_width),
            albu.Normalize(normalization="min_max"),
            ToTensorV2(),
        ],
    )
