from pathlib import Path
from typing import Union

import pyvips
from numpy import uint8
from numpy.typing import NDArray


def read_image_vips(image_path: Union[Path, str]) -> NDArray[uint8]:
    """
    Reads an image using PyVIPS, converts it to sRGB color space,
    and removes the alpha channel if it exists.

    Parameters:
    - image_path (Union[Path, str]): Path to the image file.

    Returns:
    - NDArray[np.uint8]: The image as a NumPy array with 3 channels (RGB).
    """
    # Load the image using PyVIPS
    image = pyvips.Image.new_from_file(
        str(image_path),
        access="sequential",
    )

    # Convert image to sRGB color space
    image = image.colourspace("srgb")

    # Convert the image to NumPy array
    image_np = image.numpy()

    if image_np.shape[-1] == 4:
        image_np = image_np[..., :3]

    return image_np
