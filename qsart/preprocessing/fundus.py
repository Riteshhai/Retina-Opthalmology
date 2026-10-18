"""Conservative black-border cropping, without modifying source images."""
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def crop_fundus(image: Image.Image, threshold: int = 10) -> Image.Image:
    """Crop a visible non-dark bounding box; preserve dark/empty images.

    This geometric operation does not certify that an image depicts a retina.
    A 1% minimum foreground avoids a tiny isolated bright pixel dominating.
    """
    rgb = ImageOps.exif_transpose(image).convert('RGB')
    mask = np.asarray(rgb).max(axis=2) > threshold
    if mask.mean() < .01:
        return rgb
    ys, xs = np.nonzero(mask)
    return rgb.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))


def preprocess_image(image: Image.Image, image_size: int = 224, crop: bool = True) -> Image.Image:
    """Convert RGB, optionally crop, letterbox square, then resize."""
    if not isinstance(image_size, int) or isinstance(image_size, bool) or image_size < 16:
        raise ValueError('image_size must be an integer >= 16')
    rgb = crop_fundus(image) if crop else ImageOps.exif_transpose(image).convert('RGB')
    return ImageOps.pad(rgb, (image_size, image_size), method=Image.Resampling.BILINEAR,
                        color=(0, 0, 0), centering=(.5, .5))


def load_fundus_image(path_or_bytes: str | Path | bytes, image_size: int = 224,
                      crop: bool = True) -> Image.Image:
    source = BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes
    with Image.open(source) as image:
        image.load()
        return preprocess_image(image, image_size=image_size, crop=crop)
