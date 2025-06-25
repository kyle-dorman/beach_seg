import logging
from pathlib import Path

import numpy as np
from numpy.ma.core import MaskedArray
from PIL import Image

local_logger = logging.getLogger(__name__)


def contrast_stretch(image: np.ndarray | MaskedArray, p_low: int = 2, p_high: int = 98) -> np.ndarray:
    """Perform contrast stretching using percentiles."""
    image = image.astype(np.float32)
    orig_shape = image.shape
    if len(orig_shape) == 2:
        image = image[None]
    for idx in range(image.shape[0]):
        channel = image[idx]
        if isinstance(channel, MaskedArray):
            v_min, v_max = np.percentile(channel.compressed(), (p_low, p_high))
        else:
            v_min, v_max = np.percentile(channel, (p_low, p_high))

        image[idx] = np.clip((channel - v_min) / (v_max - v_min), 0, 1)

    if len(orig_shape) == 2:
        image = image[0]

    return image


def masked_contrast_stretch(image: np.ndarray, mask: np.ndarray, p_low: int | None = 2, p_high: int = 98) -> np.ndarray:
    """Perform contrast stretching using percentiles."""
    image = image.astype(np.float32)
    orig_shape = image.shape
    if len(orig_shape) == 2:
        image = image[None]
    for idx in range(image.shape[0]):
        channel = image[idx]
        pp_low = p_low if p_low is not None else 0
        v_min, v_max = np.percentile(channel[mask], (pp_low, p_high))

        # If no p_low use 0 (absolute)
        if p_low is None:
            v_min = 0

        image[idx] = np.clip((channel - v_min) / (v_max - v_min), 0, 1)

    if len(orig_shape) == 2:
        image = image[0]

    return image


def write_1bit_png(filename: Path, img: np.ndarray) -> None:
    i = Image.fromarray(img).convert(mode="1", dither=Image.Dither.NONE)
    i.save(filename, optimize=True)
