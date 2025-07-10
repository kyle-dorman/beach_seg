import logging
from pathlib import Path

import numpy as np
import rasterio
from numpy.ma.core import MaskedArray
from PIL import Image, ImageColor

local_logger = logging.getLogger(__name__)


CLASS_COLORS = {"nodata": None, "water": "yellow", "veg": "blue", "sand": "hotpink"}


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


# ---------------------------------------------------------------------------
# Utility: write a single-band prediction mask to GeoTIFF
# ---------------------------------------------------------------------------
def write_mask_tif(mask: np.ndarray, transform, crs, path: Path) -> None:
    """
    Save a 2-D uint8 mask as a single-band GeoTIFF.

    Parameters
    ----------
    mask : np.ndarray
        2-D array of class indices (H, W) - uint8/uint16/etc.
    transform : affine.Affine
        Rasterio transform mapping pixel → CRS coordinates.
    crs : rasterio.crs.CRS
        Coordinate reference system.
    path : pathlib.Path
        Destination file path.
    """
    height, width = mask.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=mask.dtype,
        transform=transform,
        crs=crs,
        compress="lzw",
    ) as dst:
        dst.write(mask, 1)


def overlay_prediction(img: np.ndarray, pred: np.ndarray, classes: tuple[str, ...]) -> Image.Image:
    h, w, _ = img.shape
    base_img = Image.fromarray(img)  # RGB

    # Create a transparent RGBA layer for class overlays
    overlay_rgba = np.zeros((h, w, 4), dtype=np.uint8)  # RGBA

    alpha_val = int(255 * 0.3)  # 30 % opacity
    for cls_idx, color_name in enumerate([CLASS_COLORS[c] for c in classes]):
        if color_name is None:
            continue
        rgb = ImageColor.getrgb(color_name)  # convert "red" → (255, 0, 0)
        mask = pred == cls_idx
        overlay_rgba[mask] = (*rgb, alpha_val)

    overlay_img = Image.fromarray(overlay_rgba, mode="RGBA")
    blended = Image.alpha_composite(base_img.convert("RGBA"), overlay_img).convert("RGB")

    return blended
