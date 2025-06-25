import logging
from pathlib import Path

import numpy as np
import rasterio
from numpy.ma.core import MaskedArray
from PIL import Image, ImageDraw
from shapely import Polygon

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


def polygon_to_mask(image_size: tuple[int, int], polygon: Polygon) -> np.ndarray:
    """
    Convert a shapely polygon to a binary mask (numpy array).

    Args:
        image_size (tuple): Size of the output mask (width, height).
        polygon (shapely.geometry.Polygon): The Shapely polygon to be drawn.

    Returns:
        numpy.ndarray: Binary mask with 1 inside the polygon and 0 outside.
    """
    # Create a blank image (black background)
    img = Image.new("L", image_size, 0)  # 'L' mode is for grayscale (8-bit pixels)

    # Convert the polygon coordinates to a format that ImageDraw can use
    polygon_coords = [(x, y) for x, y in polygon.exterior.coords]

    # Draw the polygon on the image (255 for white)
    ImageDraw.Draw(img).polygon(polygon_coords, outline=1, fill=1)

    # Convert the image to a numpy array
    mask = np.array(img)

    return mask


def tif_paths(directory: Path) -> list[Path]:
    return sorted([pth for pth in directory.iterdir() if pth.suffix == ".tif"])


def geojson_paths(directory: Path) -> list[Path]:
    return sorted([pth for pth in directory.iterdir() if pth.suffix == ".geojson"])


def save_tif(output_path: Path, data: np.ndarray, meta: dict, onebit: bool = False) -> None:
    """
    Saves a raster dataset to a TIFF file.

    Parameters:
        output_path (Path): File path to save the output TIFF.
        data (numpy.ndarray): Raster data array.
        meta (dict): Raster metadata.
        onebit (bool): If the data should be saved as binary.
    """
    meta = meta.copy()
    meta.update({"driver": "GTiff"})

    if onebit:
        meta["nbits"] = 1

    if len(data.shape) == 2:
        data = data[None]

    meta.update({"count": data.shape[0]})

    with rasterio.open(output_path, "w", **meta) as dst:
        for i in range(data.shape[0]):
            dst.write(data[i], i + 1)
