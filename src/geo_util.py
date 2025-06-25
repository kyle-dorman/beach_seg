import logging
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from shapely import Polygon

local_logger = logging.getLogger(__name__)


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
