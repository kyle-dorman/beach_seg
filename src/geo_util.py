import logging
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from PIL import Image, ImageDraw
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.windows import Window, from_bounds
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import linemerge
from skimage import exposure, measure

from src.multichannel_img import broad_band

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


def get_masks(mask_dir: Path, pattern: str) -> list[Path]:
    """
    List shapefiles in mask_dir matching pattern.
    """
    return sorted(mask_dir.glob(pattern))


def load_and_merge_masks(mask_paths: list[Path]) -> gpd.GeoDataFrame:
    # Load and combine all geometries
    all_gdfs = [gpd.read_file(shp) for shp in mask_paths]
    assert all(gdf.crs == all_gdfs[0].crs for gdf in all_gdfs)
    combined_gdf = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True), crs=all_gdfs[0].crs)

    # Merge into a single geometry (can also use dissolve if attributes match)
    merged_geom = combined_gdf.union_all()

    # Wrap as a GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=[merged_geom], crs=combined_gdf.crs)

    return gdf


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


def extract_linestring(
    mask: np.ndarray, nodata_mask: np.ndarray, length_threshold: float = 0.3
) -> MultiLineString | LineString | None:
    """
    Extract a clean boundary LineString from a binary mask.
    Removes any segments that touch nodata or image edge.
    Filters out small lines (< threshold * max length).

    Args:
        mask (np.ndarray): Binary mask where 1 = true.
        nodata_mask (np.ndarray): Binary mask where 1 = nodata.
        length_threshold (float): Min fraction of longest segment to keep.

    Returns:
        shapely LineString or MultiLineString or None
    """
    h, w = mask.shape
    contours = measure.find_contours(mask.astype(float), level=0.5)

    if not contours:
        return None

    all_segments = []

    for contour in contours:
        for i in range(len(contour) - 1):
            p1 = contour[i]
            p2 = contour[i + 1]

            # Skip point if it touches the image edge
            if p1[0] <= 0 or p1[0] >= h - 1 or p1[1] <= 0 or p1[1] >= w - 1:
                continue

            mid = (p1 + p2) / 2.0
            row, col = int(round(mid[0])), int(round(mid[1]))

            # Skip if touching nodata
            y0 = row - 1
            y1 = row + 2
            x0 = col - 1
            x1 = col + 2
            if nodata_mask[y0:y1, x0:x1].any():
                continue

            all_segments.append((tuple(p1[::-1]), tuple(p2[::-1])))  # (x, y)

    if not all_segments:
        print("no segments")
        return None

    lines = [LineString([a, b]) for a, b in all_segments]
    merged = linemerge(lines)

    # Normalize to list
    if merged.is_empty:
        return None
    elif isinstance(merged, LineString):
        lines = [merged]
    elif isinstance(merged, MultiLineString):
        lines = list(merged.geoms)
    else:
        return None

    # Filter by length
    max_len = max(line.length for line in lines)
    min_len = length_threshold * max_len
    filtered = [line for line in lines if line.length >= min_len]

    if not filtered:
        return None
    elif len(filtered) == 1:
        return filtered[0]
    else:
        return MultiLineString(filtered)


def compute_raster_extent(
    tif_paths: list[Path],
) -> tuple[Affine, tuple[int, int], str]:
    """
    Compute the global affine transform, image shape (rows, cols), and crs
    for all GeoTIFFs in tif_dir. Assumes all share the same CRS and pixel resolution.
    Returns:
      transform: Affine transform for the full extent
      shape: (height, width) in pixels
      crs: (str) the CRS
    """
    assert len(tif_paths) > 0

    tif_paths = sorted(tif_paths)
    # Initialize from first file
    with rasterio.open(tif_paths[0]) as src0:
        left, bottom, right, top = src0.bounds
        tx, ty = src0.transform.a, -src0.transform.e
        base_crs = src0.crs
    # Accumulate bounds
    for tif in tif_paths[1:]:
        with rasterio.open(tif) as src:
            if src.crs != base_crs:
                raise ValueError(f"CRS mismatch: {tif} has {src.crs}, expected {base_crs}")
            b = src.bounds
            left = min(left, b.left)
            bottom = min(bottom, b.bottom)
            right = max(right, b.right)
            top = max(top, b.top)
    # Compute shape
    width = int(math.ceil((right - left) / tx))
    height = int(math.ceil((top - bottom) / ty))
    # Construct new transform
    new_transform = Affine(tx, 0.0, left, 0.0, -ty, top)
    return new_transform, (height, width), base_crs


def infer_date(mask_paths: list[Path]) -> str:
    """
    Ensure masks share a single YYYYMMDD date and return it.
    """
    dates = {p.stem.split("_")[1] for p in mask_paths}
    if len(dates) != 1:
        raise ValueError(f"Inconsistent mask dates: {dates}")
    return dates.pop()


def group_images_by_date(img_paths: list[Path]) -> dict[str, list[Path]]:
    """
    Group TIFFs by YYYYMMDD prefix.
    """
    groups: dict[str, list[Path]] = {}
    for p in img_paths:
        date = p.stem.split("_")[0]
        groups.setdefault(date, []).append(p)
    return groups


def rasterize_gdf(gdf: gpd.GeoDataFrame, out_shape: tuple[int, int], out_transform: Affine):
    # Rasterize into a binary mask
    return rasterize(
        [(geom, 1) for geom in gdf.geometry], out_shape=out_shape, transform=out_transform, fill=0, dtype="uint8"
    )


def merged_no_data_mask(water_mask: np.ndarray, veg_mask: np.ndarray) -> np.ndarray:
    h, w = veg_mask.shape
    line_no_data = np.zeros((h, w), dtype=np.bool_)
    for row in range(h):
        water_row = np.where(water_mask[row])[0]
        veg_row = np.where(veg_mask[row])[0]
        if not len(water_row) and not len(veg_row):
            continue

        elif not len(water_row):
            veg_start = veg_row[0]
            veg_end = veg_row[-1]

            line_no_data[row, :veg_start] = True
            line_no_data[row, veg_end:] = True
        elif not len(veg_row):
            water_start = water_row[0]
            water_end = water_row[-1]

            line_no_data[row, :water_start] = True
            line_no_data[row, water_end:] = True
        else:
            water_start = water_row[0]
            water_end = water_row[-1]
            veg_start = veg_row[0]
            veg_end = veg_row[-1]

            # veg to right of water
            if veg_start > water_end:
                line_no_data[row, :water_start] = True
                line_no_data[row, veg_end:] = True
            elif veg_start < water_end:
                # veg to left of water
                line_no_data[row, :veg_start] = True
                line_no_data[row, water_end:] = True

    return line_no_data


def create_per_day_crops(
    crops: list[tuple[int, int, int, int]],
    out_transform: Affine,
    tif_paths: list[Path],
    mask_array: np.ndarray,
    crop_size: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray]:
    """
    Generate image, nodata crops and mask crops from a reference TIFF and its mask array.
    """
    prompt_imgs = [np.zeros((crop_size, crop_size, 3), dtype=np.uint8) for _ in range(len(crops))]
    prompt_masks = [np.zeros((crop_size, crop_size), dtype=np.uint8) for _ in range(len(crops))]
    prompt_nodata_masks = [np.ones((crop_size, crop_size), dtype=np.uint8) for _ in range(len(crops))]
    full_size_yes_data = np.zeros_like(mask_array, dtype=np.bool_)

    for c_idx, (xmin, ymin, xmax, ymax) in enumerate(crops):
        for tif_path in tif_paths:
            with rasterio.open(tif_path) as src:
                ul_x, ul_y = out_transform * (xmin, ymin)  # type: ignore
                lr_x, lr_y = out_transform * (xmax, ymax)  # type: ignore

                win = from_bounds(left=ul_x, top=ul_y, right=lr_x, bottom=lr_y, transform=src.transform)
                img_crop, nodata = crop_with_mask(tif_path, win, crop_size)

                if np.all(nodata):
                    continue

                safe_assign_crop(full_size_yes_data, ~nodata, ymin, ymax, xmin, xmax, logic="or")

                update_mask = prompt_nodata_masks[c_idx] & ~nodata
                for i in range(3):
                    channel_update = np.where(update_mask, img_crop[:, :, i], prompt_imgs[c_idx][:, :, i])
                    prompt_imgs[c_idx][:, :, i] = channel_update
                prompt_nodata_masks[c_idx] = np.where(update_mask, 0, prompt_nodata_masks[c_idx])

        mask_crop = padded_mask_crop(mask_array, xmin, ymin, xmax, ymax, crop_size)
        prompt_masks[c_idx] = mask_crop

    return prompt_imgs, prompt_masks, prompt_nodata_masks, ~full_size_yes_data


def padded_mask_crop(
    mask_array: np.ndarray,
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    crop_size: int,
) -> np.ndarray:
    """
    Extract a crop from mask_array with zero padding for out-of-bounds areas.
    """
    padded = np.zeros((crop_size, crop_size), dtype=mask_array.dtype)
    h, w = mask_array.shape
    x0 = max(xmin, 0)
    x1 = min(xmax, w)
    y0 = max(ymin, 0)
    y1 = min(ymax, h)
    # Compute slices
    ystart = y0 - ymin
    yend = y1 - ymin
    xstart = x0 - xmin
    xend = x1 - xmin
    padded[ystart:yend, xstart:xend] = mask_array[y0:y1, x0:x1]
    return padded


def safe_assign_crop(
    output: np.ndarray, crop: np.ndarray, ymin: int, ymax: int, xmin: int, xmax: int, logic: str = "or"
) -> None:
    """
    Safely write crop into output[ymin:ymax, xmin:xmax],
    clipping to output bounds if necessary.
    """
    h, w = output.shape
    # Clip destination bounds
    dy0 = max(ymin, 0)
    dy1 = min(ymax, h)
    dx0 = max(xmin, 0)
    dx1 = min(xmax, w)

    # Corresponding source slice offsets
    sy0 = dy0 - ymin
    sy1 = sy0 + (dy1 - dy0)
    sx0 = dx0 - xmin
    sx1 = sx0 + (dx1 - dx0)

    if sy1 <= sy0 or sx1 <= sx0:
        return

    if logic == "or":
        output[dy0:dy1, dx0:dx1] |= crop[sy0:sy1, sx0:sx1]
    else:
        to_update = output[dy0:dy1, dx0:dx1] < crop[sy0:sy1, sx0:sx1]
        output[dy0:dy1, dx0:dx1][to_update] = crop[sy0:sy1, sx0:sx1][to_update]


def crop_with_mask(pth: Path, win: Window, crop_size: int) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(pth) as src:
        if src.count == 8:
            mask = src.read(
                1,
                window=win,
                masked=True,
                out_shape=(crop_size, crop_size),
                boundless=True,
                resampling=Resampling.nearest,
                fill_value=0,
            ).mask
            if np.all(mask):
                return np.zeros((crop_size, crop_size, 3), dtype=np.uint8), mask
            data = src.read(
                list(range(1, 9)),
                window=win,
                out_dtype=np.float32,
                out_shape=(8, crop_size, crop_size),
                boundless=True,
                masked=False,
                resampling=Resampling.bilinear,
                fill_value=0,
            )
            img = broad_band(data, mask)
        else:
            mask = src.read(
                [4, 3, 2],
                window=win,
                masked=True,
                out_shape=(3, crop_size, crop_size),
                boundless=True,
                resampling=Resampling.nearest,
                fill_value=0,
            ).mask
            if np.all(mask):
                return np.zeros((crop_size, crop_size, 3), dtype=np.uint8), mask
            data = src.read(
                (4, 3, 2),
                window=win,
                out_dtype=np.float32,
                out_shape=(3, crop_size, crop_size),
                boundless=True,
                masked=False,
                resampling=Resampling.bilinear,
                fill_value=0,
            )
            img = np.log10(1 + data)
            img -= img[~mask].min()
            img /= img[~mask].max()
            img[mask] = 0
            img = img.transpose((1, 2, 0)).copy()
            mask = mask[0]

    img = np.array(exposure.equalize_adapthist(img) * 255, dtype=np.uint8)
    return img, mask
