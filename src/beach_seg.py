import logging
import math
import random as py_random
import shutil
from collections import deque
from pathlib import Path

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
import torchvision.transforms as T
from affine import Affine
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.windows import Window, from_bounds
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge, transform, unary_union
from skimage import exposure, measure
from tqdm import tqdm
from transformers import SegGptForImageSegmentation, SegGptImageProcessor

from src.multichannel_img import broad_band
from src.geo_util import save_tif

SEED = 42
py_random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


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


def load_and_merge_masks(mask_paths: list[Path]) -> gpd.GeoDataFrame:
    # Load and combine all geometries
    all_gdfs = [gpd.read_file(shp) for shp in mask_paths]
    assert all(gdf.crs == all_gdfs[0].crs for gdf in all_gdfs)
    combined_gdf = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True), crs=all_gdfs[0].crs)

    # Merge into a single geometry (can also use dissolve if attributes match)
    merged_geom = unary_union(combined_gdf.geometry)

    # Wrap as a GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=[merged_geom], crs=combined_gdf.crs)

    return gdf


def generate_square_crops_along_line(
    line: LineString | MultiLineString, crop_size: int, overlap: int
) -> list[tuple[int, int, int, int]]:
    """
    Compute a series of square crop windows along a LineString.

    Parameters
    ----------
    line : LineString
        A Shapely LineString whose coords are in pixel (x, y) space of the image.
    crop_size : int
        Side length (in pixels) of each square crop (N x N).
    overlap : int
        Number of pixels by which successive windows overlap (M). Must be < crop_size.
    Returns
    -------
    List of (xmin, ymin, xmax, ymax) integer tuples for each crop window.
    """

    if not (0 <= overlap < crop_size):
        raise ValueError("`overlap` must be >=0 and < `crop_size`")
    total_length = line.length
    step = crop_size - overlap

    # 1) build list of distances along the line where we'll center crops
    distances = list(np.arange(0, total_length + step, step))
    # ensure final point is covered
    if distances[-1] < total_length:
        distances.append(total_length)

    boxes = []
    half = crop_size / 2.0

    for d in distances:
        # 2) find the line position
        pt: Point = line.interpolate(d)
        cx, cy = pt.x, pt.y

        # 3) compute raw window
        xmin = int(round(cx - half))
        ymin = int(round(cy - half))
        xmax = xmin + crop_size
        ymax = ymin + crop_size

        boxes.append((xmin, ymin, xmax, ymax))

    return boxes


def get_masks(mask_dir: Path, pattern: str) -> list[Path]:
    """
    List shapefiles in mask_dir matching pattern.
    """
    return sorted(mask_dir.glob(pattern))


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


def save_shapefile(line: LineString | MultiLineString, out_fp: Path, crs) -> None:
    """
    Save a LineString to a Shapefile with given CRS.
    """
    gdf = gpd.GeoDataFrame({"geometry": [line]}, crs=crs)
    gdf.to_file(str(out_fp), driver="ESRI Shapefile")


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


# ──────────────────────────────────────────────────────────────────────────────
# Helper transforms
# ──────────────────────────────────────────────────────────────────────────────


def randomise_mask_rgb(mask_np: np.ndarray) -> np.ndarray:
    """Randomly recolour class-IDs → RGB (H × W × 3, uint8). Class 0 stays black."""
    lut = (np.random.rand(256, 3) * 255).astype("uint8")
    lut[0] = 0
    return lut[mask_np]  # (H,W,3)


# ──────────────────────────────────────────────────────────────────────────────


def train_learnable_prompt(
    model: SegGptForImageSegmentation,
    processor: SegGptImageProcessor,
    prompt_imgs: list[np.ndarray],
    prompt_masks: list[np.ndarray],
    prompt_nodatas: list[np.ndarray],
    *,
    num_labels: int = 3,
    epochs: int = 300,
    lr: float = 1e-2,
    n_prompts: int = 1,
    prompt_dropout: float = 0.1,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Learn one or more image-space prompt tensors.

    Returns
    -------
    prompt_pixel_values : torch.Tensor  # (P, 3, H, W)
    prompt_masks        : torch.Tensor  # (P, 1, H, W)
    """
    model.to(device).eval()  # type: ignore

    g = torch.Generator(device=device)
    g.manual_seed(SEED)

    # Suppose your processor defines these
    mean = processor.image_mean  # e.g. [0.485, 0.456, 0.406]
    std = processor.image_std  # e.g. [0.229, 0.224, 0.225]
    assert isinstance(mean, list)
    assert isinstance(std, list)

    # Build a “denormalize” transform
    denormalize = T.Normalize(mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std])
    normalize = T.Normalize(mean=mean, std=std)
    photo_jitter = T.ColorJitter(hue=0.1, saturation=0.15, brightness=0.15)
    geom_jitter = T.RandomAffine(degrees=5, translate=(0.02, 0.02))  # ≈±3 px on 224-px crop
    jitter_tensor = T.Compose(
        [denormalize, photo_jitter, geom_jitter, normalize]  # get back into [0,1]  # back to zero‐mean/unit‐var
    )

    # ── pick the P crops with most valid pixels ────────────────────────────────
    order = np.argsort([np.sum(n) for n in prompt_nodatas])[:n_prompts]

    prompt_params, fixed_masks, ema_buffers = [], [], []
    for idx in order:
        # pixel tensor
        init_px = processor(images=[prompt_imgs[idx]], data_format="channels_first", return_tensors="pt")[
            "pixel_values"
        ][0].to(device)
        p = torch.nn.Parameter(init_px, requires_grad=True)
        prompt_params.append(p)
        ema_buffers.append(p.detach().clone())

        # coloured mask tensor (CHW)
        coloured = randomise_mask_rgb(prompt_masks[idx]).transpose(2, 0, 1)
        fm = processor.preprocess(
            # prompt_images=[prompt_imgs[idx]],
            prompt_masks=[coloured],
            num_labels=num_labels,
            data_format="channels_first",
            return_tensors="pt",
            do_convert_rgb=False,
        )["prompt_masks"][0].to(device)
        fixed_masks.append(fm)

    params = torch.nn.ParameterList(prompt_params)
    ema = deque(ema_buffers, maxlen=n_prompts)

    # ── build query batches (exclude the P prompt crops) ───────────────────────
    batch_data = [
        processor.preprocess(
            prompt_images=[im],
            prompt_masks=[msk],
            num_labels=num_labels,
            data_format="channels_first",
            return_tensors="pt",
        )
        for i, (im, msk) in enumerate(zip(prompt_imgs, prompt_masks))
        if i not in order
    ]

    opt = torch.optim.AdamW(params, lr=lr)
    alpha = 0.99  # EMA decay

    for ep in tqdm(range(epochs), desc="Prompt training", unit="epoch", leave=False):
        running = 0.0
        for query in batch_data:
            opt.zero_grad()

            # choose prompt index
            pi: int = torch.randint(0, n_prompts, (), generator=g).item()  # type: ignore
            px = params[pi]

            # optional dropout preserving gradient flow
            if torch.rand((), generator=g).item() < prompt_dropout:
                px_fwd = px * 0
            else:
                px_fwd = jitter_tensor(px)
                # recolour mask for this epoch
                coloured = randomise_mask_rgb(prompt_masks[order[pi]]).transpose(2, 0, 1)
                fixed_masks[pi] = processor.preprocess(
                    # prompt_images=[prompt_imgs[order[pi]]],
                    prompt_masks=[coloured],
                    num_labels=num_labels,
                    data_format="channels_first",
                    return_tensors="pt",
                    do_convert_rgb=False,
                )["prompt_masks"][0].to(device)

            out = model(
                pixel_values=query["prompt_pixel_values"].to(device),
                prompt_pixel_values=px_fwd.unsqueeze(0),
                prompt_masks=fixed_masks[pi].unsqueeze(0),
                labels=query["prompt_masks"].to(device),
                embedding_type="semantic",
            )
            out.loss.backward()
            opt.step()
            running += out.loss.item()

            # EMA update
            ema[pi].mul_(alpha).add_(px.data, alpha=1 - alpha)

        tqdm.write(f"epoch {ep:03d}  loss={running / len(batch_data):.4f}")

    stacked_px = torch.stack(list(ema)).cpu()
    stacked_msk = torch.stack(fixed_masks).cpu()
    return stacked_px, stacked_msk


def run_seg_gpt_on_crops(
    prompt_img_tensors: list[torch.Tensor] | torch.Tensor,
    prompt_mask_tensors: list[torch.Tensor] | torch.Tensor,
    image_inputs: list[np.ndarray],
    input_nodatas: list[np.ndarray],
    crops: list[tuple[int, int, int, int]],
    out_size: tuple[int, int],
    buffer: int,
    crop_size: int,
    model: SegGptForImageSegmentation,
    processor: SegGptImageProcessor,
) -> np.ndarray:
    """
    For each date key in prediction_crops, run SegGPT with the prompt example
    and return a dict of prediction masks. Ignores nodata masks in prediction_crops.
    """
    output = np.zeros(out_size, dtype=np.uint8)

    valid_idxes = [i for i, nodata in enumerate(input_nodatas) if np.any(~nodata)]
    M = len(prompt_img_tensors)  # number of prompts you want to use

    with torch.no_grad():
        for idx in valid_idxes:
            query_img = image_inputs[idx]
            queries = [query_img] * M  # duplicate the SAME query image M times
            inputs = processor.preprocess(
                images=queries,
                num_labels=3,
                return_tensors="pt",
                data_format="channels_first",
            )
            batch_out = model(
                pixel_values=inputs["pixel_values"],
                prompt_pixel_values=prompt_img_tensors,
                prompt_masks=prompt_mask_tensors,
                embedding_type="semantic",
                feature_ensemble=True,
            )
            batch_out.pred_masks = batch_out.pred_masks[:1]

            target_sizes = [(crop_size, crop_size)]
            pred_tensors = processor.post_process_semantic_segmentation(batch_out, target_sizes, num_labels=3)

            pred = pred_tensors[0].cpu().numpy()
            pred[input_nodatas[idx]] = 0

            # Safely assign inner region of prediction into full output
            inner_pred = pred[buffer:-buffer, buffer:-buffer]
            (xmin, ymin, xmax, ymax) = crops[idx]
            safe_assign_crop(
                output, inner_pred, ymin + buffer, ymax - buffer, xmin + buffer, xmax - buffer, logic="ascending"
            )

    return output


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


def load_model() -> tuple[SegGptForImageSegmentation, SegGptImageProcessor]:
    checkpoint = "BAAI/seggpt-vit-large"
    image_processor = SegGptImageProcessor.from_pretrained(checkpoint)
    model = SegGptForImageSegmentation.from_pretrained(checkpoint)
    for p in model.parameters():  # freeze the backbone
        p.requires_grad_(False)
    model = torch.compile(model, mode="default")

    return model, image_processor  # type: ignore


# --- CLI Entrypoint with Click ---
@click.command()
@click.option("--base-dir", "-b", type=click.Path(exists=True), required=True, help="Base project directory")
@click.option("--crop-size", "-c", default=224, show_default=True, help="Crop size in pixels")
@click.option("--buffer-factor", "-f", default=0.125, show_default=True, help="Buffer as fraction of crop size")
@click.option("--train-prompt/--no-train-prompt", default=False, help="Optimise a learnable prompt before inference")
@click.option("--epochs", "-e", default=300, show_default=True, help="Prompt-training epochs")
@click.option("--lr", default=1e-2, show_default=True, help="Prompt-training learning-rate")
@click.option(
    "--prompt-ckpt", type=click.Path(exists=True, dir_okay=False), help="Load an existing learned prompt (.pt)"
)
@click.option(
    "--n-prompts",
    default=1,
    show_default=True,
    help="Number of independent prompt tensors to train (feature-ensemble).",
)
@click.option(
    "--prompt-dropout",
    default=0.1,
    show_default=True,
    help="Probability to zero-out the prompt tensor during training.",
)
def main(
    base_dir: str,
    crop_size: int,
    buffer_factor: float,
    train_prompt: bool,
    epochs: int,
    lr: float,
    n_prompts: int,
    prompt_dropout: float,
    prompt_ckpt: str | None,
) -> None:
    """
    Main entrypoint: runs SegGPT shoreline extraction and saves results.
    """
    base_path = Path(base_dir)
    buffer_px = int(crop_size * buffer_factor)

    # Paths
    mask_dir = base_path / "Masks"
    img_paths = list(base_path.glob("SatelliteImagery/*/*.tif"))
    classification_dir = base_path / "Classifications"
    shp_dir = base_path / "Lines"
    if classification_dir.exists():
        shutil.rmtree(classification_dir)
    classification_dir.mkdir(parents=True, exist_ok=True)
    if shp_dir.exists():
        shutil.rmtree(shp_dir)
    shp_dir.mkdir(parents=True, exist_ok=True)

    # Masks and dates
    veg_masks = get_masks(mask_dir, "Mask_*.shp")
    water_masks = get_masks(mask_dir, "WaterMask_*.shp")
    mask_date = infer_date(veg_masks + water_masks)

    # Group images
    groups = group_images_by_date(img_paths)
    ref_imgs = groups.pop(mask_date, [])

    # Compute extent & raster masks
    out_transform, out_shape, crs = compute_raster_extent(ref_imgs + sum(groups.values(), []))
    veg_gdf = load_and_merge_masks(veg_masks)
    veg_mask = rasterize_gdf(veg_gdf, out_shape, out_transform) == 1
    water_gdf = load_and_merge_masks(water_masks)
    water_mask = rasterize_gdf(water_gdf, out_shape, out_transform) == 1
    full_no_data = merged_no_data_mask(water_mask, veg_mask)

    sand_mask = ~(full_no_data | water_mask | veg_mask)
    merged_mask = np.zeros(water_mask.shape, dtype=np.uint8)
    merged_mask[water_mask] = 1
    merged_mask[veg_mask] = 2
    merged_mask[sand_mask] = 3

    # Extract lines and generate crops
    veg_line = extract_linestring(veg_mask, full_no_data)
    assert veg_line is not None
    water_line = extract_linestring(water_mask, full_no_data)
    assert water_line is not None
    # veg_crops = generate_square_crops_along_line(veg_line, crop_size, int(crop_size / 2))
    # veg_prompt_crops = generate_square_crops_along_line(veg_line, crop_size, 0)
    crops = generate_square_crops_along_line(water_line, crop_size, int(crop_size / 2))
    prompt_crops = generate_square_crops_along_line(water_line, crop_size, 0)

    # Load model
    logger.info("Loading model")
    model, processor = load_model()
    logger.info("Done loading model")

    # ------------------------------------------------------------------
    # PROMPT  (load | train | static)
    # ------------------------------------------------------------------
    if prompt_ckpt:
        logger.info(f"Loading learned prompt ➜ {prompt_ckpt}")
        ckpt = torch.load(prompt_ckpt, map_location="cpu")
        prompt_img_tensors: torch.Tensor = ckpt["prompt_pixel_values"]
        prompt_mask_tensors: torch.Tensor = ckpt["prompt_masks"]

        if prompt_img_tensors.ndim == 3:
            prompt_img_tensors = prompt_img_tensors.unsqueeze(0)
            prompt_mask_tensors = prompt_mask_tensors.unsqueeze(0)

    else:
        p_imgs, p_masks, p_nodata, _ = create_per_day_crops(
            prompt_crops, out_transform, ref_imgs, merged_mask, crop_size
        )
        keep = [i for i, nd in enumerate(p_nodata) if np.any(~nd)]
        p_imgs = [p_imgs[i] for i in keep]
        p_masks = [p_masks[i] for i in keep]
        p_nodata = [p_nodata[i] for i in keep]

        if train_prompt:
            logger.info(f"Training learnable prompt on {len(p_imgs)} crops")
            prompt_img_tensors, prompt_mask_tensors = train_learnable_prompt(
                model,
                processor,
                p_imgs,
                p_masks,
                p_nodata,
                epochs=epochs,
                lr=lr,
                n_prompts=n_prompts,
                prompt_dropout=prompt_dropout,
            )
            save_prompt = base_path / "prompt.pt"
            torch.save(
                {"prompt_pixel_values": prompt_img_tensors, "prompt_masks": prompt_mask_tensors},
                save_prompt,
            )
            logger.info(f"Saved prompt ➜ {save_prompt}")
        else:
            logger.info(f"Using {len(p_imgs)} static prompt images")
            inp = processor.preprocess(
                prompt_images=p_imgs,
                prompt_masks=p_masks,
                num_labels=3,
                return_tensors="pt",
                data_format="channels_first",
            )
            prompt_img_tensors = inp["prompt_pixel_values"]
            prompt_mask_tensors = inp["prompt_masks"]

    logger.info("Prompt tensors ready")

    for date, tif_paths in tqdm(groups.items(), desc="Generating Masks"):
        # if (classification_dir / "WetDryLine" / f"{date}.tif").exists():
        #     continue
        input_imgs, _, input_nodata, date_no_data = create_per_day_crops(
            crops, out_transform, tif_paths, merged_mask, crop_size
        )
        pred_mask = run_seg_gpt_on_crops(
            prompt_img_tensors,
            prompt_mask_tensors,
            input_imgs,
            input_nodata,
            crops,
            out_shape,
            buffer_px,
            crop_size,
            model,
            processor,
        )

        # Save per class results
        for kind, kind_idx in [("WetDryLine", 1), ("VegLine", 2)]:
            kind_mask = pred_mask == kind_idx

            # Save TIFF
            out_tif = classification_dir / kind / f"{date}.tif"
            out_shp = shp_dir / kind / f"{date}.shp"
            (classification_dir / kind).mkdir(exist_ok=True, parents=True)
            (shp_dir / kind).mkdir(exist_ok=True, parents=True)
            meta = {
                "driver": "GTiff",
                "dtype": "uint8",
                "width": out_shape[1],
                "height": out_shape[0],
                "count": 1,
                "crs": crs,
                "transform": out_transform,
            }
            save_tif(out_tif, kind_mask, meta, onebit=True)

            # Extract line and save shapefile
            line = extract_linestring(kind_mask, date_no_data)
            if line is not None and not line.is_empty:
                out_line = transform(lambda x, y: out_transform * (x, y), line)  # type: ignore
                save_shapefile(out_line, out_shp, crs)

        break


if __name__ == "__main__":
    logging.getLogger("fiona").setLevel(logging.WARNING)
    logging.getLogger("fiona._env").setLevel(logging.WARNING)
    main()
