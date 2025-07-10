import logging
import shutil
from pathlib import Path

import click
import numpy as np
import torch
from shapely.ops import transform
from tqdm import tqdm
from transformers import SegGptForImageSegmentation, SegGptImageProcessor

from src.util.geo_util import (
    compute_raster_extent,
    create_per_day_crops,
    extract_linestring,
    get_masks,
    group_images_by_date,
    infer_date,
    load_and_merge_masks,
    merged_no_data_mask,
    rasterize_gdf,
    safe_assign_crop,
    save_shapefile,
    save_tif,
)
from src.util.ml_util import generate_square_crops_along_line, load_model, load_processor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


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


# --- CLI Entrypoint with Click ---
@click.command()
@click.option("--base-dir", "-b", type=click.Path(exists=True), required=True, help="Base project directory")
@click.option("--crop-size", "-c", default=224, show_default=True, help="Crop size in pixels")
@click.option("--buffer-factor", "-f", default=0.125, show_default=True, help="Buffer as fraction of crop size")
@click.option(
    "--prompt-ckpt", type=click.Path(exists=True, dir_okay=False), help="Load an existing learned prompt (.pt)"
)
def main(
    base_dir: str,
    crop_size: int,
    buffer_factor: float,
    prompt_ckpt: str,
) -> None:
    """
    Main entrypoint: runs SegGPT shoreline extraction and saves results using a pre‑trained prompt.
    """
    if prompt_ckpt is None:
        raise click.BadParameter("--prompt-ckpt is now required. Prompt training has moved to train_prompts.py.")
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

    # Load model
    logger.info("Loading model")
    model = load_model("BAAI/seggpt-vit-large")
    processor = load_processor("BAAI/seggpt-vit-large")
    logger.info("Done loading model")

    # ------------------------------------------------------------------
    # PROMPT  (load only; training removed)
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
        # Training logic removed; use train_prompts.py to create a prompt checkpoint
        raise click.BadParameter("--prompt-ckpt must be provided; training has been removed.")

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
