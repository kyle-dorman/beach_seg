import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import tqdm
from affine import Affine
from dotenv import find_dotenv, load_dotenv
from omegaconf import OmegaConf
from shapely.ops import transform

from src.config import CLASSES
from src.util.geo_util import (
    compute_raster_extent,
    crop_tif,
    extract_linestring,
    get_masks,
    group_images_by_date,
    infer_date,
    load_and_merge_masks,
    merge_tifs,
    merged_no_data_mask,
    rasterize_gdf,
    save_shapefile,
)
from src.util.img_util import overlay_prediction, write_mask_tif
from src.util.ml_util import generate_square_crops_along_line, load_model, load_processor
from src.util.util import setup_logger

logger = logging.getLogger(__name__)


@dataclass
class PredConfig:
    data: Path | None = None
    results_dir: Path | None = None
    checkpoint: str = "BAAI/seggpt-vit-large"
    classes: tuple[str, ...] = CLASSES
    crop_size: int = 336
    n_prompts: int = 2
    debug: bool = False


@dataclass
class Dataset:
    out_transform: Affine
    crs: str
    out_shape: tuple[int, int]
    prompt_crops: list[tuple[int, int, int, int]]
    promt_img: np.ndarray
    prompt_nodata: np.ndarray
    prompt_label: np.ndarray
    grouped_imgs: dict[str, list[Path]]


def create_prompt_dataset(config: PredConfig) -> Dataset:
    assert config.data is not None

    # ── Gather reference imagery and masks ───────────────────────────────────
    mask_dir = config.data / "Masks"
    veg_masks = get_masks(mask_dir, "Mask_*.shp")
    water_masks = get_masks(mask_dir, "WaterMask_*.shp")
    mask_date = infer_date(veg_masks + water_masks)

    img_paths = list((config.data / "SatelliteImagery").glob("*/*.tif"))
    groups = group_images_by_date(img_paths)
    ref_imgs = groups.pop(mask_date, [])  # reference date imagery only
    assert len(ref_imgs)

    out_transform, out_shape, crs = compute_raster_extent(ref_imgs)
    veg_gdf = load_and_merge_masks(veg_masks)
    veg_mask = rasterize_gdf(veg_gdf, out_shape, out_transform) == 1
    water_gdf = load_and_merge_masks(water_masks)
    water_mask = rasterize_gdf(water_gdf, out_shape, out_transform) == 1
    full_no_data = merged_no_data_mask(water_mask, veg_mask)
    sand_mask = ~(full_no_data | water_mask | veg_mask)
    merged_mask = np.zeros(veg_mask.shape, dtype=np.uint8)
    if "water" in config.classes:
        merged_mask[water_mask] = config.classes.index("water")
    if "veg" in config.classes:
        merged_mask[veg_mask] = config.classes.index("veg")
    if "sand" in config.classes:
        merged_mask[sand_mask] = config.classes.index("sand")
    assert config.classes.index("nodata") == 0

    # ── Build crops along water line ─────────────────────────────────────────
    water_line = extract_linestring(water_mask, full_no_data)
    assert water_line is not None
    prompt_crops = generate_square_crops_along_line(water_line, config.crop_size, 0)

    merged_img, merged_no_data = merge_tifs(ref_imgs, out_shape, out_transform, crs)
    merged_img[merged_no_data, 1] = 255

    return Dataset(
        out_transform=out_transform,
        out_shape=out_shape,
        crs=crs,
        prompt_crops=prompt_crops,
        promt_img=merged_img,
        prompt_nodata=merged_no_data,
        prompt_label=merged_mask,
        grouped_imgs=groups,
    )


class Accumulator:
    def __init__(
        self,
        save_dir: Path,
        img: np.ndarray,
        nodata: np.ndarray,
        date: str,
        out_shape: tuple[int, int],
        classes: tuple[str, ...],
        out_transform: Any,
        crs: Any,
    ):
        self.out_shape = out_shape
        self.classes = classes
        self.out_transform = out_transform
        self.crs = crs

        self.img_dir = save_dir / "images"
        self.img_dir.mkdir(exist_ok=True)
        self.mask_dir = save_dir / "masks"
        self.mask_dir.mkdir(exist_ok=True)
        self.tif_dir = save_dir / "tif"
        self.tif_dir.mkdir(exist_ok=True)
        self.line_dir = save_dir / "lines"
        self.line_dir.mkdir(exist_ok=True)

        self.img = img
        self.nodata = nodata
        self.pred_counter = np.zeros((*self.out_shape, len(classes)), dtype=np.uint8)
        self.date = date

    def save(self):
        pred = np.argmax(self.pred_counter, axis=2).astype(np.uint8)

        blended = overlay_prediction(self.img, pred, self.classes)

        # Save image overlayed with mask
        save_path = self.img_dir / f"{self.date}.png"
        blended.save(save_path)

        # Save prediction as mask
        cv2.imwrite(str(self.mask_dir / f"{self.date}.png"), pred)

        # GeoTIFF export
        tif_path = self.tif_dir / f"{self.date}.tif"
        write_mask_tif(pred, self.out_transform, self.crs, tif_path)

        # Extract line and save shapefile
        for idx, cls in enumerate(self.classes[1:], 1):
            line = extract_linestring(pred == idx, self.nodata)
            if line is not None and not line.is_empty:
                out_line = transform(lambda x, y: self.out_transform * (x, y), line)  # type: ignore
                save_shapefile(out_line, self.line_dir / f"{cls}.shp", self.crs)

    def update(
        self,
        crop: tuple[int, int, int, int],
        one_hot_pred: np.ndarray,
    ):
        h, w = self.out_shape
        (xmin, ymin, xmax, ymax) = crop
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
            logger.warning(f"Invalid crop! {crop}")
            return

        self.pred_counter[dy0:dy1, dx0:dx1] += one_hot_pred[sy0:sy1, sx0:sx1]


def main():
    """
    predicts model
    """
    base_conf = OmegaConf.structured(PredConfig)
    cli_conf = OmegaConf.from_cli()
    conf: PredConfig = OmegaConf.merge(base_conf, cli_conf)  # type: ignore

    assert conf.results_dir is not None
    assert conf.data is not None

    # Save prediciton results
    model_predict_root = Path(conf.results_dir)
    model_predict_root.mkdir(exist_ok=True, parents=True)
    not_dot_dirs = filter(lambda p: not p.name.startswith("."), model_predict_root.iterdir())
    runs = [int(p.name) for p in not_dot_dirs]
    last_id = max(runs + [-1])
    predict_dir = model_predict_root / str(last_id + 1).zfill(5)
    predict_dir.mkdir(parents=True)

    # Setup logger
    setup_logger(logger, predict_dir)

    logger.info(f"Saving results to {predict_dir}")

    # Load model and data
    logger.info("Loading prompt")
    dataset = create_prompt_dataset(conf)

    prompt_path = predict_dir / "prompt.png"
    logger.info(f"Saving prompt to {prompt_path}")
    blended = overlay_prediction(dataset.promt_img, dataset.prompt_label, conf.classes)
    blended.save(predict_dir / "prompt_w_label.png")
    cv2.imwrite(str(prompt_path), dataset.promt_img[:, :, ::-1])

    logger.info("Loading model")
    model = load_model(conf.checkpoint)
    processor = load_processor(conf.checkpoint)

    crops = dataset.prompt_crops
    assert (
        len(crops) >= conf.n_prompts
    ), f"n_prompts({conf.n_prompts}) must be less than the number of valid crops({len(crops)})"
    num_classes = len(conf.classes)
    crop_prompts = []
    crop_labels = []
    for crop in tqdm.tqdm(crops, desc="Generating Prompts"):
        crop_img, crop_nodata, crop_label = crop_tif(
            crop, dataset.promt_img, dataset.prompt_nodata, dataset.prompt_label, conf.crop_size
        )
        assert crop_label is not None
        inputs = processor.preprocess(
            prompt_images=[crop_img],
            prompt_masks=[crop_label],
            num_labels=num_classes - 1,
            return_tensors="pt",
            data_format="channels_first",
        )
        crop_labels.append(crop_label)
        crop_prompts.append(inputs)

    best_crop_idxes = np.argsort([(cl != conf.classes[1]).sum() for cl in crop_labels])

    to_run = list(dataset.grouped_imgs.items())
    if conf.debug:
        to_run = to_run[:2]
    with torch.no_grad():
        for date, img_pths in tqdm.tqdm(to_run, desc="Model Inference"):
            merged_img, merged_no_data = merge_tifs(img_pths, dataset.out_shape, dataset.out_transform, dataset.crs)
            merged_img[merged_no_data, 1] = 255
            acc = Accumulator(
                predict_dir,
                merged_img,
                merged_no_data,
                date,
                dataset.out_shape,
                conf.classes,
                dataset.out_transform,
                dataset.crs,
            )

            for crop_idx, crop in enumerate(crops):
                crop_img, crop_nodata, _ = crop_tif(crop, merged_img, merged_no_data, None, conf.crop_size)
                if np.all(crop_nodata):
                    continue

                if crop_idx in best_crop_idxes[: conf.n_prompts]:
                    crop_idxes = best_crop_idxes[: conf.n_prompts]
                else:
                    stop = conf.n_prompts - 1
                    crop_idxes = [crop_idx] + best_crop_idxes[:stop].tolist()

                prompts = [crop_prompts[i] for i in crop_idxes]

                inputs = processor.preprocess(
                    images=[crop_img] * len(prompts),
                    num_labels=num_classes - 1,
                    return_tensors="pt",
                    data_format="channels_first",
                )
                batch_out = model(
                    pixel_values=inputs["pixel_values"],
                    prompt_pixel_values=torch.concat([p["prompt_pixel_values"] for p in prompts]),
                    prompt_masks=torch.concat([p["prompt_masks"] for p in prompts]),
                    embedding_type="instance",
                    feature_ensemble=True,
                )

                target_sizes = [(conf.crop_size, conf.crop_size)]
                batch_out.pred_masks = batch_out.pred_masks.mean(dim=0).unsqueeze(0)
                pred_tensors = processor.post_process_semantic_segmentation(
                    batch_out, target_sizes, num_labels=num_classes - 1
                )
                pred = pred_tensors[0].cpu().numpy()
                pred[crop_nodata] = 0
                one_hot = np.eye(num_classes, dtype=np.uint8)[pred]

                acc.update(crops[crop_idx], one_hot)
            acc.save()

    logger.info("Done!")


if __name__ == "__main__":
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
