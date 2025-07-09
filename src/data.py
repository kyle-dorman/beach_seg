import logging
from typing import Any

import kornia.augmentation as K
import lightning.pytorch as pl
import numpy as np
import torch
import tqdm
from kornia.constants import DataKey, Resample
from PIL import Image
from PIL.Image import Resampling
from torch.utils.data import DataLoader, Dataset

from src.config import BeachSegConfig, num_workers
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
)
from src.util.ml_util import generate_square_crops_along_line, load_processor

logger = logging.getLogger(__name__)


# Based on lightning_lite.utilities.exceptions
class MisconfigurationException(Exception):
    """Exception used to inform users of misuse with Lightning."""


class BeachSegDataset(Dataset):
    """
    Wraps a list of prompt crops.
    """

    def __init__(
        self,
        date_merged_imgs: dict[str, tuple[np.ndarray, np.ndarray]],
        date_masks: dict[str, np.ndarray] | None,
        crops: list[tuple[int, int, int, int]],
        config: BeachSegConfig,
        create_prompts: bool = False,
    ) -> None:
        if date_masks is None:
            date_masks = {}

        self.imgs = []
        self.date_merged_imgs = date_merged_imgs
        self.date_masks = date_masks
        self.crops = crops
        self.config = config

        crop_size = self.config.crop_size

        for date, _ in date_merged_imgs.items():
            full_mask = self.date_masks.get(date, None)
            img, nodata = self.date_merged_imgs[date]
            for crop_idx, crop in enumerate(self.crops):
                if full_mask is not None:
                    _, _, mask = crop_tif(crop, img, nodata, full_mask, crop_size)
                    count_nodata = (mask == 0).sum()
                else:
                    count_nodata = 0
                self.imgs.append(
                    {"date": date, "crop_idx": crop_idx, "pct_nodata": count_nodata / crop_size / crop_size}
                )

        if create_prompts:
            # indices of the crops with the less than 50% nodata pixels
            self.prompt_imgs = [self.get_crop(img) for img in self.imgs]

    def __len__(self):
        return len(self.imgs)

    def get_crop(self, img_data: dict[str, Any]):
        date = img_data["date"]
        crop_idx = img_data["crop_idx"]
        crop = self.crops[crop_idx]
        img, nodata = self.date_merged_imgs[date]
        label = self.date_masks.get(date, None)

        crop_img, crop_nodata, crop_label = crop_tif(crop, img, nodata, label, self.config.crop_size)

        if crop_label is None:
            crop_label = np.zeros(crop_img.shape[:2], dtype=np.uint8)

        crop_img = Image.fromarray(crop_img)
        if self.config.inpt_size != self.config.crop_size:
            crop_img = crop_img.resize((self.config.inpt_size, self.config.inpt_size), resample=self.config.resample)
        crop_img = np.array(crop_img).astype(np.float32) / 255.0

        crop_label = Image.fromarray(crop_label)
        if self.config.inpt_size != self.config.crop_size:
            crop_label = np.array(
                crop_label.resize((self.config.inpt_size, self.config.inpt_size), resample=Resampling.NEAREST)
            )
        else:
            crop_label = np.array(crop_label)

        crop_nodata = Image.fromarray(crop_nodata)
        if self.config.inpt_size != self.config.crop_size:
            crop_nodata = np.array(
                crop_nodata.resize((self.config.inpt_size, self.config.inpt_size), resample=Resampling.NEAREST)
            )
        else:
            crop_nodata = np.array(crop_nodata)

        # Hack to get nodata mask in when we don't have labels
        if not np.all(crop_nodata) and np.all(crop_label == 0):
            crop_label[~crop_nodata] = 1

        return {
            "crop_idx": crop_idx,
            "date": date,
            "image": crop_img.transpose((2, 0, 1)).copy(),
            "mask": crop_label,
            "nodata": crop_nodata,
        }

    def __getitem__(self, idx):
        return self.get_crop(self.imgs[idx])


def create_dataset(config: BeachSegConfig, train: bool) -> tuple[BeachSegDataset, tuple[int, int], Any, Any]:
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
    merged_mask[water_mask] = config.classes.index("water")
    merged_mask[veg_mask] = config.classes.index("veg")
    merged_mask[sand_mask] = config.classes.index("sand")
    assert config.classes.index("nodata") == 0

    # ── Build crops along water line ─────────────────────────────────────────
    water_line = extract_linestring(water_mask, full_no_data)
    assert water_line is not None
    prompt_crops = generate_square_crops_along_line(water_line, config.crop_size, 0)

    if train:
        date_img_paths = {mask_date: ref_imgs}
    else:
        date_img_paths = groups

    date_merged_imgs = {}
    for date, img_paths in tqdm.tqdm(date_img_paths.items(), desc="Merging per day images"):
        merged_img, merged_no_data = merge_tifs(img_paths, out_shape, out_transform, crs)
        date_merged_imgs[date] = (merged_img, merged_no_data)

    dataset = BeachSegDataset(
        date_merged_imgs=date_merged_imgs,
        date_masks={mask_date: merged_mask},
        crops=prompt_crops,
        config=config,
        create_prompts=train,
    )

    return dataset, out_shape, out_transform, crs


class BeachSegDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: BeachSegConfig,
    ):
        super().__init__()
        logger.info("📦 Initialising BeachSegDataModule")
        self.config = config

        self.save_hyperparameters(config)
        self.preprocessor = load_processor(config)
        self.mean = self.preprocessor.image_mean
        self.std = self.preprocessor.image_std

        train_augs = [
            K.RandomVerticalFlip(p=self.config.vertical_flip),
            K.RandomHorizontalFlip(p=self.config.horizontal_flip),
            # K.RandomRGBShift(
            #     p=self.config.channel_shift_p,
            # ),
            K.ColorJiggle(
                hue=self.config.hue,
                saturation=self.config.saturation,
                contrast=self.config.contrast,
                brightness=self.config.brightness,
            ),
            K.RandomSharpness(sharpness=self.config.sharpness, p=self.config.sharpness_p),
            K.RandomErasing(scale=self.config.erasing_scale, p=self.config.erasing_p),
            K.RandomGaussianNoise(mean=self.config.gauss_mean, std=self.config.gauss_std, p=self.config.gauss_p),
            # K.RandomMosaic(
            #     output_size=(self.conf.chip_size, self.conf.chip_size),
            #     p=self.conf.mosaic_p,
            #     min_bbox_size=self.conf.chip_size // 10,
            # ),
            # K.RandomJigsaw(p=self.conf.jigsaw_p, grid=self.conf.jigsaw_grid),
            # K.RandomResizedCrop((self.config.inpt_size, self.config.inpt_size), scale=self.config.scale),
            # K.CenterCrop((self.config.inpt_size, self.config.inpt_size)),
            K.Normalize(mean=self.mean, std=self.std),
        ]
        self.train_aug = K.AugmentationSequential(
            *train_augs,
            data_keys=None,
            extra_args={DataKey.MASK: {"resample": Resample.NEAREST, "align_corners": None}},
        )

        augs = [
            K.CenterCrop((self.config.inpt_size, self.config.inpt_size)),
            K.Normalize(mean=self.mean, std=self.std),
        ]
        self.aug = K.AugmentationSequential(
            *augs,
            data_keys=None,
            extra_args={DataKey.MASK: {"resample": Resample.NEAREST, "align_corners": None}},
        )
        logger.info("   ↳ Augmentations configured " f"(train={len(self.train_aug)}, infer={len(self.aug)})")

    # ---------------- Lightning hooks ----------------

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'train', 'validate', 'test', or 'predict'.
        """
        if stage in ["fit", "train"]:
            self.train_dataset, _, _, _ = create_dataset(self.config, train=True)
            logger.info(f"🗂  Train dataset loaded with {len(self.train_dataset)} samples")

        if stage in ["fit", "validate"]:
            self.val_dataset, _, _, _ = create_dataset(self.config, train=True)
            logger.info(f"🗂  Val dataset loaded with {len(self.val_dataset)} samples")

        # if stage in ["test"]:
        #     self.test_dataset = create_dataset(self.config, False)
        #     logger.info(f"🗂  Test dataset loaded with {len(self.test_dataset)} samples")

        if stage in ["predict"]:
            self.predict_dataset, _, _, _ = create_dataset(self.config, False)
            logger.info(f"🗂  Predict dataset loaded with {len(self.predict_dataset)} samples")

        if hasattr(self, "train_dataset") and not hasattr(self, "prompt_imgs") is None:
            self.prompt_imgs = self.train_dataset.prompt_imgs
            logger.info(f"🎯 Selected {len(self.prompt_imgs)} prompt crops")

        logger.info("✅ DataModule setup complete")

    def train_dataloader(self):
        nw = num_workers(self.config)
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=nw,
            persistent_workers=nw > 0,
        )

    def val_dataloader(self):
        nw = num_workers(self.config)
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=nw,
            persistent_workers=nw > 0,
        )

    def predict_dataloader(self):
        nw = num_workers(self.config)
        return DataLoader(
            dataset=self.predict_dataset,
            batch_size=1,
            num_workers=nw,
        )

    def on_after_batch_transfer(self, batch: dict[str, torch.Tensor], dataloader_idx: int) -> dict[str, torch.Tensor]:
        """Apply batch augmentations to the batch after it is transferred to the device.

        Args:
            batch: A batch of data that needs to be altered or augmented.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A batch of data.
        """
        if self.trainer:
            if self.trainer.training:
                aug = self.train_aug
            else:
                aug = self.aug

            batch = aug(batch)

        return batch

    def _valid_attribute(self, *args: str) -> Any:
        """Find a valid attribute with length > 0.

        Args:
            args: One or more names of attributes to check.

        Returns:
            The first valid attribute found.

        Raises:
            MisconfigurationException: If no attribute is defined, or has length 0.
        """
        for arg in args:
            obj = getattr(self, arg)

            if obj is None:
                continue

            if not obj:
                msg = f"{self.__class__.__name__}.{arg} has length 0."
                raise MisconfigurationException(msg)

            return obj

        msg = f"{self.__class__.__name__}.setup must define one of {args}."
        raise MisconfigurationException(msg)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return K.Denormalize(mean=self.mean, std=self.std)(x)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return K.Normalize(mean=self.mean, std=self.std)(x)
