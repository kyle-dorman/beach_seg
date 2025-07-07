import logging
from pathlib import Path
from typing import Any

import kornia.augmentation as K
import lightning.pytorch as pl
import numpy as np
import torch
from kornia.constants import DataKey, Resample
from PIL import Image
from PIL.Image import Resampling
from torch.utils.data import DataLoader, Dataset

from src.config import BeachSegConfig, num_workers
from src.geo_util import (
    compute_raster_extent,
    create_per_day_crops,
    extract_linestring,
    get_masks,
    group_images_by_date,
    infer_date,
    load_and_merge_masks,
    merge_tifs,
    merged_no_data_mask,
    rasterize_gdf,
)
from src.ml_util import generate_square_crops_along_line, load_processor

logger = logging.getLogger(__name__)


# Based on lightning_lite.utilities.exceptions
class MisconfigurationException(Exception):
    """Exception used to inform users of misuse with Lightning."""


def randomise_mask_rgb(mask_np: np.ndarray) -> np.ndarray:
    """Randomly recolour class-IDs → RGB (H × W × 3, uint8). Class 0 stays black."""
    lut = (np.random.rand(256, 3) * 255).astype("uint8")
    lut[0] = 0
    return lut[mask_np]  # (H,W,3)


def torch_randomize_mask_rgb(input: torch.Tensor, num_labels: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorised Torch implementation.
    `input` can be (B,1,H,W) or (B,H,W) with integer class‑IDs.
    Returns (B,3,H,W) float32 in [0,1].
    """
    if input.ndim == 3:  # (B,H,W) ➜ (B,1,H,W)
        input = input.unsqueeze(1)
    mask = input.squeeze(1).to(torch.long)  # (B,H,W)
    b, h, w = mask.shape
    device = mask.device

    # Create a random colour look‑up table for each sample
    # Shape: (B, 256, 3)  uint8  –  class‑0 stays black
    lut = torch.randint(
        low=0,
        high=256,
        size=(b, num_labels, 3),
        dtype=torch.uint8,
        device=device,
    )
    lut[:, 0] = 0  # class‑0 ➜ black

    # Fancy‑index LUT with broadcasting:
    #   lut[batch_idx, class_id]  ➜ (B,H,W,3)
    rgb = lut[torch.arange(b, device=device)[:, None, None], mask]  # (B,H,W,3)

    # Rearrange to channels‑first & scale to [0,1] float32
    out = rgb.permute(0, 3, 1, 2).to(dtype=torch.float32) / 255.0  # (B,3,H,W)
    return out, lut


class BeachSegDataset(Dataset):
    """
    Wraps a list of prompt crops.

    Each __getitem__ returns SegGPT-ready tensors:
        {
            'prompt_pixel_values': Tensor(3,H,W),
            'prompt_masks'       : Tensor(1,H,W)
        }
    """

    def __init__(
        self,
        date_img_paths: dict[str, list[Path]],
        date_masks: dict[str, np.ndarray] | None,
        crops: list[tuple[int, int, int, int]],
        out_shape: tuple[int, int],
        out_transform: Any,
        crs: str,
        config: BeachSegConfig,
    ) -> None:
        if date_masks is None:
            date_masks = {}

        self.imgs = []
        logger.info("Creating crops")
        for date, img_paths in date_img_paths.items():
            mask = date_masks.get(date, None)

            merged_img, merged_no_data = merge_tifs(img_paths, out_shape, out_transform, crs)

            imgs, masks, nodatas = create_per_day_crops(crops, merged_img, merged_no_data, mask, config.crop_size)

            for idx, (img, mask, nodata) in enumerate(zip(imgs, masks, nodatas)):
                img = Image.fromarray(img)
                img = img.resize((config.inpt_size, config.inpt_size), resample=config.resample)
                img = np.array(img).astype(np.float32) / 255.0
                mask = Image.fromarray(mask)
                mask = np.array(mask.resize((config.inpt_size, config.inpt_size), resample=Resampling.NEAREST))
                nodata = Image.fromarray(nodata)
                nodata = np.array(nodata.resize((config.inpt_size, config.inpt_size), resample=Resampling.NEAREST))

                # Hack to get nodata mask in when we don't have labels
                if not np.all(nodata) and np.all(mask == 0):
                    mask[~nodata] = 1

                self.imgs.append(
                    {
                        "crop_idx": idx,
                        "date": date,
                        "image": img.transpose((2, 0, 1)).copy(),
                        "mask": mask,
                        "nodata": nodata,
                    }
                )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx]


def create_dataset(config: BeachSegConfig, train: bool) -> BeachSegDataset:
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

        return BeachSegDataset(
            date_img_paths={mask_date: ref_imgs},
            date_masks={mask_date: merged_mask},
            crops=prompt_crops,
            out_shape=out_shape,
            out_transform=out_transform,
            crs=crs,
            config=config,
        )

    # val/pred
    return BeachSegDataset(
        groups,
        {mask_date: merged_mask},
        prompt_crops,
        out_shape=out_shape,
        out_transform=out_transform,
        crs=crs,
        config=config,
    )


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
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        self.train_dataset = create_dataset(self.config, True)
        logger.info(f"🗂  Train dataset loaded with {len(self.train_dataset)} samples")

        # if stage in ["fit", "validate"]:
        #     self.val_dataset = create_dataset(self.config, False)
        #     logger.info(f"🗂  Val dataset loaded with {len(self.val_dataset)} samples")

        # if stage in ["test"]:
        #     self.test_dataset = create_dataset(self.config, False)
        #     logger.info(f"🗂  Test dataset loaded with {len(self.test_dataset)} samples")

        if stage in ["predict"]:
            self.predict_dataset = create_dataset(self.config, False)
            logger.info(f"🗂  Predict dataset loaded with {len(self.predict_dataset)} samples")

        # indices of the crops with the *least* nodata pixels
        self.prompt_indices = np.argsort([np.sum(d["mask"] == 0) for d in self.train_dataset.imgs])[
            : self.config.n_prompts
        ]
        self.prompt_imgs = [self.train_dataset.imgs[i] for i in self.prompt_indices]
        self.train_dataset.imgs = [
            self.train_dataset.imgs[i] for i in range(len(self.train_dataset.imgs)) if i not in self.prompt_indices
        ]
        logger.info(
            f"🎯 Selected {len(self.prompt_indices)} prompt crops "
            f"(keeping {len(self.train_dataset.imgs)} for training)"
        )
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
                split = "train"
            elif self.trainer.validating or self.trainer.sanity_checking:
                split = "val"
            elif self.trainer.testing:
                split = "test"
            elif self.trainer.predicting:
                split = "predict"

            aug = self._valid_attribute(f"{split}_aug", "aug")
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
