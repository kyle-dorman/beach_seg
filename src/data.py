import logging
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
from src.ml_util import generate_square_crops_along_line, load_processor

logger = logging.getLogger(__name__)


# Based on lightning_lite.utilities.exceptions
class MisconfigurationException(Exception):
    """Exception used to inform users of misuse with Lightning."""


# See https://arxiv.org/pdf/2212.02499.pdf  at 3.1 Redefining Output Spaces as "Images" - Semantic Segmentation from PAINTER paper
# Taken from
# https://github.com/Abdullah-Meda/Painter/blob/main/Painter/data/coco_semseg/gen_color_coco_panoptic_segm.py#L31
def build_palette(num_labels: int) -> list[tuple[int, int, int]]:
    base = int(num_labels ** (1 / 3)) + 1
    margin = 256 // base

    # we assume that class_idx 0 is the background which is mapped to black
    color_list = [(0, 0, 0)]
    for location in range(num_labels):
        num_seq_r = location // base**2
        num_seq_g = (location % base**2) // base
        num_seq_b = location % base

        R = 255 - num_seq_r * margin
        G = 255 - num_seq_g * margin
        B = 255 - num_seq_b * margin

        color_list.append((R, G, B))

    return color_list


def randomise_mask_rgb(mask_np: np.ndarray) -> np.ndarray:
    """Randomly recolour class-IDs → RGB (H × W × 3, uint8). Class 0 stays black."""
    lut = (np.random.rand(256, 3) * 255).astype("uint8")
    lut[0] = 0
    return lut[mask_np]  # (H,W,3)


def generate_random_rgb_palette(num_labels: int, batch_size: int, device) -> torch.Tensor:
    # Create a random colour look‑up table for each sample
    # Shape: (B, N, 3)  uint8  –  class‑0 stays black
    lut = torch.randint(
        low=0,
        high=256,
        size=(batch_size, num_labels, 3),
        dtype=torch.uint8,
        device=device,
    )
    lut[:, 0] = 0  # class‑0 ➜ black

    return lut


def torch_apply_mask_rgb(palette: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
    """
    Vectorised Torch implementation.
    `input` can be (B,1,H,W) or (B,H,W) with integer class‑IDs.
    Returns (B,3,H,W) float32 in [0,1].
    """
    if input.ndim == 3:  # (B,H,W) ➜ (B,1,H,W)
        input = input.unsqueeze(1)
    mask = input.squeeze(1).to(torch.long)  # (B,H,W)
    B, _, _ = mask.shape
    device = mask.device

    # Fancy‑index LUT with broadcasting:
    #   lut[batch_idx, class_id]  ➜ (B,H,W,3)
    rgb = palette[torch.arange(B, device=device)[:, None, None], mask]  # (B,H,W,3)

    # Rearrange to channels‑first & scale to [0,1] float32
    out = rgb.permute(0, 3, 1, 2).to(dtype=torch.float32) / 255.0  # (B,3,H,W)
    return out


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

        for date, _ in date_merged_imgs.items():
            full_mask = self.date_masks.get(date, None)
            img, nodata = self.date_merged_imgs[date]
            for crop_idx, crop in enumerate(self.crops):
                if full_mask is not None:
                    _, _, mask = crop_tif(crop, img, nodata, full_mask, self.config.crop_size)
                    count_nodata = (mask == 0).sum()
                else:
                    count_nodata = 0
                self.imgs.append({"date": date, "crop_idx": crop_idx, "count_nodata": count_nodata})

        if create_prompts:
            # indices of the crops with the *least* nodata pixels
            n = self.config.n_prompts
            self.prompt_indices = np.argsort([d["count_nodata"] for d in self.imgs])[:n]
            self.prompt_imgs = []
            for i in self.prompt_indices:
                self.prompt_imgs.append(self.get_crop(self.imgs[i]))
            self.imgs = [self.imgs[i] for i in range(len(self.imgs)) if i not in self.prompt_indices]

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
        if not np.all(nodata) and np.all(crop_label == 0):
            crop_label[~nodata] = 1

        return {
            "crop_idx": crop_idx,
            "date": date,
            "image": crop_img.transpose((2, 0, 1)).copy(),
            "mask": crop_label,
            "nodata": crop_nodata,
        }

    def __getitem__(self, idx):
        return self.get_crop(self.imgs[idx])


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
        date_img_paths = {mask_date: ref_imgs}
    else:
        date_img_paths = groups

    date_merged_imgs = {}
    for date, img_paths in date_img_paths.items():
        merged_img, merged_no_data = merge_tifs(img_paths, out_shape, out_transform, crs)
        date_merged_imgs[date] = (merged_img, merged_no_data)

    return BeachSegDataset(
        date_merged_imgs=date_merged_imgs,
        date_masks={mask_date: merged_mask},
        crops=prompt_crops,
        config=config,
        create_prompts=train,
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
            stage: Either 'fit', 'train', 'validate', 'test', or 'predict'.
        """
        if stage in ["fit", "train"]:
            self.train_dataset = create_dataset(self.config, train=True)
            logger.info(f"🗂  Train dataset loaded with {len(self.train_dataset)} samples")

        if stage in ["fit", "validate"]:
            self.val_dataset = create_dataset(self.config, train=True)
            logger.info(f"🗂  Val dataset loaded with {len(self.val_dataset)} samples")

        # if stage in ["test"]:
        #     self.test_dataset = create_dataset(self.config, False)
        #     logger.info(f"🗂  Test dataset loaded with {len(self.test_dataset)} samples")

        if stage in ["predict"]:
            self.predict_dataset = create_dataset(self.config, False)
            logger.info(f"🗂  Predict dataset loaded with {len(self.predict_dataset)} samples")

        if not hasattr(self, "prompt_indices") is None:
            self.prompt_indices = self.train_dataset.prompt_indices
            self.prompt_imgs = self.train_dataset.prompt_imgs

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
