#!/usr/bin/env python
"""
Train a learnable image-space prompt for SegGPT shoreline extraction.
Outputs a single .pt checkpoint containing:
  ├─ 'prompt_pixel_values'  # (P, 3, H, W)
  └─ 'prompt_masks'         # (P, 1, H, W)
"""
import logging
import random
from collections import deque
from pathlib import Path

import click
import numpy as np  # NEW
import torch
from torchvision import transforms as T
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
    merge_tifs,
    merged_no_data_mask,
    rasterize_gdf,
)
from src.util.ml_util import generate_square_crops_along_line, load_model, load_processor, randomise_mask_rgb

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


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


@click.command()
@click.option(
    "--base-dir",
    "-b",
    type=click.Path(exists=True),
    required=True,
    help="Project directory (expects Masks/, SatelliteImagery/ sub-folders)",
)
@click.option("--crop-size", "-c", default=224, show_default=True, help="Square crop size in pixels")
@click.option("--epochs", "-e", default=300, show_default=True, help="Prompt-training epochs")
@click.option("--lr", default=1e-2, show_default=True, help="Learning-rate")
@click.option("--n-prompts", default=1, show_default=True, help="Number of prompt tensors to learn (feature ensemble)")
@click.option("--prompt-dropout", default=0.1, show_default=True, help="Probability to zero a prompt during training")
@click.option(
    "--out-ckpt",
    "-o",
    type=click.Path(dir_okay=False, writable=True),
    required=True,
    help="Where to write the learned prompt checkpoint (.pt)",
)
def main(base_dir, crop_size, epochs, lr, n_prompts, prompt_dropout, out_ckpt):

    base_path = Path(base_dir)

    # ── Gather reference imagery and masks ───────────────────────────────────
    mask_dir = base_path / "Masks"
    veg_masks = get_masks(mask_dir, "Mask_*.shp")
    water_masks = get_masks(mask_dir, "WaterMask_*.shp")
    mask_date = infer_date(veg_masks + water_masks)

    img_paths = list((base_path / "SatelliteImagery").glob("*/*.tif"))
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
    merged_mask = np.zeros((*veg_mask.shape, 3), dtype=np.uint8)
    merged_mask[water_mask] = 1
    merged_mask[veg_mask] = 2
    merged_mask[sand_mask] = 3

    # ── Build crops along water line ─────────────────────────────────────────
    water_line = extract_linestring(water_mask, full_no_data)
    assert water_line is not None
    prompt_crops = generate_square_crops_along_line(water_line, crop_size, 0)

    full_prompt_img, full_prompt_no_data = merge_tifs(ref_imgs, out_shape, out_transform, crs)

    p_imgs, p_masks, p_nodata = create_per_day_crops(
        prompt_crops, full_prompt_img, full_prompt_no_data, merged_mask, crop_size
    )
    keep = [i for i, nd in enumerate(p_nodata) if ~np.all(nd)]
    p_imgs = [p_imgs[i] for i in keep]
    p_masks = [p_masks[i] for i in keep]
    p_nodata = [p_nodata[i] for i in keep]

    # ── Train prompt tensors ────────────────────────────────────────────────
    logger.info(f"Training {n_prompts} prompt tensor(s) on {len(p_imgs)} crops")
    model = load_model("BAAI/seggpt-vit-large")
    processor = load_processor("BAAI/seggpt-vit-large")
    prompt_px, prompt_msk = train_learnable_prompt(
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

    torch.save(
        {"prompt_pixel_values": prompt_px, "prompt_masks": prompt_msk},
        out_ckpt,
    )
    logger.info(f"✓ Saved learned prompt ➜ {out_ckpt}")


if __name__ == "__main__":
    main()
