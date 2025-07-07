import os
from dataclasses import dataclass
from pathlib import Path

from PIL.Image import Resampling

CLASSES = (
    "nodata",
    "water",
    "veg",
    "sand",
)


@dataclass
class BeachSegConfig:
    project: str = "beach_seg"
    seed: int = 42
    data: Path = Path("/Users/kyledorman/data/BorderField")
    model_training_root: Path = Path("/Users/kyledorman/data/results")
    classes: tuple[str, ...] = CLASSES
    devices: tuple[str, ...] = ("auto",)
    accelerator: str = "auto"
    # Use False b/c TorchMetrics allocates too much memory otherwise
    deterministic: bool = False
    num_viz_images: int = 9

    epochs: int = 1
    debug: bool = False
    world_size: int = 1
    grad_accum_steps: int = 1
    log_every_n_steps: int = 2
    precision: str = "16-mixed"
    workers: int = -1
    batch_size: int = 1

    checkpoint: str = "BAAI/seggpt-vit-large"

    monitor_metric: str = "f1"
    monitor_mode: str = "max"

    crop_size: int = 224
    inpt_size: int = 448
    n_prompts: int = 1
    resample: Resampling = Resampling.BICUBIC

    horizontal_flip: float = 0.5
    vertical_flip: float = 0.5
    hue: float = 0.1
    saturation: float = 0.1
    contrast: float = 0.1
    brightness: float = 0.1
    scale: tuple[float, float] = (0.4, 1.0)
    sharpness: float = 1.0
    sharpness_p: float = 0.2
    erasing_scale: tuple[float, float] = (0.02, 0.05)
    erasing_p: float = 0.1
    gauss_mean: float = 0.0
    gauss_std: float = 0.1
    gauss_p: float = 0.1
    channel_shift_limit: float = 0.01
    channel_shift_p: float = 0.2
    mosaic_p: float = 0.0
    jigsaw_grid: tuple[int, int] = (2, 2)
    jigsaw_p: float = 0.0

    lr: float = 1e-2
    base_lr_batch_size: int = 1
    warmup_epochs: int = 3
    init_lr: float = 5e-04
    min_lr: float = 5e-04
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    ema_alpha = 0.99


def num_workers(conf: BeachSegConfig) -> int:
    # number of CUDA devices
    nd = max(1, conf.world_size)
    per_gpu_count = cpu_count() // nd

    if conf.workers == -1:
        return per_gpu_count
    # number of workers
    nw = min([per_gpu_count, conf.workers])

    return nw


def cpu_count() -> int:
    cnt = os.cpu_count()
    if cnt is None:
        return 0
    return cnt
