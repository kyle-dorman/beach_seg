import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import tqdm
from dotenv import find_dotenv, load_dotenv
from lightning.pytorch import seed_everything
from omegaconf import OmegaConf

from src.config import BeachSegConfig
from src.data import BeachSegDataModule, create_dataset
from src.model import PromptModel
from src.util.img_util import overlay_prediction, write_mask_tif
from src.util.util import setup_logger

logger = logging.getLogger(__name__)


@dataclass
class PredictionConfig:
    data: Path = Path("/Users/kyledorman/data/BorderField")

    model_training_root: Path | None = None
    debug: bool = False
    workers: int = 0
    batch_size: int = 1

    config_path: Path | None = None


def num_workers(conf: PredictionConfig) -> int:
    # number of CUDA devices
    per_gpu_count = cpu_count()

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


class Accumulator:
    def __init__(
        self,
        out_shape: tuple[int, int],
        save_dir: Path,
        out_transform: Any,
        crs: Any,
        classes: tuple[str, ...],
    ):
        self.out_shape = out_shape
        self.num_classes = len(classes)
        self.out_transform = out_transform
        self.crs = crs
        self.classes = classes

        self.img_dir = save_dir / "images"
        self.img_dir.mkdir(exist_ok=True)
        self.mask_dir = save_dir / "masks"
        self.mask_dir.mkdir(exist_ok=True)
        self.tif_dir = save_dir / "tif"
        self.tif_dir.mkdir(exist_ok=True)

        self.current_label = None
        self.current_img = None
        self.current_pred_counter = None
        self.current_date = None

    def __enter__(self):
        assert self.current_label is None
        assert self.current_img is None
        assert self.current_pred_counter is None
        assert self.current_date is None

        return self

    def __exit__(self, a, b, c):
        self.save_current()

    def save_current(self):
        # --- Build RGB overlay of predicted mask using PIL ---------------------------
        assert self.current_label is not None
        assert self.current_img is not None
        assert self.current_pred_counter is not None
        assert self.current_date is not None

        pred = np.argmax(self.current_pred_counter, axis=2)  # (H, W)

        blended = overlay_prediction(self.current_img, pred, self.classes)

        # Save
        save_path = self.img_dir / f"{self.current_date}.png"
        blended.save(save_path)

        cv2.imwrite(str(self.mask_dir / f"{self.current_date}.png"), pred)

        # GeoTIFF export
        tif_path = self.tif_dir / f"{self.current_date}.tif"
        write_mask_tif(pred.astype(np.uint8), self.out_transform, self.crs, tif_path)

    def initialize_current(self, date: str):
        self.current_date = date
        self.current_img = np.zeros((*self.out_shape, 3), dtype=np.uint8)
        self.current_pred_counter = np.zeros((*self.out_shape, self.num_classes), dtype=np.uint8)
        self.current_label = np.zeros(self.out_shape, dtype=np.uint8)

    def update(
        self,
        date: str,
        crop: tuple[int, int, int, int],
        one_hot_pred: np.ndarray,
        img_crop: np.ndarray,
        label_crop: np.ndarray | None,
    ):
        if date != self.current_date:
            if self.current_pred_counter is not None:
                self.save_current()
            self.initialize_current(date)

        assert self.current_label is not None
        assert self.current_img is not None
        assert self.current_pred_counter is not None
        assert self.current_date is not None

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

        self.current_img[dy0:dy1, dx0:dx1] = img_crop[sy0:sy1, sx0:sx1]
        self.current_pred_counter[dy0:dy1, dx0:dx1] += one_hot_pred[sy0:sy1, sx0:sx1]
        if label_crop is not None:
            self.current_label[dy0:dy1, dx0:dx1] = label_crop[sy0:sy1, sx0:sx1]


def main():
    """
    predicts model
    """
    base_conf = OmegaConf.structured(PredictionConfig)
    cli_conf = OmegaConf.from_cli()
    # Hack around a single value tuple
    if "devices" in cli_conf and isinstance(cli_conf["devices"], str | int):
        cli_conf["devices"] = [cli_conf["devices"]]
    pred_conf: PredictionConfig = OmegaConf.merge(base_conf, cli_conf)  # type: ignore

    # Load config from checkpoint directory
    if pred_conf.config_path is not None:
        conf_path = pred_conf.config_path
        conf: BeachSegConfig = OmegaConf.merge(
            OmegaConf.structured(BeachSegConfig), OmegaConf.load(conf_path)
        )  # type: ignore
    else:
        conf: BeachSegConfig = OmegaConf.structured(BeachSegConfig)

    conf.data = pred_conf.data
    conf.batch_size = pred_conf.batch_size
    conf.debug = pred_conf.debug
    conf.workers = pred_conf.workers
    if pred_conf.model_training_root is not None:
        conf.model_training_root = pred_conf.model_training_root

    # Save prediciton results
    model_predict_root = Path(conf.model_training_root) / conf.project / "predict"
    model_predict_root.mkdir(exist_ok=True, parents=True)
    not_dot_dirs = filter(lambda p: not p.name.startswith("."), model_predict_root.iterdir())
    runs = [int(p.name) for p in not_dot_dirs]
    last_id = max(runs + [-1])
    predict_dir = model_predict_root / str(last_id + 1).zfill(5)
    predict_dir.mkdir(parents=True)

    # Setup logger
    setup_logger(logger, predict_dir)

    logger.info(f"Saving results to {predict_dir}")

    seed_everything(conf.seed, workers=True)

    # Load model and data
    device = "cpu"
    model = PromptModel(conf)
    model.to(device)
    datamodule = BeachSegDataModule(config=conf)
    model.post_init(datamodule)
    num_classes = len(conf.classes)

    if pred_conf.config_path is not None:
        logger.info("Reading prompt_batch.pt")
        prompt_batch = torch.load(conf_path.parent / "prompt_batch.pt", map_location="cpu")
        model.prompt_batch = prompt_batch
    else:
        # Use unaltered images as prompts
        datamodule.setup("train")
        model.create_trainable_params(datamodule)

    logger.info("Loading predict dataset")
    datamodule.setup("predict")

    train_dataset, out_shape, out_transform, crs = create_dataset(conf, train=True)

    dataloader = datamodule.predict_dataloader()

    num_samples = len(dataloader.dataset)  # type: ignore
    crops = train_dataset.crops

    with torch.no_grad():
        with Accumulator(out_shape, predict_dir, out_transform, crs, conf.classes) as acc:
            for batch in tqdm.tqdm(iter(dataloader), total=1 + num_samples // conf.batch_size, desc="Model Inference"):
                if batch["nodata"].all():
                    continue
                crop_idx = batch["crop_idx"][0]
                date = batch["date"][0]
                batch = datamodule.aug(batch)
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

                pred_mask = model(batch)
                pred_mask = pred_mask.squeeze(0).detach().cpu().numpy()
                crop_img = (
                    (
                        model.make_image_visulizable(batch["image"])
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                        .transpose((1, 2, 0))
                        .copy()
                    )
                    * 255
                ).astype(np.uint8)

                # Resize to original crop size
                crop_img = cv2.resize(crop_img, (conf.crop_size, conf.crop_size), interpolation=cv2.INTER_CUBIC)
                pred_mask = cv2.resize(pred_mask, (conf.crop_size, conf.crop_size), interpolation=cv2.INTER_NEAREST)
                one_hot = np.eye(num_classes, dtype=np.uint8)[pred_mask]

                acc.update(date, crops[crop_idx], one_hot, crop_img, None)

    logger.info("Done!")


if __name__ == "__main__":
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
