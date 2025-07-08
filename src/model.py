import logging
from collections import deque

import torch
import torch.nn.functional as F
import torchvision
from lightning.pytorch import LightningModule
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.utils.data import _utils
from torchmetrics import MetricCollection
from torchmetrics.classification import (  # MulticlassPrecision,; MulticlassRecall,
    MulticlassF1Score,
)
from torchvision.utils import draw_segmentation_masks

from src.config import BeachSegConfig
from src.data import BeachSegDataModule, build_palette, generate_random_rgb_palette, torch_apply_mask_rgb
from src.ml_util import load_model

logger = logging.getLogger(__name__)


CLASS_COLORS = ["limegreen", "hotpink", "lightcyan", "yellow"]


def plot_masks(
    images: torch.Tensor,
    masks: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Plot a sample from the dataset."""
    return torch.stack(
        [
            draw_segmentation_masks(image=img, masks=mask, colors=CLASS_COLORS, alpha=alpha)  # type: ignore
            for img, mask in zip(images, masks)
        ],
        dim=0,
    )


class PromptModel(LightningModule):
    def __init__(self, conf: BeachSegConfig, datamodule: BeachSegDataModule):
        logger.info("🔧 Initialising PromptModel")
        self.conf = conf
        self.num_classes = len(conf.classes)
        self.nodata_idx = 0

        super().__init__()
        self.save_hyperparameters(conf)

        self.model = load_model(conf)
        logger.info("✔️  SegGPT model loaded")

        logger.info("Creating metrics")

        loss_kwargs = {}
        loss_kwargs["ignore_index"] = self.nodata_idx

        metrics = MetricCollection(
            {
                "f1": MulticlassF1Score(self.num_classes, validate_args=conf.debug, **loss_kwargs),
                # "precision": MulticlassPrecision(self.num_classes, validate_args=conf.debug, **loss_kwargs),
                # "recall": MulticlassRecall(num_classes=self.num_classes, validate_args=conf.debug, **loss_kwargs),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        logger.info("✔️  Metrics configured: %s", list(self.train_metrics.keys()))
        datamodule.setup("train")
        logger.info(
            "✔️  DataModule setup: %d training samples, %d prompt crops",
            len(datamodule.train_dataset),
            len(datamodule.prompt_imgs),
        )

        prompt_params, self.ema_buffers = [], []
        self.prompt_batch: dict[str, torch.Tensor] = _utils.collate.default_collate(
            datamodule.prompt_imgs
        )  # type: ignore

        for idx in range(conf.n_prompts):
            init_px = self.prompt_batch["image"][idx]
            p = torch.nn.Parameter(init_px, requires_grad=True)
            prompt_params.append(p)
            self.ema_buffers.append(p.detach().clone())
        logger.info("✔️  Created %d trainable prompt tensors", len(prompt_params))

        self.prompt_params_list = torch.nn.ParameterList(prompt_params)
        self.prompt_batch["image"] = prompt_params  # type: ignore
        self.ema = deque(self.ema_buffers, maxlen=conf.n_prompts)

        self.val_batch_pred = []
        self.g = torch.Generator(device=self.model.device)  # type: ignore
        self.g.manual_seed(conf.seed)
        self.train_aug = datamodule.train_aug
        self.aug = datamodule.aug
        self.normalize = datamodule.normalize
        self.denormalize = datamodule.denormalize
        logger.info(
            "✔️  Augmentations ready (train=%d, infer=%d)",
            len(self.train_aug),
            len(self.aug),
        )

        logger.info("✅ PromptModel initialisation complete")

    def forward(self, x):
        assert False, "Add prompt"
        return self.model(x)  # type: ignore

    def one_hot(self, x: torch.Tensor, num_classes: int) -> torch.Tensor:
        B, H, W = x.shape
        return torch.full((B, num_classes, H, W), 0, device=x.device, dtype=x.dtype).scatter_(
            1, x.long().unsqueeze(1), 1
        )

    def process_pred_masks(self, in_pred_masks: torch.Tensor, batch_palette_norm: torch.Tensor) -> torch.Tensor:
        # batch_size x num_channels x 2*height x width
        B, C, H2, W = in_pred_masks.shape
        H = H2 // 2

        # Predicted mask and prompt are concatenated in the height dimension
        # batch_size x num_channels x height x width
        masks = in_pred_masks[:, :, H:, :]

        pred_masks = []
        for idx, mask in enumerate(masks):
            palette = batch_palette_norm[idx]
            channels, height, width = mask.shape
            dist = mask.permute(1, 2, 0).view(height, width, 1, channels)
            dist = dist - palette.view(1, 1, 4, 3)
            dist = torch.pow(dist, 2)
            dist = torch.sum(dist, dim=-1)
            pred = dist.argmin(dim=-1)
            pred_masks.append(pred)

        return torch.stack(pred_masks, dim=0).to(self.device)

    def prepare_prompt(
        self, idx: int, batch_palette: torch.Tensor, train: bool
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        start = idx
        end = start + 1
        prompt_batch = {k: self.prompt_batch[k][start:end] for k in self.prompt_batch.keys()}
        # convert list of Parameters into a single Tensor for augmentation
        prompt_batch["image"] = torch.stack(prompt_batch["image"], dim=0).to(self.device)  # type: ignore
        if train:
            prompt_batch = self.train_aug(prompt_batch)
        else:
            prompt_batch = self.aug(prompt_batch)
        prompt_color_mask = torch_apply_mask_rgb(batch_palette, prompt_batch["mask"])
        prompt_color_mask_norm = self.normalize(prompt_color_mask).to(self.device)

        return prompt_batch, prompt_color_mask_norm

    def create_palette(self, batch_size: int, train: bool) -> tuple[torch.Tensor, torch.Tensor]:
        palette = torch.Tensor(build_palette(self.num_classes - 1))
        if train:
            batch_palette = generate_random_rgb_palette(self.num_classes, batch_size, self.device)
        else:
            batch_palette = torch.stack([palette for _ in range(batch_size)])
        palettes = []
        for palelle in batch_palette:
            palette_norm = (
                self.normalize(palelle.view((self.num_classes, 3, 1, 1)).to(torch.float32) / 255)
                .squeeze(-1)
                .squeeze(-1)
            )
            palettes.append(palette_norm)
        batch_palette_norm = torch.stack(palettes)

        return batch_palette, batch_palette_norm

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        B, _, _, _ = batch["mask"].shape
        batch_palette, batch_palette_norm = self.create_palette(B, train=True)

        # Colorize mask
        color_mask = torch_apply_mask_rgb(batch_palette, batch["mask"])
        color_mask_norm = self.normalize(color_mask).to(self.device)

        # choose prompt index
        # TODO: Should pick once per batch item. Assumes batch size 1.
        prompt_idx: int = torch.randint(0, self.conf.n_prompts, (), generator=self.g).item()  # type: ignore
        prompt_batch, prompt_masks = self.prepare_prompt(prompt_idx, batch_palette, train=True)

        out = self.model(
            pixel_values=batch["image"],
            labels=color_mask_norm,
            prompt_pixel_values=prompt_batch["image"],
            prompt_masks=prompt_masks,
            embedding_type="semantic",
        )
        pred_masks = self.process_pred_masks(out.pred_masks, batch_palette_norm)

        # self.ema[pi].mul_(self.conf.ema_alpha).add_(px.data, alpha=1 - self.conf.ema_alpha)
        loss = out.loss
        self.train_metrics.update(pred_masks, batch["mask"].squeeze(1).to(self.device))

        self.log_dict({"train/loss": loss}, prog_bar=True, sync_dist=True, batch_size=len(batch["image"]))
        # # Assume batch size 1 here
        # self.train_batch_pred.append(
        #     (
        #         batch["image"].squeeze(0).detach(),
        #         batch["mask"].squeeze(0).squeeze(0).detach(),
        #         pred_masks[0].detach(),
        #         prompt_batch["image"].squeeze(0).detach(),
        #     )
        # )

        return loss  # type: ignore

    def on_train_epoch_end(self) -> None:
        self.log_dict(self.train_metrics.compute(), sync_dist=True)
        self.train_metrics.reset()

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0):
        B, _, _, _ = batch["mask"].shape
        batch_palette, batch_palette_norm = self.create_palette(B, train=True)

        # Colorize mask
        color_mask = torch_apply_mask_rgb(batch_palette, batch["mask"])
        color_mask_norm = self.normalize(color_mask).to(self.device)

        # choose prompt index
        # TODO: Should pick once per batch item. Assumes batch size 1.
        prompt_idx: int = torch.randint(0, self.conf.n_prompts, (), generator=self.g).item()  # type: ignore
        prompt_batch, prompt_masks = self.prepare_prompt(prompt_idx, batch_palette, train=False)

        out = self.model(
            pixel_values=batch["image"],
            labels=color_mask_norm,
            prompt_pixel_values=prompt_batch["image"],
            prompt_masks=prompt_masks,
            embedding_type="semantic",
        )
        pred_masks = self.process_pred_masks(out.pred_masks, batch_palette_norm)

        # self.ema[pi].mul_(self.conf.ema_alpha).add_(px.data, alpha=1 - self.conf.ema_alpha)
        loss = out.loss
        self.val_metrics.update(pred_masks, batch["mask"].squeeze(1).to(self.device))

        self.log_dict({"val/loss": loss}, prog_bar=True, batch_size=len(batch["image"]))
        # # Assume batch size 1 here
        self.val_batch_pred.append(
            (
                batch["image"].squeeze(0).detach(),
                batch["mask"].squeeze(0).squeeze(0).detach(),
                pred_masks[0].detach(),
            )
        )

        return loss  # type: ignore

    def on_validation_epoch_end(self):
        if self._should_plot_image():
            grid = self._plot_examples()  # type: ignore
            self.logger.experiment.add_image("val_images", grid, self.current_epoch)  # type: ignore
            # Resize to (B, C, viz_size, viz_size)
            prompt_imgs = F.interpolate(
                self.prompt_batch["image"],
                size=(self.conf.viz_size, self.conf.viz_size),
                mode="bilinear",
                align_corners=False,
            )
            grid = torchvision.utils.make_grid(prompt_imgs, nrow=1)
            self.logger.experiment.add_image("prompt_images", grid, self.current_epoch)  # type: ignore

        self.val_batch_pred = []
        self.log_dict(self.val_metrics.compute(), sync_dist=True)
        self.val_metrics.reset()

    def _should_plot_image(self) -> bool:
        return (
            self.conf.num_viz_images > 0
            and self.trainer is not None
            and self.logger.experiment is not None  # type: ignore
        )

    def _plot_examples(self) -> torch.Tensor:
        with torch.no_grad():
            batch_size = len(self.val_batch_pred)
            num_images = min(batch_size, self.conf.num_viz_images)

            x = torch.stack([b[0] for b in self.val_batch_pred[:num_images]])
            target = torch.stack([b[1] for b in self.val_batch_pred[:num_images]])
            pred = torch.stack([b[2] for b in self.val_batch_pred[:num_images]])

            # Mark nodata as nodata if ignoring this class
            nodata = target == self.nodata_idx
            pred[nodata] = self.nodata_idx
            one_hot_pred = self.one_hot(pred, num_classes=len(self.conf.classes)).bool()
            one_hot_target = self.one_hot(target.long(), num_classes=len(self.conf.classes)).bool()

            viz_image = self.make_image_visulizable(x)  # type: ignore
            target_images = plot_masks(viz_image, one_hot_target)
            pred_images = plot_masks(viz_image, one_hot_pred)

            # Resize to (B, C, viz_size, viz_size)
            viz_image = F.interpolate(
                viz_image, size=(self.conf.viz_size, self.conf.viz_size), mode="bilinear", align_corners=False
            )
            target_images = F.interpolate(
                target_images, size=(self.conf.viz_size, self.conf.viz_size), mode="bilinear", align_corners=False
            )
            pred_images = F.interpolate(
                pred_images, size=(self.conf.viz_size, self.conf.viz_size), mode="bilinear", align_corners=False
            )

            # Stack tensors along a new axis, then interleave and reshape
            interleaved = torch.stack(
                (
                    viz_image,
                    target_images,
                    pred_images,
                ),
                dim=1,
            )  # Shape: (B, 3, C, W, H)
            interleaved = interleaved.view(-1, *target_images.shape[1:])  # Shape: (B * 3, C, W, H)
            return torchvision.utils.make_grid(interleaved, nrow=3)

    def configure_optimizers(self):
        global_batch_size = self.conf.batch_size * self.conf.world_size * self.conf.grad_accum_steps
        batch_ratio = (global_batch_size / self.conf.base_lr_batch_size) ** 0.5
        num_gpus = 0  # calc_num_gpus(self.conf)

        logger.info(f"batch_ratio is {batch_ratio:.3f}. Can see {num_gpus} gpus")

        lr = self.conf.lr * batch_ratio
        init_lr = self.conf.init_lr * batch_ratio
        min_lr = self.conf.min_lr * batch_ratio
        warmup_epochs = self.conf.warmup_epochs

        if self.conf.optimizer == "adamw":
            optimizer = optim.AdamW(self.parameters(), lr=lr)
        else:
            raise RuntimeError(f"Unexpected optimizer {self.conf.optimizer}")

        # Learning rate scheduler; adjusts the learning rate during training
        def linear_warmup(epoch):
            pct = epoch / warmup_epochs
            diff = lr - init_lr
            scale = diff * pct
            return (scale + init_lr) / lr

        schedulers = []
        milestones = []
        if warmup_epochs:
            schedulers.append(LambdaLR(optimizer, [linear_warmup]))
            milestones.append(warmup_epochs)

        if self.conf.scheduler == "cosine":
            schedulers.append(CosineAnnealingLR(optimizer, self.conf.epochs, min_lr))
        else:
            raise RuntimeError(f"Unexpected scheduler {self.conf.scheduler}")

        config = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": SequentialLR(optimizer, schedulers, milestones),
                "interval": "epoch",
                "frequency": 1,
            },
        }
        return config

    def make_image_visulizable(
        self,
        batch_image: torch.Tensor,
    ) -> torch.Tensor:
        # Denorm
        batch_image = self.denormalize(batch_image)
        batch_image = batch_image.clip(0, 1)

        return batch_image
