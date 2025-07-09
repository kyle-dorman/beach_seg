import numpy as np
import torch
from shapely.geometry import LineString, MultiLineString, Point
from transformers import SegGptForImageSegmentation, SegGptImageProcessor

from src.config import BeachSegConfig


def load_model(config: BeachSegConfig) -> SegGptForImageSegmentation:
    model = SegGptForImageSegmentation.from_pretrained(config.checkpoint)
    for p in model.parameters():  # freeze the backbone
        p.requires_grad_(False)
    model = torch.compile(model, mode="default").eval()

    return model  # type: ignore


def load_processor(config: BeachSegConfig) -> SegGptImageProcessor:
    return SegGptImageProcessor.from_pretrained(config.checkpoint)


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
