import numpy as np
import torch
from shapely.geometry import LineString, MultiLineString, Point
from transformers import SegGptForImageSegmentation, SegGptImageProcessor


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


def load_model() -> tuple[SegGptForImageSegmentation, SegGptImageProcessor]:
    checkpoint = "BAAI/seggpt-vit-large"
    image_processor = SegGptImageProcessor.from_pretrained(checkpoint)
    model = SegGptForImageSegmentation.from_pretrained(checkpoint)
    for p in model.parameters():  # freeze the backbone
        p.requires_grad_(False)
    model = torch.compile(model, mode="default")

    return model, image_processor  # type: ignore
