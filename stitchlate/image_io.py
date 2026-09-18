""" """
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_AIDA_COUNT = 14


def load_grid(path, stitch_width):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no image found at {path}")
    if stitch_width < 1:
        raise ValueError(f"stitch_width must be at least 1, got {stitch_width}")

    img = Image.open(path).convert("RGB")

    w, h = img.size
    stitch_height = max(1, round(h * stitch_width / w))

    img = img.resize((stitch_width, stitch_height), Image.LANCZOS)

    return np.asarray(img, dtype=np.uint8)


def finished_size(grid_shape, aida_count=DEFAULT_AIDA_COUNT):
    h, w = grid_shape[:2]
    return (w / aida_count, h / aida_count)