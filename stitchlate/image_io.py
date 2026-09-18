"""Loads an image and shrinks it down to a grid of stitches.

A pattern is a grid of stitches, so the photo gets shrunk to that size before
anything else happens.
"""
from pathlib import Path

import numpy as np
from PIL import Image

# Default Aida cloth count, for the finished size estimate.
DEFAULT_AIDA_COUNT = 14


def load_grid(path, stitch_width):
    """Load an image and shrink it to stitch_width stitches across.

    Gives back an (h, w, 3) array with one entry per stitch. The width is in
    stitches instead of pixels because that's what decides how big the
    finished piece is and how long it takes to sew. 
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no image found at {path}")
    if stitch_width < 1:
        raise ValueError(f"stitch_width must be at least 1, got {stitch_width}")

    # This handles grayscale, palette images, and transparency.
    img = Image.open(path).convert("RGB")

    w, h = img.size
    # Using round instead of int, keeps a really wide image
    # from ending up with zero rows.
    stitch_height = max(1, round(h * stitch_width / w))

    # LANCZOS blends the pixels it's shrinking together.
    img = img.resize((stitch_width, stitch_height), Image.LANCZOS)

    return np.asarray(img, dtype=np.uint8)


def finished_size(grid_shape, aida_count=DEFAULT_AIDA_COUNT):
    """How big the finished piece will be in inches, as (width, height)."""
    h, w = grid_shape[:2]
    return (w / aida_count, h / aida_count)