"""Turns a grid of stitches into images you can look at."""

import numpy as np
from PIL import Image


def render_preview(labels, palette, cell=8):
    """Draw the stitch grid as solid blocks of thread color.

    labels : (h, w) numbers that index into the palette
    cell   : how many pixels wide each stitch is drawn

    This is the image that shows what the finished piece will look like.
    """
    labels = np.asarray(labels)
    rgb = palette.rgb[labels]
    # Repeat each stitch into a cell x cell block of pixels.
    blocks = np.repeat(np.repeat(rgb, cell, axis=0), cell, axis=1)
    return Image.fromarray(blocks.astype(np.uint8))


def thread_usage(labels, palette):
    """Count how many stitches use each thread, most used first."""
    idx, counts = np.unique(np.asarray(labels), return_counts=True)
    rows = [(palette.codes[i], palette.names[i],
             tuple(int(v) for v in palette.rgb[i]), int(c))
            for i, c in zip(idx, counts)]
    rows.sort(key=lambda r: -r[3])
    return rows