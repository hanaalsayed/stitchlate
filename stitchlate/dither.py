"""Floyd-Steinberg dithering.

Dithering keeps track of how far off each choice was and pushes that
error onto the stitches nearby that haven't been done yet. The mistakes end up
cancelling out over a small area, so the average color stays right.
"""

import numpy as np


def floyd_steinberg(lab_image, palette_lab, serpentine=True):
    """Match every stitch to a palette color, spreading the error as it goes.

    lab_image   : (h, w, 3) LAB colors
    palette_lab : (k, 3) LAB colors to choose from
    serpentine  : go right on even rows and left on odd ones

    Gives back an (h, w) array of palette indexes.
    """
    work = lab_image.astype(np.float64).copy()
    h, w, _ = work.shape
    out = np.zeros((h, w), dtype=int)

    for y in range(h):
        # Alternate direction to spread error more evenly.
        # This is called serpentine scanning.
        rev = serpentine and (y % 2 == 1)
        xs = reversed(range(w)) if rev else range(w)

        for x in xs:
            old = work[y, x].copy()
            idx = int(((palette_lab - old) ** 2).sum(-1).argmin())
            out[y, x] = idx
            err = old - palette_lab[idx]

            # Standard Floyd-Steinberg weights. The error gets split between
            # the next stitch over and the three below it:
            #
            #          X    7/16
            #   3/16  5/16  1/16
            nbrs = ([(0, -1, 7/16), (1, 1, 3/16), (1, 0, 5/16), (1, -1, 1/16)] if rev
                    else [(0, 1, 7/16), (1, -1, 3/16), (1, 0, 5/16), (1, 1, 1/16)])

            for dy, dx, wgt in nbrs:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    work[ny, nx] += err * wgt

    return out