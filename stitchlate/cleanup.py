"""Getting rid of stray stitches.

Dithering leaves single stitches of one color scattered all over.
"""

import numpy as np


def count_isolated(labels):
    """Count stitches whose color doesn't show up in any of their 4 neighbors."""
    return int(_isolated_mask(np.asarray(labels)).sum())


def _isolated_mask(labels):
    """True wherever a stitch has no neighbor sharing its color."""
    h, w = labels.shape
    isolated = np.ones((h, w), dtype=bool)

    # Compare the whole grid against itself shifted one step in each direction.
    isolated[:-1, :] &= labels[:-1, :] != labels[1:, :]   # neighbor below
    isolated[1:, :] &= labels[1:, :] != labels[:-1, :]    # neighbor above
    isolated[:, :-1] &= labels[:, :-1] != labels[:, 1:]   # neighbor right
    isolated[:, 1:] &= labels[:, 1:] != labels[:, :-1]    # neighbor left

    return isolated


def reduce_confetti(labels, passes=1):
    """Replace isolated stitches with whatever color surrounds them.

    labels : (h, w) palette indexes
    passes : how many times to repeat.

    Gives back (cleaned_labels, number_changed).
    """
    labels = np.asarray(labels).copy()
    h, w = labels.shape
    total_changed = 0

    for _ in range(passes):
        isolated = _isolated_mask(labels)
        if not isolated.any():
            break

        # Work off a copy so stitches fixed earlier don't change 
        # what the later ones see.
        source = labels.copy()
        changed = 0

        for y, x in zip(*np.where(isolated)):
            nbrs = [source[y + dy, x + dx]
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
                    if 0 <= y + dy < h and 0 <= x + dx < w]
            if not nbrs:
                continue
            # Take whichever neighbor color shows up most.
            vals, counts = np.unique(nbrs, return_counts=True)
            labels[y, x] = int(vals[counts.argmax()])
            changed += 1

        total_changed += changed
        if changed == 0:
            break

    return labels, total_changed