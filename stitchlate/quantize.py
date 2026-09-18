"""Color quantization: reducing an image to a small set of thread colors.

A photo has thousands of colors and a pattern can only use about 25, so we
need to pick which ones. That's a clustering problem.

Median cut is the faster option, but it can only split the color space along
straight lines on each axis, so it does badly when the colors are grouped at
an angle. k-means follows the actual shape of the data, so I used that even
though it's slower.

This runs in LAB. Averaging RGB values doesn't give you the color that sits in
the middle of a group.
"""

import numpy as np


def kmeans(points, k, iters=30, seed=0):
    """Cluster points into k groups. Returns (centers, labels).

    points : (n, d) array -- here, LAB colors, so d == 3
    k      : number of clusters (thread colors)
    iters  : maximum passes; converges earlier if labels stop changing
    seed   : fixed by default so identical input gives identical output,
             which is what makes the tests and the CLI reproducible
    """
    points = np.asarray(points, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(points)

    if k >= n:
        # Asked for more colors than there are points, so each point is its own.
        return points.copy(), np.arange(n)

    # k-means++ startup. Each new center is picked with a higher chance the
    # further it is from the centers already chosen.
    centers = [points[rng.integers(n)]]
    for _ in range(k - 1):
        # Squared distance from each point to its closest chosen center.
        d2 = np.min(((points[:, None, :] - np.array(centers)[None, :, :]) ** 2).sum(-1),
                    axis=1)
        total = d2.sum()
        probs = d2 / total if total > 0 else np.full(n, 1 / n)
        centers.append(points[rng.choice(n, p=probs)])
    centers = np.array(centers, dtype=np.float64)

    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        # Assign step. Comparing (n,1,d) to (1,k,d) builds an (n,k) table of
        # distances, which is the part that uses the most memory. These are
        # squared distances.
        d2 = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        new_labels = d2.argmin(axis=1)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # Move each center to the average of its points.
        for j in range(k):
            m = labels == j
            if m.any():
                centers[j] = points[m].mean(axis=0)
            # If a cluster ended up empty, leave it where it is so it doesn't
            # turn into NaN. It might get points next round.

    return centers, labels


def sample_pixels(pixels, max_samples=20000, seed=0):
    """Pick a random subset of the pixels to cluster on.

    The assign step makes an (n, k) table, and a 4000x3000 photo has 12 million
    pixels, which is way too big. A few thousand pixels is already enough to
    show what colors an image uses, and adding more barely moves the centers.
    I measured about 7x faster on a 400x400 image with k=25.
    """
    pixels = np.asarray(pixels)
    if len(pixels) <= max_samples:
        return pixels
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pixels), size=max_samples, replace=False)
    return pixels[idx]


def quantize_image(lab_image, k, max_samples=20000, seed=0):
    """Reduce a LAB image to k colors. Returns (centers, labels_2d).

    centers   : (k, 3) LAB colors -- the representative palette
    labels_2d : (h, w) ints indexing into centers, one per stitch

    Clusters on a sample, then goes through every pixel once to label it.
    """
    lab_image = np.asarray(lab_image, dtype=np.float64)
    h, w, _ = lab_image.shape
    flat = lab_image.reshape(-1, 3)

    centers, _ = kmeans(sample_pixels(flat, max_samples, seed), k, seed=seed)

    # Done in chunks so it doesn't use too much memory.
    labels = np.empty(len(flat), dtype=int)
    chunk = 100_000
    for start in range(0, len(flat), chunk):
        block = flat[start:start + chunk]
        d2 = ((block[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        labels[start:start + chunk] = d2.argmin(axis=1)

    return centers, labels.reshape(h, w)