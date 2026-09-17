"""Color quantization: reducing an image to a small set of thread colors.

A cross-stitch pattern can only use a handful of thread colors, so the first
real decision is *which* colors. This is a clustering problem: find the k
colors that best represent the thousands present in the photo.

Two standard approaches:

  median cut -- recursively split the color space along its longest axis.
                Fast and deterministic, but the splits are axis-aligned, so
                it does poorly when colors cluster diagonally.
  k-means    -- iteratively find k centers minimizing within-cluster
                distance. Slower, but adapts to the actual shape of the
                data. Chosen here for that reason.

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
        return points.copy(), np.arange(n)

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
        d2 = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        new_labels = d2.argmin(axis=1)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        for j in range(k):
            m = labels == j
            if m.any():
                centers[j] = points[m].mean(axis=0)

    return centers, labels


def sample_pixels(pixels, max_samples=20000, seed=0):
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
    """
    lab_image = np.asarray(lab_image, dtype=np.float64)
    h, w, _ = lab_image.shape
    flat = lab_image.reshape(-1, 3)

    centers, _ = kmeans(sample_pixels(flat, max_samples, seed), k, seed=seed)

    labels = np.empty(len(flat), dtype=int)
    chunk = 100_000
    for start in range(0, len(flat), chunk):
        block = flat[start:start + chunk]
        d2 = ((block[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        labels[start:start + chunk] = d2.argmin(axis=1)

    return centers, labels.reshape(h, w)