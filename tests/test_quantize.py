import numpy as np

from stitchlate.color import rgb_to_lab
from stitchlate.quantize import kmeans, quantize_image, sample_pixels


def test_recovers_known_clusters():
    """Three tight synthetic clusters should be found at their true centers."""
    pts = np.vstack([
        np.random.default_rng(1).normal([0, 0, 0], 0.4, (200, 3)),
        np.random.default_rng(2).normal([50, 50, 50], 0.4, (200, 3)),
        np.random.default_rng(3).normal([0, 80, -40], 0.4, (200, 3)),
    ])
    centers, _ = kmeans(pts, 3, seed=7)
    # Cluster order is arbitrary, so sort before comparing.
    found = sorted(tuple(np.round(c)) for c in centers)
    expected = sorted([(0., 0., 0.), (50., 50., 50.), (0., 80., -40.)])
    for f, e in zip(found, expected):
        assert max(abs(np.array(f) - np.array(e))) < 2


def test_deterministic_with_fixed_seed():
    pts = np.random.default_rng(0).normal(0, 30, (500, 3))
    a, la = kmeans(pts, 6, seed=5)
    b, lb = kmeans(pts, 6, seed=5)
    assert np.allclose(a, b) and np.array_equal(la, lb)


def test_handles_k_greater_than_n():
    centers, labels = kmeans(np.array([[1., 2, 3], [4, 5, 6]]), 5)
    assert len(centers) == 2 and len(labels) == 2


def test_no_nan_centers():
    pts = np.random.default_rng(2).normal(0, 1, (50, 3))
    centers, _ = kmeans(pts, 20, seed=1)
    assert not np.isnan(centers).any()


def test_sample_pixels_respects_cap():
    pix = np.arange(300).reshape(100, 3)
    assert len(sample_pixels(pix, max_samples=40, seed=0)) == 40
    assert len(sample_pixels(pix, max_samples=500, seed=0)) == 100


def test_quantize_image_shape_and_range():
    img = np.random.default_rng(0).integers(0, 256, (60, 80, 3))
    centers, labels = quantize_image(rgb_to_lab(img), 12, seed=0)
    assert centers.shape == (12, 3)
    assert labels.shape == (60, 80)
    assert labels.min() >= 0 and labels.max() < 12

