import numpy as np

from stitchlate.color import rgb_to_lab
from stitchlate.dither import floyd_steinberg


def test_error_is_conserved():
    """A gray ramp dithered to black and white keeps its average brightness."""
    palette = rgb_to_lab(np.array([[0, 0, 0], [255, 255, 255]]))
    ramp = np.stack([np.tile(np.linspace(0, 255, 64), (32, 1))] * 3, axis=-1)
    lab = rgb_to_lab(ramp)

    idx = floyd_steinberg(lab, palette)
    assert abs(lab[..., 0].mean() - palette[idx][..., 0].mean()) < 3


def test_only_uses_palette_indexes():
    palette = rgb_to_lab(np.array([[0, 0, 0], [255, 255, 255]]))
    lab = rgb_to_lab(np.random.default_rng(0).integers(0, 256, (20, 20, 3)))
    idx = floyd_steinberg(lab, palette)
    assert set(np.unique(idx)).issubset({0, 1})


def test_shape_matches_input():
    palette = rgb_to_lab(np.array([[0, 0, 0], [128, 128, 128], [255, 255, 255]]))
    lab = rgb_to_lab(np.random.default_rng(1).integers(0, 256, (13, 7, 3)))
    assert floyd_steinberg(lab, palette).shape == (13, 7)


def test_does_not_modify_input():
    palette = rgb_to_lab(np.array([[0, 0, 0], [255, 255, 255]]))
    lab = rgb_to_lab(np.random.default_rng(2).integers(0, 256, (10, 10, 3)))
    before = lab.copy()
    floyd_steinberg(lab, palette)
    assert np.array_equal(lab, before)


def test_solid_color_needs_no_dithering():
    """An image already made of one palette color comes back unchanged."""
    palette = rgb_to_lab(np.array([[10, 20, 30], [200, 100, 50]]))
    lab = np.broadcast_to(palette[1], (8, 8, 3)).copy()
    assert (floyd_steinberg(lab, palette) == 1).all()