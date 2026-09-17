import numpy as np
from stitchlate.palette import Palette


def test_exact_colors_map_to_themselves():
    p = Palette.load()
    assert np.array_equal(p.nearest(p.lab), np.arange(len(p)))


def test_loads_expected_fields():
    p = Palette.load()
    assert len(p.codes) == len(p.names) == len(p.rgb) == len(p.lab)
    assert p.rgb.shape[1] == 3 and p.lab.shape[1] == 3


def test_nearest_preserves_shape():
    p = Palette.load()
    img_lab = np.random.default_rng(0).normal(50, 20, (6, 4, 3))
    assert p.nearest(img_lab).shape == (6, 4)


def test_subset_dedupes_and_preserves_order():
    p = Palette.load()
    s = p.subset([5, 2, 5, 9])
    assert s.codes == [p.codes[5], p.codes[2], p.codes[9]]
