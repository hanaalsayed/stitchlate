import numpy as np
import pytest
from PIL import Image

from stitchlate.image_io import finished_size, load_grid


@pytest.fixture
def landscape(tmp_path):
    p = tmp_path / "landscape.png"
    Image.fromarray(
        np.random.default_rng(0).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    ).save(p)
    return p


def test_downsamples_to_requested_width(landscape):
    grid = load_grid(landscape, 100)
    assert grid.shape == (75, 100, 3)
    assert grid.dtype == np.uint8


def test_preserves_aspect_ratio(landscape):
    grid = load_grid(landscape, 80)
    assert abs((80 / grid.shape[0]) - (320 / 240)) < 0.05


def test_normalizes_grayscale_and_alpha_to_three_channels(tmp_path):
    for mode in ("L", "RGBA"):
        p = tmp_path / f"{mode}.png"
        Image.new(mode, (100, 80)).save(p)
        assert load_grid(p, 20).shape[2] == 3


def test_extreme_aspect_ratio_keeps_at_least_one_row(tmp_path):
    p = tmp_path / "pano.png"
    Image.new("RGB", (2000, 30)).save(p)
    assert load_grid(p, 50).shape[0] >= 1


def test_missing_file_raises_with_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="no image found"):
        load_grid(tmp_path / "absent.png", 100)


def test_rejects_zero_width(landscape):
    with pytest.raises(ValueError, match="at least 1"):
        load_grid(landscape, 0)


def test_finished_size_uses_aida_count():
    w, h = finished_size((70, 140), aida_count=14)
    assert (w, h) == (10.0, 5.0)