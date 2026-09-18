import numpy as np
from stitchlate.color import rgb_to_lab, delta_e

def test_white():
    lab = rgb_to_lab([255, 255, 255])
    assert abs(lab[0] - 100) < 1e-3
    assert abs(lab[1]) < 1e-3 and abs(lab[2]) < 1e-3

def test_black():
    assert abs(rgb_to_lab([0, 0, 0])[0]) < 1e-3

def test_red_reference():
    # Published sRGB reference value for pure red
    lab = rgb_to_lab([255, 0, 0])
    assert np.allclose(lab, [53.2408, 80.0925, 67.2032], atol=0.01)

def test_identical_colors_have_zero_distance():
    assert delta_e(rgb_to_lab([12, 200, 9]), rgb_to_lab([12, 200, 9])) == 0

def test_vectorized_over_image():
    img = np.random.randint(0, 256, (7, 5, 3))
    assert rgb_to_lab(img).shape == (7, 5, 3)