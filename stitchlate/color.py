import numpy as np

# sRGB to XYZ conversion matrix and D65 white point
MATRIX = np.array([[0.4124564, 0.3575761, 0.1804375],
               [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]])
WHITE = np.array([0.95047, 1.00000, 1.08883])

# Convert sRGB to linear RGB
def srgb_to_linear(c):
    # Convert sRGB values (0-255) to linear RGB values (0-1)
    c = np.asarray(c, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def rgb_to_lab(rgb):
    rgb = np.asarray(rgb, dtype=np.float64)
    # Convert RGB to LAB color space
    shape = rgb.shape
    reshape = rgb.reshape(-1, 3)
    xyz = (srgb_to_linear(reshape) @ MATRIX.T) / WHITE
    # Convert XYZ to LAB
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    l = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    return np.stack((l, a, b), axis=-1).reshape(shape)

def diff(lab1, lab2):
    # Calculate the Euclidean distance between two LAB colors
    return np.sqrt(np.sum((np.asarray(lab1, float) - np.asarray(lab2, float)) ** 2, 
    axis=-1))