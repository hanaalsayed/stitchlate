"""Measure how often naive RGB matching picks a different thread than LAB."""

import argparse

import numpy as np

from stitchlate.color import delta_e, rgb_to_lab
from stitchlate.palette import Palette


def nearest_rgb(pixels, palette_rgb):
    """Nearest palette entry by naive Euclidean distance in RGB."""
    d2 = ((pixels[:, None, :].astype(np.float64) - palette_rgb[None, :, :]) ** 2).sum(-1)
    return d2.argmin(axis=1)


def nearest_lab(pixels_lab, palette_lab):
    """Nearest palette entry by perceptual distance in LAB."""
    d2 = ((pixels_lab[:, None, :] - palette_lab[None, :, :]) ** 2).sum(-1)
    return d2.argmin(axis=1)


def sample_uniform(n, rng):
    """Random colors spread evenly through the RGB cube."""
    return rng.integers(0, 256, (n, 3))


def sample_image(path, n, rng):
    """Random pixels from a real image, which is NOT a uniform distribution."""
    from PIL import Image  # imported here so uniform mode has no Pillow dependency

    pixels = np.asarray(Image.open(path).convert("RGB")).reshape(-1, 3)
    if len(pixels) <= n:
        return pixels
    idx = rng.choice(len(pixels), size=n, replace=False)
    return pixels[idx]


def main():
    ap = argparse.ArgumentParser(
        description="Compare RGB vs LAB nearest-thread matching.")
    ap.add_argument("--image", metavar="PATH",
                    help="sample pixels from this image instead of uniformly")
    ap.add_argument("--samples", type=int, default=20000,
                    help="number of pixels to test (default: 20000)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pal = Palette.load()
    rng = np.random.default_rng(args.seed)

    if args.image:
        pixels = sample_image(args.image, args.samples, rng)
        source = f"image: {args.image}"
    else:
        pixels = sample_uniform(args.samples, rng)
        source = "uniform random RGB"

    pixels_lab = rgb_to_lab(pixels)
    pick_rgb = nearest_rgb(pixels, pal.rgb.astype(np.float64))
    pick_lab = nearest_lab(pixels_lab, pal.lab)

    disagree = pick_rgb != pick_lab
    err_rgb = delta_e(pixels_lab, pal.lab[pick_rgb])
    err_lab = delta_e(pixels_lab, pal.lab[pick_lab])

    print(f"source:              {source}")
    print(f"palette size:        {len(pal)} threads")
    print(f"pixels tested:       {len(pixels)}")
    print(f"seed:                {args.seed}")
    print(f"disagreement rate:   {disagree.mean() * 100:.1f}%")
    print(f"mean delta-E (RGB):  {err_rgb.mean():.2f}")
    print(f"mean delta-E (LAB):  {err_lab.mean():.2f}")
    improvement = (err_rgb.mean() - err_lab.mean()) / err_rgb.mean() * 100
    print(f"mean error reduction: {improvement:.1f}%")

    if disagree.any():
        penalty = (err_rgb - err_lab)[disagree]
        print(f"when they disagree, RGB is worse by {penalty.mean():.2f} delta-E on average")
        print(f"                    worst case:      {penalty.max():.2f} delta-E")

        worst = int(np.argmax(np.where(disagree, err_rgb - err_lab, -np.inf)))
        print("\nworst single disagreement:")
        print(f"  target pixel : RGB {tuple(int(v) for v in pixels[worst])}")
        print(f"  RGB picks    : {pal.codes[pick_rgb[worst]]:>6}  "
              f"RGB {tuple(int(v) for v in pal.rgb[pick_rgb[worst]])}  "
              f"(delta-E {err_rgb[worst]:.1f})")
        print(f"  LAB picks    : {pal.codes[pick_lab[worst]]:>6}  "
              f"RGB {tuple(int(v) for v in pal.rgb[pick_lab[worst]])}  "
              f"(delta-E {err_lab[worst]:.1f})")


if __name__ == "__main__":
    main()