"""Check whether dithering actually helps at cross-stitch resolutions.

Dithering is the standard way to reduce an image to a small palette, so I
expected it to help. It didn't. This script is how I worked out why.

Two error metrics, and the difference matters:

  per-pixel     -- how far each single stitch is from the original color
  area-average  -- the same thing after blurring 3x3, which is closer to how
                   an eye actually reads a stitched piece
"""

import argparse

import numpy as np

from stitchlate.cleanup import count_isolated, reduce_confetti
from stitchlate.color import delta_e, rgb_to_lab
from stitchlate.dither import floyd_steinberg
from stitchlate.image_io import load_grid
from stitchlate.quantize import quantize_image


def blur(a, k=3):
    """Average each value with its k x k neighbors.
    """
    out = np.zeros_like(a, dtype=np.float64)
    for dy in range(-(k // 2), k // 2 + 1):
        for dx in range(-(k // 2), k // 2 + 1):
            out += np.roll(np.roll(a, dy, 0), dx, 1)
    return out / (k * k)


def errors(lab, palette, idx):
    """Per-pixel and area-average color error for one labelling."""
    rendered = palette[idx]
    return (float(delta_e(lab, rendered).mean()),
            float(delta_e(blur(lab), blur(rendered)).mean()))


def nearest(lab, palette):
    """Closest palette color for every pixel, with no error diffusion."""
    flat = lab.reshape(-1, 3)
    d2 = ((flat[:, None, :] - palette[None, :, :]) ** 2).sum(-1)
    return d2.argmin(axis=1).reshape(lab.shape[:2])


def fixed_palette(k):
    """A generic palette of evenly spaced colors that ignores the image.
    """
    steps = np.linspace(0, 255, 4).astype(int)
    grid = np.array([[r, g, b] for r in steps for g in steps for b in steps])
    return rgb_to_lab(grid[:k])


def compare_widths(path, widths, colors, seed):
    print("Flat quantization vs dithering, on the real pipeline")
    print(f"{'width':>6} {'mode':<12} {'isolated':>9} {'% grid':>7} "
          f"{'per-px dE':>10} {'area dE':>8}")
    for w in widths:
        lab = rgb_to_lab(load_grid(path, w))
        centers, flat_idx = quantize_image(lab, colors, seed=seed)
        total = flat_idx.size

        for mode, idx in [("flat", flat_idx),
                          ("dithered", floyd_steinberg(lab, centers))]:
            iso = count_isolated(idx)
            pp, area = errors(lab, centers, idx)
            print(f"{w:>6} {mode:<12} {iso:>9} {iso / total * 100:>6.1f}% "
                  f"{pp:>10.2f} {area:>8.2f}")

        # One declutter pass on the dithered version, to show the cost of
        # making it sewable again.
        dith = floyd_steinberg(lab, centers)
        cleaned, changed = reduce_confetti(dith, passes=1)
        iso = count_isolated(cleaned)
        pp, area = errors(lab, centers, cleaned)
        print(f"{w:>6} {'+declutter':<12} {iso:>9} {iso / total * 100:>6.1f}% "
              f"{pp:>10.2f} {area:>8.2f}   ({changed} stitches rewritten)")
        print()


def compare_palettes(path, width, colors, seed):
    """The explanation: dithering helps a lot more when the palette is fixed."""
    print("Why dithering barely helps here: adaptive vs fixed palette")
    print(f"{'palette':<24} {'flat':>8} {'dithered':>9} {'change':>8}   "
          f"(area-average dE)")

    lab = rgb_to_lab(load_grid(path, width))
    adaptive, _ = quantize_image(lab, colors, seed=seed)

    for name, pal in [("adaptive (k-means)", adaptive),
                      ("fixed (generic grid)", fixed_palette(colors))]:
        _, flat_area = errors(lab, pal, nearest(lab, pal))
        _, dith_area = errors(lab, pal, floyd_steinberg(lab, pal))
        pct = (dith_area - flat_area) / flat_area * 100
        print(f"{name:<24} {flat_area:>8.2f} {dith_area:>9.2f} {pct:>+7.0f}%")



def main():
    ap = argparse.ArgumentParser(description="Measure whether dithering helps.")
    ap.add_argument("image", help="input image path")
    ap.add_argument("--widths", type=int, nargs="+", default=[100, 250],
                    help="pattern widths in stitches to test")
    ap.add_argument("--colors", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    compare_widths(args.image, args.widths, args.colors, args.seed)
    compare_palettes(args.image, args.widths[0], args.colors, args.seed)


if __name__ == "__main__":
    main()