"""Command line entry point.

Runs the steps in order: load, shrink to a grid, quantize, match to thread,
render.
"""

import argparse
from pathlib import Path

import numpy as np

from .chart import render_preview, thread_usage
from .cleanup import count_isolated, reduce_confetti
from .color import rgb_to_lab
from .dither import floyd_steinberg
from .image_io import finished_size, load_grid
from .palette import Palette
from .quantize import quantize_image


def build_parser():
    """Set up the command line options."""
    p = argparse.ArgumentParser(
        prog="stitchlate",
        description="Convert an image into a cross-stitch pattern "
                    "using DMC thread colors.")
    p.add_argument("image", help="input image path")
    p.add_argument("-w", "--width", type=int, default=100,
                   help="pattern width in stitches (default: 100)")
    p.add_argument("-c", "--colors", type=int, default=25,
                   help="number of thread colors (default: 25)")
    p.add_argument("--dither", action="store_true",
                help="spread color error to nearby stitches for smoother "
                    "gradients (makes more confetti)")
    p.add_argument("--declutter", type=int, default=0, metavar="N",
                   help="passes of confetti removal; 1-2 is usually enough "
                        "(recommended with --dither)")
    p.add_argument("-o", "--out", default="out", help="output directory")
    p.add_argument("--aida", type=int, default=14,
                   help="Aida cloth count, for the finished size estimate")
    p.add_argument("--cell", type=int, default=8,
                   help="preview pixels per stitch (default: 8)")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed; fixed by default so runs are reproducible")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    # Shrink the image to a stitch grid.
    grid = load_grid(args.image, args.width)
    h, w = grid.shape[:2]
    # Pick the k best colors.
    lab = rgb_to_lab(grid)
    centers, labels = quantize_image(lab, args.colors, seed=args.seed)
    if args.dither:
        labels = floyd_steinberg(lab, centers)
    if args.declutter:
        before = count_isolated(labels)
        labels, changed = reduce_confetti(labels, passes=args.declutter)
        after = count_isolated(labels)
    # Match each color to the closest real DMC thread.
    full = Palette.load()
    center_to_thread = full.nearest(centers)
    used = full.subset(center_to_thread)
    # Point the labels at the new smaller palette.
    code_to_pos = {code: i for i, code in enumerate(used.codes)}
    remap = np.array([code_to_pos[full.codes[t]] for t in center_to_thread])
    stitches = remap[labels]
    collapsed = args.colors - len(used.codes)
    # Draw the preview and save it.
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    preview_path = outdir / "preview.png"
    render_preview(stitches, used, cell=args.cell).save(preview_path)

    # Print the size, how many threads it used, and the thread list.
    fw, fh = finished_size(grid.shape, args.aida)
    usage = thread_usage(stitches, used)
    print(f"grid:          {w} x {h} stitches ({w * h:,} total)")
    print(f"finished size: {fw:.1f} x {fh:.1f} in on {args.aida}-count Aida")
    print(f"colors:        {len(usage)} threads (requested {args.colors})")
    if collapsed > 0:
        print(f"               {collapsed} cluster(s) collapsed -- distinct "
              f"clusters matched the same thread")
    print(f"preview:       {preview_path}")
    if args.declutter:
        print(f"declutter:     {before} -> {after} isolated stitches "
                f"({changed} changed)")
    print()
    print("top threads:")
    for code, name, _, count in usage[:10]:
        pct = count / (w * h) * 100
        print(f"  {code:>6}  {name[:28]:<28} {count:>6} stitches ({pct:4.1f}%)")


if __name__ == "__main__":
    main()