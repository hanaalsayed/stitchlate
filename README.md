# stitchlate

![CI](https://github.com/hanals0876/stitchlate/actions/workflows/ci.yml/badge.svg)

stitch + translate. Turn a photo into a cross-stitch pattern using real DMC thread colors.

![250 stitches wide, 25 colors](examples/preview_250.png)

## Why I built this

I wanted to take a creative hobby and look at it technically, to find where the two actually meet. I ran a small craft shop for a couple of years, so stitch patterns were something I already understood as a craft. This project was me translating what they look like as a computing problem.

There was more to it than I expected. Picking the colors is a clustering problem, matching them to real thread is a nearest-neighbor search, and both of those depend on measuring color the way people see it instead of the way screens store it.

## Install

```bash
git clone https://github.com/hanals0876/stitchlate.git
cd stitchlate
pip install -e .
```

## Usage

```bash
stitchlate photo.jpg -w 100 -c 25 --out out
```

```
grid:          100 x 75 stitches (7,500 total)
finished size: 7.1 x 5.4 in on 14-count Aida
colors:        25 threads (requested 25)
preview:       out/preview.png

top threads:
    3033  Mocha Brown - VY LT             970 stitches (12.9%)
     817  Coral Red - VY DK               592 stitches ( 7.9%)
    3861  Cocoa - LT                      537 stitches ( 7.2%)
     986  Forest Green - VY DK            404 stitches ( 5.4%)
     317  Pewter Gray                     367 stitches ( 4.9%)
```

| flag | what it does |
|---|---|
| `-w, --width` | pattern width in stitches (default 100) |
| `-c, --colors` | how many thread colors to use (default 25) |
| `--aida` | cloth count for the finished size estimate (default 14) |
| `--dither` | spread color error to nearby stitches. Off by default; see below for why |
| `--declutter N` | N passes of confetti removal |
| `--seed` | fixed by default, so the same input always gives the same pattern |
| `-o, --out` | output directory |

Width is in stitches rather than pixels because that is the unit that decides
both the finished size and how long the piece takes to sew. 100 stitches on
14-count Aida is about 7 inches; 250 is about 18.

## How it works

1. Shrink the photo to the stitch grid, using LANCZOS so pixels get averaged rather than dropped
2. Convert to CIELAB
3. Pick the best N colors with k-means, using k-means++ initialization
4. Match each of those colors to the nearest real DMC thread
5. Render the preview and the thread list

### Color matching happens in LAB, not RGB

RGB describes how a screen mixes light. It does not describe how colors look to
a person, so two colors can be close in RGB numbers and obviously different to
the eye. The clearest evidence is in the sRGB conversion matrix itself: green
accounts for 72% of perceived brightness, red 21%, and blue only 7%.

CIELAB was built from experiments on human vision, so distance in it lines up
with what people actually see. This whole tool is a nearest-neighbor search, so
the space that search runs in decides which thread gets picked.

Measured against the 456-color DMC palette, 50,000 pixels, seed 7:

| | uniform random colors | real photograph |
|---|---|---|
| Different thread chosen | 56.7% | 36.1% |
| Mean error, RGB matching | 16.60 ΔE | 6.88 ΔE |
| Mean error, LAB matching | 12.99 ΔE | 5.55 ΔE |
| **Mean error reduction** | **21.7%** | **19.3%** |
| Penalty when they disagree | 6.38 ΔE | 3.67 ΔE |
| Worst case | 41.80 ΔE | 30.37 ΔE |

A ΔE around 2.3 is the smallest difference most people can see, so a 3.67
average penalty is comfortably visible rather than a rounding error.

Both columns are reported because the uniform number needs a caveat: random RGB
colors are not distributed like real photographs, which cluster in skin tones,
sky, and foliage. The photograph column is the one that describes actual use.

The worst case on a real photo shows *how* RGB fails. For a very dark green,
RGB picks thread 939, a near-black navy. LAB picks thread 500, a dark green.
Both are dark, so RGB treats them as similar, but the hue is completely wrong.
Stitched up, foliage shadows would come out blue. The same pair turned up as
the worst case across separate runs, so it is a systematic failure in dark
tones rather than a one-off.

Reproduce it:

```bash
python -m scripts.rgb_vs_lab --samples 50000 --seed 7
python -m scripts.rgb_vs_lab --image examples/gals.png --samples 50000 --seed 7
```

### Why dithering is off by default

| 100 stitches, no dither | 100 stitches, dithered |
|---|---|
| ![](examples/preview_100.png) | ![](examples/preview_100_dither.png) |

| 250 stitches, no dither | 250 stitches, dithered |
|---|---|
| ![](examples/preview_250.png) | ![](examples/preview_250_dither.png) |

Floyd-Steinberg dithering is the standard way to reduce an image to a small
palette, so I implemented it and expected it to help. It made the patterns
worse. Working out why turned out to be the most interesting part of this
project.

**At 100 stitches wide** (7,500 stitches):

| | isolated stitches | % of grid | per-pixel ΔE | area ΔE |
|---|---|---|---|---|
| Flat quantization | 676 | 9.0% | 6.13 | 3.77 |
| Dithered | 1,696 | 22.6% | 8.44 | 3.14 |
| Dithered + declutter ×1 | 320 | 4.3% | 8.45 | 4.47 |

**At 250 stitches wide** (47,000 stitches):

| | isolated stitches | % of grid | per-pixel ΔE | area ΔE |
|---|---|---|---|---|
| Flat quantization | 2,408 | 5.1% | 6.11 | 4.31 |
| Dithered | 10,761 | 22.9% | 9.25 | 3.50 |
| Dithered + declutter ×1 | 2,197 | 4.7% | 8.43 | 4.88 |

Two error metrics, because the difference matters. Per-pixel ΔE measures how
far each single stitch is from the original color. Area ΔE measures the same
thing after a 3x3 blur, which is closer to how a finished piece reads from a
distance. Dithering deliberately makes individual stitches less accurate so
that small areas average out correctly, so judging it per-pixel misses the
point.

**Dithering does improve color.** Area ΔE drops 17% at 100 wide (3.77 to 3.14)
and 19% at 250 (4.31 to 3.50). That is a real gain, not noise.

**It pays for that with spatial noise.** Isolated single stitches go from 9.0%
to 22.6% of the grid at 100 wide, and from 5.1% to 22.9% at 250, a 4.5x
increase. Stitchers call these confetti, and each one means cutting the thread,
rethreading the needle, sewing one X, and starting over. So the version with
the best color is the version nobody wants to sew.

**Cleaning it up gives back more than dithering gained.** One declutter pass
brings the 250-wide dithered pattern to 4.7% isolated, essentially matching
flat quantization's 5.1%, but at worse color accuracy: 4.88 ΔE against 4.31. It
rewrites 10,761 of 47,000 stitches, 23% of the pattern, to get there. Flat
quantization wins on both axes at once, so there is no tradeoff left to tune.

**Higher resolution makes it worse, not better.** I expected more stitches to
give dithering room to work. The opposite happened: the confetti increase went
from 2.5x to 4.5x going from 100 to 250 wide, because more stitches means more
places to scatter error.

**One hypothesis I tested and dropped.** I thought dithering might be redundant
here because k-means fits the palette to each specific image, leaving little
error to diffuse. If that were true, a fixed generic palette should benefit far
more. It does not:

| palette | flat | dithered | change |
|---|---|---|---|
| Adaptive (k-means, fitted to this image) | 3.77 | 3.14 | −17% |
| Fixed (generic evenly-spaced colors) | 23.31 | 19.47 | −16% |

Dithering works about as well here as in the fixed-palette case it was designed
for. The reason it loses is the confetti cost, which only exists because the
output is something a person sews by hand rather than an image on a screen.

Reproduce it:

```bash
python -m scripts.dither_tradeoff examples/gals.png --widths 100 250
```

`--dither` and `--declutter` are both still there and tested, since they are
the evidence for this.

## Design decisions

- **Match colors in LAB, not RGB.** 19% lower color error on a real photograph.
- **k-means over median cut.** Median cut is faster, but it can only split the color space along axis-aligned planes, so it does badly when colors are grouped diagonally.
- **Cluster on a sample, not every pixel.** The assign step builds an (n, k) array, which is 12 million rows for a large photo. A few thousand pixels already capture an image's color distribution,so clustering a sample and then labelling every pixel in one pass is enough — 5.4x faster at 120,000 pixels with k=25, and the gap widens as images get bigger.
- **LANCZOS when shrinking.** NEAREST would sample one pixel per stitch and throw the rest away, so small details disappear.
- **Width in stitches, not pixels.** It is the unit that decides finished size and sewing time.
- **Seeded RNG by default.** The same photo always produces the same pattern, which also makes the tests possible.
- **Detect CSV column names instead of requiring a fixed format.** Published DMC color files all name their columns differently, so the loader figures out which scheme it is looking at rather than needing the file edited by hand.
- **Thread codes stay strings.** Some real DMC codes are not numbers: Ecru, Blanc, B5200.

## Limitations

- Thread RGB values are approximations. Real floss color depends on dye lot, fiber sheen, and lighting, so this gets you close but does not replace holding the thread up to the light.
- DMC has no thread for some colors. Very saturated blues have no good match, and the tool picks the least wrong option without telling you the match is poor.
- The area ΔE metric is a 3x3 blur, which is a crude stand-in for human vision. A proper perceptual model would weight by spatial frequency.
- Large images are sampled before clustering, so results are an approximation rather than the exact optimum.
- Dithering is sequential by nature, since each stitch depends on error from the ones before it, so that loop cannot be vectorized the way the rest of the pipeline is.
- `--colors N` does not guarantee N threads. Two clusters can match the same thread and collapse into one.

## Testing

```bash
python -m pytest -v
python -m ruff check .
```

33 tests across all six modules: color conversion, palette loading, image loading, quantization, dithering, and confetti removal.

The tests worth knowing about:

- `test_red_reference` checks the LAB conversion against the published sRGB value for pure red. Any wrong constant or skipped step in the pipeline shows up here.
- `test_error_is_conserved` dithers a gray ramp down to black and white and checks the average brightness survives. That is the real proof error diffusion is distributing error rather than losing it.
- `test_exact_colors_map_to_themselves` checks every DMC thread's nearest match is itself, which catches almost any indexing bug in the palette lookup.

One thing I hit early: my first white-point test used an exact tolerance and
failed. The published sRGB matrix constants are rounded to 7 decimals, so pure
white lands about 4e-6 off from L*=100 rather than exactly on it. The tests use
a tolerance instead.

## Roadmap

- Chart output with grid lines, per-color symbols, and a printable legend
- PDF export
- Warn about threads used for only a handful of stitches, since 20 stitches means buying a whole skein for 1% of it
- Guarantee `--colors N` actually yields N threads
- Backstitch outlines
- Merge palette entries that are perceptually identical, using a ΔE threshold

## Credits

DMC color data compiled from
[nathantspencer/DMC-ColorCodes](https://github.com/nathantspencer/DMC-ColorCodes),
which scrapes hex approximations from DMC's published color list. That repo has
no license file, so it's credited here rather than claimed. Any CSV with a
thread code column and either hex or r/g/b columns will work instead — see
`--palette` below.

Thread codes and color names are DMC trademarks. This project is not
affiliated with DMC.
