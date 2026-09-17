"""DMC thread palette: loading, and nearest-thread lookup in LAB space."""

import csv
from pathlib import Path

import numpy as np
from .color import rgb_to_lab

DATA = Path(__file__).parent / "data" / "dmc.csv"

_CODE_KEYS = ("code", "floss", "number", "dmc", "dmccode", "dmccolor",
              "dmcnumber", "id")
_NAME_KEYS = ("name", "description", "colorname", "color", "desc")
_R_KEYS = ("r", "red")
_G_KEYS = ("g", "green")
_B_KEYS = ("b", "blue")
_HEX_KEYS = ("hex", "hexcode", "hexcolor", "rgbcolor", "rgb", "hexrgb", "html")

def _normalize(s):
    return s.strip().lower().replace(" ", "").replace("_", "").replace("#", "")

def _find(fieldnames, candidates):
    """Return the real column name matching any candidate, or None."""
    lookup = {_normalize(f): f for f in fieldnames if f}
    for c in candidates:
        if c in lookup:
            return lookup[c]
    return None

def _hex_to_rgb(value):
    v = value.strip().lstrip("#")
    if len(v) != 6:
        raise ValueError(f"expected a 6-digit hex color, got {value!r}")
    return [int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)]

class Palette:
    """A set of threads, with RGB and precomputed LAB coordinates."""

    def __init__(self, codes, names, rgb):
        self.codes = list(codes)
        self.names = list(names)
        self.rgb = np.asarray(rgb, dtype=np.uint8)
        self.lab = rgb_to_lab(self.rgb)

    def __len__(self):
        return len(self.codes)

    @classmethod
    def load(cls, path=DATA):
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            if not fields:
                raise ValueError(f"{path} has no header row")

            code_col = _find(fields, _CODE_KEYS)
            name_col = _find(fields, _NAME_KEYS)
            hex_col = _find(fields, _HEX_KEYS)
            r_col = _find(fields, _R_KEYS)
            g_col = _find(fields, _G_KEYS)
            b_col = _find(fields, _B_KEYS)

            if code_col is None:
                raise ValueError(
                    f"no thread-code column found in {path}. "
                    f"Columns present: {fields}")
            if hex_col is None and not (r_col and g_col and b_col):
                raise ValueError(
                    f"no color columns found in {path}. Need either a hex "
                    f"column or separate r/g/b columns. Columns present: {fields}")

            codes, names, rgb = [], [], []
            for lineno, row in enumerate(reader, start=2):
                code = (row.get(code_col) or "").strip()
                if not code:
                    continue  # skip blank trailing lines
                try:
                    if hex_col is not None:
                        color = _hex_to_rgb(row[hex_col])
                    else:
                        color = [int(float(row[r_col])),
                                 int(float(row[g_col])),
                                 int(float(row[b_col]))]
                except (ValueError, TypeError, AttributeError) as e:
                    raise ValueError(f"{path} line {lineno}: {e}") from e

                codes.append(code)
                names.append((row.get(name_col) or code).strip() if name_col else code)
                rgb.append(color)

        if not codes:
            raise ValueError(f"{path} contained no usable rows")
        return cls(codes, names, rgb)

    def nearest(self, lab):
        """Index of the closest thread to each LAB color. (...,3) -> (...)"""
        lab = np.asarray(lab, dtype=np.float64)
        flat = lab.reshape(-1, 1, 3)
        d2 = ((flat - self.lab[None, :, :]) ** 2).sum(-1)
        return d2.argmin(axis=1).reshape(lab.shape[:-1])

    def subset(self, indices):
        """A new Palette containing only the given indices, deduped, in order."""
        idx = list(dict.fromkeys(int(i) for i in indices))
        return Palette([self.codes[i] for i in idx],
                       [self.names[i] for i in idx],
                       self.rgb[idx])