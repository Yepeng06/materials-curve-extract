"""Small shared helpers: image IO, binarization, number parsing."""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Unicode superscript digits and minus sign (matplotlib log-axis tick labels
# render e.g. "10⁻³" as real superscript glyphs; OCR returns them as-is).
# Superscript digits become '^N' so '10⁰' parses as 10^0 = 1, not 100.
_SUPERSCRIPT = str.maketrans({
    "⁰": "^0", "¹": "^1", "²": "^2", "³": "^3", "⁴": "^4",
    "⁵": "^5", "⁶": "^6", "⁷": "^7", "⁸": "^8", "⁹": "^9",
})

# exponent after "10": ^-3 | -^3 | -3 | ^3  (handles superscripts both orders)
_EXP = r"(?:\^(-?\d+)|(-)\^?(\d+)|([+-]\d+))"
_SCI_1 = re.compile(r"^([+-]?\d*\.?\d+)\s*[xX]\s*10\s*" + _EXP + r"$")  # 1.5x10^-3 / 1.5x10-^3
_SCI_2 = re.compile(r"^10" + _EXP + r"$")  # 10^3 / 10-3 / 10^-3 (NOT plain '1000')
_PERCENT = re.compile(r"^([+-]?\d*\.?\d+)%?$")


# Confusable glyph → digit map (applied ONLY when the whole token otherwise
# fails to parse and every char is a known confusable: OCR of small tick
# fonts misreads 0/O/o, 1/l/I, 5/S, 8/B, 6/b, 9/g).
_CONFUSABLE = str.maketrans({
    "O": "0", "o": "0", "l": "1", "I": "1", "S": "5", "B": "8",
    "b": "6", "g": "9", "Z": "2", "z": "2",
})

# Common trailing units / decorations on tick or axis-corner labels that
# parse_number_text may still meet ('10 MPa', '0.5 mm', '~25', '>100').
_TRAILING_UNITS = (
    "MPa", "GPa", "kPa", "Pa", "mm", "cm", "km", "µm", "μm", "um",
    "min", "sec", "ms", "ks", "hr", "°C", "°F", "K", "h", "s", "%",
    "C", "F",
)


def read_image(path: str) -> np.ndarray:
    """Read an image file as BGR uint8. Raises ValueError on failure."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read image: {path}")
    return img


def otsu_ink(gray: np.ndarray) -> np.ndarray:
    """Binarize: 1 = ink (dark foreground), 0 = background, uint8."""
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return (th // 255).astype(np.uint8)


def ink_mask(gray: np.ndarray) -> np.ndarray:
    """Robust ink binarization (1 = ink, uint8).

    Primary: adaptive Gaussian threshold — captures light-colored curves and
    survives JPEG noise / uneven scan background, which global Otsu misses
    when the foreground fraction is tiny (a known Otsu failure mode).
    Fallback: global Otsu for dark-background figures (adaptive assumes a
    light background).
    """
    if gray.mean() < 128:  # dark-background figure
        return otsu_ink(gray)
    ad = cv2.adaptiveThreshold(
        gray, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12
    )
    return ad.astype(np.uint8)


def parse_number_text(text: Optional[str]) -> Optional[float]:
    """Parse an OCR'd tick label into a float.

    Handles: plain floats, thousands separators, '1e-3', '1.5x10^3',
    '10^-3', unicode superscripts ('10⁻³', '10⁰'), unicode minus signs,
    a trailing '%', matplotlib mathtext markup (e.g.
    '$\\mathdefault{10^{-2}}$'), plus the L4 extensions: '±'/'~'/'>'/'<'
    prefixes, '°', trailing unit tokens ('10 MPa' -> 10), uppercase 'E'
    notation, and confusable-glyph tokens ('l0' -> 10) — the latter only
    when every char is a known confusable and the result parses.
    Returns None when not a number.
    """
    if not text:
        return None
    s = str(text).strip()
    # strip mathtext markup
    s = s.replace("$", "").replace("{", "").replace("}", "")
    s = s.replace("\\mathdefault", "").replace("\\times", "x").replace("\\cdot", "x")
    s = s.replace("⁻", "-").replace("−", "-").replace("–", "-").replace("—", "-")
    s = s.translate(_SUPERSCRIPT)
    s = s.replace(",", "").replace(" ", "").replace("×", "x").replace("✕", "x")
    # L4: leading approximation/comparison markers, degree signs, ± ranges
    s = s.lstrip("~≈><≥≤±")
    s = s.replace("°", "")
    if not s:
        return None

    m = _SCI_1.match(s)
    if m:
        # groups: 1=mantissa, 2=^exp, 3='-', 4=digits, 5=±digits
        e = m.group(2)
        if e is None:
            e = (m.group(3) or "") + (m.group(4) or "") if m.group(3) is not None else m.group(5)
        return float(m.group(1)) * 10.0 ** float(e)
    m = _SCI_2.match(s)
    if m:
        # groups: 1=^exp, 2='-', 3=digits, 4=±digits
        e = m.group(1)
        if e is None:
            e = (m.group(2) or "") + (m.group(3) or "") if m.group(2) is not None else m.group(4)
        return 10.0 ** float(e)
    m = _PERCENT.match(s)
    if m:
        return float(m.group(1))
    try:
        return float(s)
    except ValueError:
        pass
    # L4: trailing unit token ('10MPa' -> 10; '%' already handled above)
    for u in _TRAILING_UNITS:
        if s.endswith(u) and len(s) > len(u):
            tail = s[:-len(u)]
            if tail and tail[-1] in "+-":
                tail = tail[:-1]
            try:
                return float(tail)
            except ValueError:
                pass
            break
    # L4: confusable-glyph rescue — only if EVERY char maps to a digit/sign
    rescue = s.translate(_CONFUSABLE)
    if rescue != s and rescue and all(
            c.isdigit() or c in "+-." for c in rescue):
        try:
            return float(rescue)
        except ValueError:
            pass
    return None


def downsample_chain(chain: List[Tuple[int, int]], max_points: int) -> List[Tuple[int, int]]:
    """Uniformly subsample a pixel chain to at most max_points."""
    if len(chain) <= max_points:
        return chain
    stride = len(chain) / max_points
    return [chain[int(i * stride)] for i in range(max_points)]


def dedupe_consecutive(chain: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for p in chain:
        if not out or p != out[-1]:
            out.append(p)
    return out
