"""Legend matching — placeholder for the future multi-curve stage.

Baseline behaviour: single curve -> nothing to match, curves pass through.
The full implementation (Phase 2, multi-curve) will:
  1. locate the legend box (detector output / layout heuristics);
  2. OCR the legend texts;
  3. associate each legend entry with a curve via colour/line-style distance
     (curve colours come from Curve.color, already computed);
  4. fill Curve.legend_label.

Keeping the protocol here means the multi-curve backend can be developed and
unit-tested without touching any other module.
"""
from __future__ import annotations

from typing import Dict, List

from ..schema import ChartStructure, Curve
from .base import TextBox


def match_legends(
    curves: List[Curve],
    structure: ChartStructure,
    text_boxes: List[TextBox],
    cfg: Dict | None = None,
) -> List[Curve]:
    """Baseline: identity. Multi-curve backend replaces this function."""
    return curves
