from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_csv(points_data, path: Path) -> None:
    df = pd.DataFrame(points_data, columns=["x", "y"])
    df.to_csv(path, index=False)
