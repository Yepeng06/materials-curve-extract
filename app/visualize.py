from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd


def draw_overlay(image_path: Path, points: list[dict], output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"failed to read image: {image_path}")

    for p in points:
        x = int(round(p["pixel_x"]))
        y = int(round(p["pixel_y"]))
        cv2.circle(image, (x, y), radius=2, color=(0, 0, 255), thickness=-1)

    cv2.imwrite(str(output_path), image)


def draw_redrawn_curve(csv_path: Path | None = None, data_points: list[dict] | None = None, output_path: Path | None = None) -> None:
    if output_path is None:
        raise ValueError("output_path is required")
    if csv_path is None and data_points is None:
        raise ValueError("either csv_path or data_points is required")

    if data_points is None:
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(data_points)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    ax.plot(df["x"], df["y"], color="#1f77b4", linewidth=1.8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
