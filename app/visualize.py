from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd


def draw_overlay(image_path: Path, points: list[dict], output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"failed to read image: {image_path}")

    if len(points) > 2000:
        radius = 1
        thickness = 1
    else:
        radius = 2
        thickness = -1

    for p in points:
        x = int(round(p["pixel_x"]))
        y = int(round(p["pixel_y"]))
        cv2.circle(image, (x, y), radius=radius, color=(0, 0, 255), thickness=thickness)

    if len(points) >= 2:
        line_pts = [(int(round(p["pixel_x"])), int(round(p["pixel_y"]))) for p in points]
        for i in range(1, len(line_pts)):
            cv2.line(image, line_pts[i - 1], line_pts[i], color=(0, 255, 0), thickness=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def draw_redrawn_curve(data_points: list[dict], output_path: Path) -> None:
    df = pd.DataFrame(data_points)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    ax.plot(df["x"], df["y"], color="#1f77b4", linewidth=1.8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Extracted Curve (Redrawn)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
