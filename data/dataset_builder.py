"""materials-curve-dataset-platform → mci baseline 数据适配器（Phase A.1）。

把 dataset-platform（V0fix-final-2）的生成器 API 接入 baseline 数据格式：

    <stem>.png             最终图像（可选退化后）                    [baseline]
    <stem>_mask.png        曲线二值掩码（多曲线合一，实线绘制）       [baseline]
    <stem>.csv             GT (x,y) —— 仅单曲线样本                    [baseline]
    <stem>_cN.csv          每条曲线 GT (x,y) —— 多曲线样本             [baseline]
    <stem>_curves.json     曲线清单（curve_id / label / csv 文件）     [baseline]
    <stem>_meta.json       轴类型/范围/刻度值/样式/退化记录            [baseline]
    <stem>_labels.json     stub-OCR 刻度标签侧车（确定性测试用）       [baseline]
    <stem>_mcg.json        平台 MCG-JSON（平台生态兼容）               [platform]
    <stem>_yolo.txt        YOLOv8 标签（--yolo，Phase B 预置）         [platform]

设计说明
--------
* 平台自带的渲染器（renderer.py / renderer_enhanced.py）不实现模板的
  image_settings / key_effects、不支持对数轴、不输出刻度值与坐标范围，
  且用 savefig 落盘（baseline 坑清单：GT 像素必须取自画布缓冲区）。
  因此本适配器复用平台的「参数采样 + 蠕变曲线模型 + MCG-JSON 模式 +
  质量检查 + YOLO 标签收集」，但自行从 Agg 画布缓冲区渲染，保证 GT
  像素级精确（buffer_rgba 与 transData 一一对应），并内置自检。
* 对平台的扩展（均在此文件内，平台仓库零改动）：
  - 线性/对数轴随机组合（平台仅线性）；
  - 退化流水线（JPEG/缩放/模糊/亮度对比度/椒盐 + low_quality 截图风）；
  - 模板 image_settings（分辨率/DPI）与 axis_settings（轴名/单位/范围）
    生效（平台渲染器忽略它们）；
  - 图例 inside_lower_left 映射修正（平台回退到 upper right）；
  - 模板 YAML 编码容错（UTF-8 优先、GBK 兜底 —— 平台自带 3 个 GBK
    模板会导致其 generate_training_data.py 的 utf-8 加载崩溃）；
  - YOLO 标签 bbox 的 y 轴翻转修正（平台按 matplotlib y-up 直接输出，
    与图像 y-down 坐标不一致）。
* 单曲线样本（--num-curves 1）可直接被 scripts/evaluate.py 评估；
  多曲线样本输出每曲线 CSV + 清单（多曲线评估属 Phase C）。

用法（mci 环境）:
    python data/dataset_builder.py --out-dir data/train_platform --count 2000 \
        --seed 20260815
    python data/dataset_builder.py --out-dir data/eval_platform --count 100 \
        --num-curves 1 --seed 20260816
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_PLATFORM_ROOT = r"F:\CLAUDE\NewProject1\materials-curve-dataset-platform"

# ---------------------------------------------------------------------------
# 平台生成器（纯数据源：采样 / 曲线模型 / MCG-JSON / 质量检查 / YOLO 标签）
# ---------------------------------------------------------------------------
_PLATFORM: dict | None = None


def _import_platform(platform_root: str) -> dict:
    """定位并导入 dataset-platform 的 generator 包（无需 torch/paddle）。"""
    global _PLATFORM
    if _PLATFORM is not None:
        return _PLATFORM
    root = Path(platform_root)
    if not (root / "generator").is_dir():
        raise SystemExit(
            f"dataset-platform not found under {root} — pass --platform-root"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import generator.annotation_schema as aschema
    import generator.curve_models as curve_models
    import generator.quality_check as quality_check
    import generator.sampler as sampler
    from generator.training_labels import collect_yolo_labels

    _PLATFORM = {
        "aschema": aschema,
        "curve_models": curve_models,
        "quality_check": quality_check,
        "sampler": sampler,
        "collect_yolo_labels": collect_yolo_labels,
    }
    return _PLATFORM


def load_templates(templates_dir: str) -> list[dict]:
    """加载全部模板 YAML（UTF-8 优先、GBK 兜底；平台自带 3 个 GBK 模板）。"""
    templates = []
    for yml in sorted(Path(templates_dir).glob("*.yaml")):
        raw = yml.read_bytes()
        data = None
        for enc in ("utf-8", "gbk"):
            try:
                data = yaml.safe_load(raw.decode(enc))
                break
            except UnicodeDecodeError:
                continue
        if not isinstance(data, dict):
            data = {}
        data["template_file"] = yml.stem
        templates.append(data)
    return templates


# ---------------------------------------------------------------------------
# 样式常量（与平台渲染器一致，含修正）
# ---------------------------------------------------------------------------
BW_PALETTE = ["#111111", "#444444", "#666666", "#888888", "#222222"]
MARKER_MAP = {"triangle": "^", "circle": "o", "square": "s"}

# 平台 _legend_loc 缺 inside_lower_left（会回退到 upper right）——此处补全
LEGEND_LOC_MAP = {
    "inside_upper_right": "upper right",
    "inside_upper_left": "upper left",
    "inside_lower_right": "lower right",
    "inside_lower_left": "lower left",
}

X_LABEL_UNITS = [
    ("Creep time", "h"), ("Time", "h"), ("Time", "s"),
    ("Creep time", "min"), ("Test time", "h"), ("t", "s"),
]
Y_LABEL_UNITS = [
    ("Creep strain", "%"), ("Strain", "%"), ("Stress", "MPa"),
    ("Creep strain", "mm/mm"), ("Displacement", "mm"),
]
TITLES = [
    "Creep curves", "Creep test of alloy", "Stress relaxation test",
    "Creep behaviour", "Strain evolution",
]

_PROB_FIELDS = {"num_curves", "curve_shape", "line_style", "marker",
                "legend_position", "grid"}


# ---------------------------------------------------------------------------
# 参数采样（平台 sample_parameters + 模板 settings 组 + 对数轴扩展）
# ---------------------------------------------------------------------------
def _settings_overrides(template: dict) -> dict:
    """模板 settings 组（image/axis/curve/style/legend/grid），平台渲染器
    忽略它们；这里只对「非概率采样」且「default_parameters 未覆盖」的键生效。"""
    out: dict = {}
    for group in ("image_settings", "axis_settings", "curve_settings",
                  "style_settings", "legend_settings", "grid_settings"):
        d = template.get(group) or {}
        for k, v in d.items():
            if k in ("width", "height", "dpi"):
                continue  # 分辨率在 sample_params 中单独处理
            if k in _PROB_FIELDS:
                continue  # 概率采样字段由模板分布决定
            if k in (template.get("default_parameters") or {}):
                continue  # default_parameters 优先
            out[k] = v
    return out


def sample_params(rng: np.random.Generator, template: dict, sample_seed: int,
                  sample_index: int, args) -> dict:
    """模板概率采样 → 平台参数；再叠加本适配器的轴类型/分辨率扩展。"""
    P = _PLATFORM
    tpl_id = template["template_file"]
    tpl_name = template.get("template_name", tpl_id)
    base = {
        "dataset_name": "training", "version": "v1", "mode": "probabilistic",
        "template_id": tpl_id, "seed": sample_seed,
        "num_curves": 3, "curve_shape": "three_stage",
        "line_style": "solid", "line_width": 1.5, "marker": "none",
        "grid": False, "legend_position": "inside_upper_right",
        "x_label": "Creep time", "x_unit": "h",
        "y_label": "Creep strain", "y_unit": "%",
        "x_range": [0.0, 1000.0], "points_per_curve": 160,
        "noise_level": 0.02, "title": f"Creep Curves - {tpl_name}",
    }
    params = P["sampler"].sample_parameters(
        "probabilistic", base, template_data=template, sample_index=sample_index
    )
    if args.num_curves and args.num_curves > 0:
        params["num_curves"] = int(min(5, args.num_curves))

    # 模板 settings 组（平台渲染器未实现的扩展）
    for k, v in _settings_overrides(template).items():
        params[k] = v

    # 轴标签/单位：模板 axis_settings 优先，否则在常用组合中随机
    ax_set = template.get("axis_settings") or {}
    x_label = ax_set.get("x_label") or params.get("x_label")
    x_unit = ax_set.get("x_unit") or params.get("x_unit")
    y_label = ax_set.get("y_label") or params.get("y_label")
    y_unit = ax_set.get("y_unit") or params.get("y_unit")
    if not (x_label and x_unit and y_label and y_unit) and rng.random() < 0.6:
        x_label, x_unit = rng.choice(X_LABEL_UNITS)
        y_label, y_unit = rng.choice(Y_LABEL_UNITS)

    # 坐标类型（平台仅线性 —— 本适配器扩展，保证对数轴训练覆盖）
    x_kind, y_kind = "linear", "linear"
    if args.allow_log:
        if rng.random() < 0.35:
            x_kind = "log"
        if rng.random() < 0.30:
            y_kind = "log"

    x_range = [float(v) for v in (ax_set.get("x_range") or params["x_range"])]
    if x_kind == "log":
        if x_range[0] <= 0:
            x_range[0] = 10.0 ** rng.uniform(-1.0, 0.3)
        if x_range[1] <= x_range[0]:
            x_range[1] = x_range[0] * 10.0
        dec = np.log10(x_range[1] / x_range[0])
        if dec < 1.5:  # 至少 1.5 个数量级，保证 ≥3 个对数刻度
            x_range[1] = x_range[0] * 10.0 ** 1.5

    img_set = template.get("image_settings") or {}
    dpi = float(img_set.get("dpi", 120))
    if img_set.get("width") and img_set.get("height"):
        figsize = (float(img_set["width"]) / dpi, float(img_set["height"]) / dpi)
    else:
        figsize = (8.0, 5.0)

    cfg = dict(params)
    cfg.update(x_kind=x_kind, y_kind=y_kind, dpi=dpi, figsize=figsize,
               x_range=x_range, x_label=x_label, x_unit=x_unit,
               y_label=y_label, y_unit=y_unit,
               template_file=tpl_id, template_name=tpl_name, seed=sample_seed)
    cfg["hard_crossing"] = float(getattr(args, "hard_crossing", 0.0))
    return cfg


# ---------------------------------------------------------------------------
# 曲线生成（平台曲线模型）
# ---------------------------------------------------------------------------
def _apply_hard_crossing(curves: list[dict], rng: np.random.Generator) -> None:
    """P2: make two curves cross or run close together (approach zones).

    The platform generator vertically layers curves by index, so crossing /
    touching curves are essentially absent from training -- which is exactly
    the measured weakness (17/31 of the >5% failures are approach-zone
    switching).  With probability per sample, pick a pair and either:
      'cross': shift curve i vertically so it intersects curve j at a random
               x (X-type crossing);
      'hug':   shift curve i's segment in [x_lo, x_hi] to run parallel to j
               at a small offset (2-6 px equivalent), with linear edge
               blends, then shift the whole curve to keep continuity.
    Data-domain offsets: y spans ~0-1.2 over ~600 px, so 1 px ≈ 0.002.
    """
    if len(curves) < 2:
        return
    i, j = rng.choice(len(curves), 2, replace=False)
    ci, cj = curves[i], curves[j]
    pts_i = np.asarray(ci["data_points"], dtype=np.float64)
    pts_j = np.asarray(cj["data_points"], dtype=np.float64)
    x0, x1 = float(pts_i[0, 0]), float(pts_i[-1, 0])
    span = max(x1 - x0, 1e-9)
    mode = rng.choice(["cross", "hug", "steep"], p=[0.35, 0.35, 0.3])
    if mode == "steep":
        # tertiary tail: exponentially amplify the last 15-35% of the curve
        # (creep rupture tail, accelerated_obvious failure pattern)
        x_tail = x0 + rng.uniform(0.65, 0.85) * span
        k = rng.uniform(2.0, 6.0)
        m = pts_i[:, 0] >= x_tail
        if int(m.sum()) >= 8:
            t = (pts_i[m, 0] - x_tail) / max(span * 0.4, 1e-9)
            pts_i[m, 1] = pts_i[m, 1] * np.exp(k * np.clip(t, 0.0, 1.0))
            ymax = max(float(pts_i[:, 1].max()), float(pts_j[:, 1].max())) * 1.3
            pts_i[:, 1] = np.minimum(pts_i[:, 1], ymax)
        ci["data_points"] = [[float(v) for v in p] for p in pts_i]
        return
    if mode == "cross":
        # ---- cross: shift curve i so it intersects j at a random x ----
        xc = x0 + rng.uniform(0.3, 0.7) * span
        yj = float(np.interp(xc, pts_j[:, 0], pts_j[:, 1]))
        yi = float(np.interp(xc, pts_i[:, 0], pts_i[:, 1]))
        pts_i[:, 1] += yj - yi
    else:
        # ---- hug: run parallel to j inside [x_lo, x_hi] ----
        x_lo = x0 + rng.uniform(0.08, 0.25) * span
        x_hi = x_lo + rng.uniform(0.2, 0.5) * span
        off = rng.uniform(0.004, 0.012) * (1.0 if rng.random() < 0.5 else -1.0)
        m = (pts_i[:, 0] >= x_lo) & (pts_i[:, 0] <= x_hi)
        if int(m.sum()) >= 8:
            yj_seg = np.interp(pts_i[m, 0], pts_j[:, 0], pts_j[:, 1])
            target = yj_seg + off
            blend = max(int(0.06 * span / max(np.diff(pts_i[m, 0]).mean(), 1e-9)), 2)
            n = int(m.sum())
            ramp = np.ones(n)
            if blend < n:
                ramp[:blend] = np.linspace(0, 1, blend)
                ramp[-blend:] = np.linspace(1, 0, blend)
            pts_i[m, 1] = pts_i[m, 1] * (1 - ramp) + target * ramp
    ci["data_points"] = [[float(v) for v in p] for p in pts_i]


def make_curves(cfg: dict, sample_seed: int) -> list[dict]:
    P = _PLATFORM
    rng = np.random.default_rng(sample_seed)
    curves = []
    for ci in range(cfg["num_curves"]):
        pts = P["curve_models"].generate_curve_data(
            curve_shape=cfg["curve_shape"],
            x_min=float(cfg["x_range"][0]), x_max=float(cfg["x_range"][1]),
            num_points=int(cfg["points_per_curve"]),
            noise_level=float(cfg["noise_level"]),
            rng=rng, curve_index=ci, total_curves=cfg["num_curves"],
        )
        curves.append({
            "curve_id": f"curve_{ci + 1}", "label": f"Curve {ci + 1}",
            "shape_type": cfg["curve_shape"],
            "line_style": cfg["line_style"], "line_width": cfg["line_width"],
            "marker": cfg["marker"], "curve_index": ci,
            "total_curves": cfg["num_curves"], "data_points": pts,
        })
    if float(cfg.get("hard_crossing", 0.0)) > 0 and rng.random() < float(cfg["hard_crossing"]):
        _apply_hard_crossing(curves, rng)
    return curves


# ---------------------------------------------------------------------------
# 渲染（GT 像素级精确：Agg 画布缓冲区）
# ---------------------------------------------------------------------------
def render_chart(cfg: dict, curves: list[dict]) -> tuple:
    """渲染一张图；返回 (img_bgr, mask, gt, labels, plot_bbox_xyxy_down, extra_bboxes)。

    画布整幅输出（不裁剪），buffer_rgba 像素与 transData 显示坐标一一对应。
    """
    with plt.rc_context({"axes.formatter.use_mathtext": False}):
        fig, ax = plt.subplots(figsize=cfg["figsize"], dpi=cfg["dpi"])
        ax.set_title(str(cfg.get("title", "")))
        x_unit = "" if str(cfg.get("x_unit", "")).lower() in ("", "none") else cfg["x_unit"]
        y_unit = "" if str(cfg.get("y_unit", "")).lower() in ("", "none") else cfg["y_unit"]
        ax.set_xlabel(f"{cfg['x_label']} ({x_unit})" if x_unit else str(cfg["x_label"]))
        ax.set_ylabel(f"{cfg['y_label']} ({y_unit})" if y_unit else str(cfg["y_label"]))
        if cfg.get("grid"):
            ax.grid(True)
        if cfg["x_kind"] == "log":
            ax.set_xscale("log")
        if cfg["y_kind"] == "log":
            ax.set_yscale("log")

        bw = cfg["template_file"] == "black_white_paper"
        for idx, curve in enumerate(curves):
            xs = [p[0] for p in curve["data_points"]]
            ys = [p[1] for p in curve["data_points"]]
            mkey = curve["marker"]
            marker = None if mkey in ("none", None) else MARKER_MAP.get(mkey, mkey)
            markevery = None if marker is None else max(8, min(20, len(xs) // 10))
            lw = max(2.5, float(curve["line_width"]))
            line, = ax.plot(
                xs, ys, linestyle=curve["line_style"], marker=marker,
                markevery=markevery, linewidth=lw, label=curve["label"],
                color=(BW_PALETTE[idx % len(BW_PALETTE)] if bw else None),
            )
            curve["line_color"] = line.get_color()
            curve["line_width_pt"] = lw

        legend_obj = None
        pos = cfg["legend_position"]
        if pos not in ("none", None):
            if pos == "outside_right":
                legend_obj = ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
                                       borderaxespad=0.0)
                fig.tight_layout(rect=(0, 0, 0.86, 1))
            else:
                legend_obj = ax.legend(loc=LEGEND_LOC_MAP.get(pos, "upper right"))
                fig.tight_layout()
        else:
            fig.tight_layout()

        fig.canvas.draw()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # ---- 对数轴视图修正：对齐到十年边界、保证 ≥3 个数量级 ----
        # （平台曲线数据通常 < 2 个数量级，自动缩放会只剩 1 个主刻度，
        #   不满足刻度判别/映射的 ≥2~3 刻度需求 —— 与 gen_synthetic 同策略）
        if cfg["x_kind"] == "log" and xlim[0] > 0:
            x_lo = 10.0 ** np.floor(np.log10(xlim[0]))
            x_hi = max(10.0 ** (np.floor(np.log10(xlim[0])) + 3.0), xlim[1] * 1.05)
            ax.set_xlim(x_lo, x_hi)
        if cfg["y_kind"] == "log" and ylim[0] > 0:
            y_lo = 10.0 ** np.floor(np.log10(ylim[0]))
            y_hi = max(10.0 ** (np.floor(np.log10(ylim[0])) + 3.0), ylim[1] * 1.1)
            ax.set_ylim(y_lo, y_hi)

        fig.canvas.draw()  # 限幅修正后的最终渲染
        W, H = fig.canvas.get_width_height()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        img = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
        renderer = fig.canvas.get_renderer()

        def y_down(py: float) -> float:
            return H - py

        # ---- 曲线像素点 + bbox（y-down，图像坐标）----
        curve_px_all: list[list[list[float]]] = []
        for curve in curves:
            disp = ax.transData.transform(
                [(p[0], p[1]) for p in curve["data_points"]]
            )
            pts = [[float(px), float(y_down(py))] for px, py in disp]
            curve_px_all.append(pts)
            xs = [c[0] for c in pts]
            ys = [c[1] for c in pts]
            curve["bbox_xyxy"] = [min(xs), min(ys), max(xs), max(ys)]
            curve["pixel_points"] = pts

        # ---- 掩码：实线绘制（虚线/点线也让网络学补全）----
        mask = np.zeros((H, W), np.uint8)
        for curve, pts in zip(curves, curve_px_all):
            lw_px = max(3, int(round(curve["line_width_pt"] * cfg["dpi"] / 72.0)))
            arr = np.array([(int(p[0]), int(p[1])) for p in pts], dtype=np.int32)
            cv2.polylines(mask, [arr], False, 255, lw_px)

        # ---- 轴 GT（实际渲染后的范围与刻度值）----
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        def _in_view(v: float, lo: float, hi: float) -> bool:
            return lo * (1 - 1e-9) <= v <= hi * (1 + 1e-9)

        x_tick_vals = [float(v) for v in ax.get_xticks() if _in_view(v, *xlim)]
        y_tick_vals = [float(v) for v in ax.get_yticks() if _in_view(v, *ylim)]

        # ---- stub-OCR 侧车：框锚定在刻度标记像素处（anchored=True）----
        label_h = max(14, int(cfg["dpi"] * 0.16))

        def _box(cx: float, cy: float, w: int) -> list:
            hw = w // 2
            hh = label_h // 2
            return [[cx - hw, cy - hh], [cx + hw, cy - hh],
                    [cx + hw, cy + hh], [cx - hw, cy + hh]]

        labels = []
        for v in x_tick_vals:
            mx, my = ax.transData.transform((v, ylim[0]))
            px, py = float(mx), y_down(float(my))
            if 0 <= px < W:
                labels.append({"box": _box(round(px), round(py + label_h // 2 + 6), 48),
                               "text": f"{v:g}", "score": 1.0})
        for v in y_tick_vals:
            mx, my = ax.transData.transform((xlim[0], v))
            px, py = float(mx), y_down(float(my))
            if 0 <= py < H:
                labels.append({"box": _box(round(px - label_h // 2 - 6), round(py), 48),
                               "text": f"{v:g}", "score": 1.0})

        # ---- 元素窗口范围（翻转 y：matplotlib y-up → 图像 y-down）----
        def _flip(bb) -> list:
            return [float(bb.x0), H - float(bb.y1), float(bb.x1), H - float(bb.y0)]

        plot_bb = _flip(ax.get_window_extent(renderer=renderer))
        extra = {"tick_label": [], "axis_title": [],
                 "x_axis_line": [], "y_axis_line": [], "legend_box": []}
        for lb in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            if lb.get_text().strip():
                extra["tick_label"].append(_flip(lb.get_window_extent(renderer=renderer)))
        for obj in (ax.xaxis.get_label(), ax.yaxis.get_label(), ax.title):
            if obj.get_text().strip():
                extra["axis_title"].append(_flip(obj.get_window_extent(renderer=renderer)))
        for spine_name, key in (("bottom", "x_axis_line"), ("top", "x_axis_line"),
                                ("left", "y_axis_line"), ("right", "y_axis_line")):
            sp = ax.spines[spine_name]
            if sp.get_visible():
                bb = _flip(sp.get_window_extent(renderer=renderer))
                if bb[2] > bb[0] and bb[3] > bb[1]:
                    extra[key].append(bb)
        if legend_obj is not None:
            extra["legend_box"].append(_flip(legend_obj.get_window_extent(renderer=renderer)))

        gt = {
            "x_kind": cfg["x_kind"], "y_kind": cfg["y_kind"],
            "x_range": [float(xlim[0]), float(xlim[1])],
            "y_range": [float(ylim[0]), float(ylim[1])],
            "x_tick_values": x_tick_vals, "y_tick_values": y_tick_vals,
            "title": str(cfg.get("title", "")),
            "x_label": str(cfg.get("x_label", "")),
            "x_unit": str(cfg.get("x_unit", "")),
            "y_label": str(cfg.get("y_label", "")),
            "y_unit": str(cfg.get("y_unit", "")),
            "curve_px": curve_px_all[0][::10],
            "curves_px": [c[::10] for c in curve_px_all],
            "template_id": cfg["template_file"],
            "template_name": cfg["template_name"],
            "num_curves": len(curves),
            "curve_shape": cfg["curve_shape"],
            "legend": cfg["legend_position"] not in ("none", None),
            "legend_position": cfg["legend_position"],
            "grid": bool(cfg.get("grid")), "spines": "box",
            "line_style": cfg["line_style"], "line_width": cfg["line_width"],
            "marker": cfg["marker"], "dpi": cfg["dpi"],
            "figsize": list(cfg["figsize"]), "seed": cfg["seed"],
            "degradations": [],
        }
        plt.close(fig)
        return img, mask, gt, labels, plot_bb, extra


# ---------------------------------------------------------------------------
# 退化（baseline 同款 + low_quality_screenshot 截图风 + right_fade 右端淡出）
# ---------------------------------------------------------------------------
def degrade(img_bgr: np.ndarray, rng: np.random.Generator, screenshot: bool = False,
            right_fade: float = 0.0) -> tuple:
    """图像退化流水线。

    ``right_fade`` (0-1)：以该概率对图像右端 20-40% 区域做额外的高斯模糊 +
    对比度降低（模拟"曲线右端概率峰丢失"的真实图模式 —— chain 模型诊断：
    >5% 桶 ~50% 是右端曲线丢失，集中在重模糊/低质量截图）。GT 掩码不变，
    模型必须学会在右端低置信区仍保持链的位置精度。
    """
    applied = []
    img = img_bgr
    if screenshot:
        s = rng.uniform(0.65, 0.85)
        small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (img.shape[1], img.shape[0]),
                         interpolation=cv2.INTER_LINEAR)
        applied.append(f"shot_resize_{s:.2f}")
        sigma = rng.uniform(0.4, 1.0)
        img = cv2.GaussianBlur(img, (0, 0), sigma)
        applied.append(f"shot_blur_{sigma:.2f}")
        q = int(rng.uniform(55, 82))
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        applied.append(f"shot_jpeg_{q}")
        return img, applied

    if rng.random() < 0.5:
        q = int(rng.uniform(55, 92))
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        applied.append(f"jpeg_q{q}")
    if rng.random() < 0.4:
        s = rng.uniform(0.72, 0.92)
        small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (img.shape[1], img.shape[0]),
                         interpolation=cv2.INTER_LINEAR)
        applied.append(f"resize_{s:.2f}")
    if rng.random() < 0.3:
        sigma = rng.uniform(0.3, 1.0)
        img = cv2.GaussianBlur(img, (0, 0), sigma)
        applied.append(f"blur_{sigma:.2f}")
    if rng.random() < 0.4:
        g = rng.uniform(0.85, 1.15)
        b = rng.uniform(-12, 12)
        img = np.clip(img.astype(np.float32) * g + b, 0, 255).astype(np.uint8)
        applied.append(f"bc_{g:.2f}_{b:.0f}")
    if rng.random() < 0.2:
        m = rng.random(img.shape[:2])
        img = img.copy()
        img[m < 0.0004] = 0
        img[m > 1 - 0.0004] = 255
        applied.append("snp")
    if right_fade > 0 and rng.random() < right_fade:
        # 右端 20-40% 区域：高斯模糊 + 对比度降低 + 轻微亮化（模拟照片
        # 右侧失焦/过曝导致曲线概率峰丢失）
        w = img.shape[1]
        x0 = int(w * rng.uniform(0.60, 0.80))
        region = img[:, x0:]
        sigma = rng.uniform(1.2, 3.0)
        region = cv2.GaussianBlur(region, (0, 0), sigma)
        g = rng.uniform(0.55, 0.85)
        b = rng.uniform(0, 30)
        region = np.clip(region.astype(np.float32) * g + b, 0, 255).astype(np.uint8)
        img[:, x0:] = region
        applied.append(f"right_fade_{x0 / w:.2f}_{sigma:.2f}")
    return img, applied


# ---------------------------------------------------------------------------
# 侧车写出
# ---------------------------------------------------------------------------
def write_sidecars(stem: str, curves: list[dict], img: np.ndarray,
                   mask: np.ndarray, gt: dict, labels: list, plot_bb: list,
                   extra: dict, cfg: dict, args) -> None:
    P = _PLATFORM
    cv2.imwrite(stem + ".png", img)
    cv2.imwrite(stem + "_mask.png", mask)

    # GT CSV：单曲线 → <stem>.csv；多曲线 → <stem>_cN.csv + 清单
    if len(curves) == 1:
        with open(stem + ".csv", "w", encoding="utf-8") as f:
            f.write("x,y\n")
            for px, py in curves[0]["data_points"]:
                f.write(f"{px:.8g},{py:.8g}\n")
        manifest = {"num_curves": 1, "curves": [
            {"curve_id": c["curve_id"], "label": c["label"],
             "csv": os.path.basename(stem) + ".csv"} for c in curves]}
    else:
        for ci, c in enumerate(curves):
            with open(f"{stem}_c{ci + 1}.csv", "w", encoding="utf-8") as f:
                f.write("x,y\n")
                for px, py in c["data_points"]:
                    f.write(f"{px:.8g},{py:.8g}\n")
        manifest = {"num_curves": len(curves), "curves": [
            {"curve_id": c["curve_id"], "label": c["label"],
             "csv": f"{os.path.basename(stem)}_c{ci + 1}.csv"}
            for ci, c in enumerate(curves)]}
    with open(stem + "_curves.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)

    with open(stem + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=1, ensure_ascii=False)
    with open(stem + "_labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f)

    # 平台 MCG-JSON（兼容平台生态）
    ann = build_mcg_payload(cfg, curves, img.shape, plot_bb, extra,
                            y_range=gt["y_range"])
    with open(stem + "_mcg.json", "w", encoding="utf-8") as f:
        json.dump(ann, f, indent=2, ensure_ascii=False)

    # YOLOv8 标签（Phase B 预置；bbox 已翻转为 y-down）
    if args.yolo:
        yolo_lines = P["collect_yolo_labels"](
            {"plot_area_bbox_xyxy": plot_bb}, img.shape[1], img.shape[0], extra
        )
        with open(stem + "_yolo.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))


def build_mcg_payload(cfg: dict, curves: list[dict], img_shape: tuple,
                      plot_bb: list, extra: dict, y_range: list | None = None) -> dict:
    P = _PLATFORM
    W, H = img_shape[1], img_shape[0]
    payload = {
        "dataset_info": {
            "name": "mci-baseline", "version": "v1", "dataset_version": "v1",
            "generator_version": "mci-adapter-V1",
            "mode": cfg.get("mode", "probabilistic"),
            "template_id": cfg["template_file"],
            "template_name": cfg["template_name"],
            "seed": cfg["seed"],
            "template_defaults_applied": cfg.get("template_defaults_applied", False),
        },
        "image": {"image_path": "", "width": int(W), "height": int(H)},
        "plot_area": {"bbox_xyxy": plot_bb},
        "axis": {
            "x_label": cfg["x_label"], "x_unit": cfg["x_unit"],
            "y_label": cfg["y_label"], "y_unit": cfg["y_unit"],
            "x_range": cfg["x_range"], "x_kind": cfg["x_kind"],
            "y_kind": cfg["y_kind"], "y_range": y_range or [],
        },
        "style": {
            "grid": cfg.get("grid"), "line_style": cfg["line_style"],
            "line_width": cfg["line_width"], "marker": cfg["marker"],
            "legend_position": cfg["legend_position"],
            "curve_shape": cfg["curve_shape"],
            "points_per_curve": cfg.get("points_per_curve"),
            "noise_level": cfg.get("noise_level"), "seed": cfg["seed"],
            "mode": cfg.get("mode", "probabilistic"),
            "template_id": cfg["template_file"],
            "template_name": cfg["template_name"],
            "sampled_parameters": cfg.get("sampled_parameters", {}),
            "template_enforced_fields": cfg.get("template_enforced_fields", {}),
            "template_defaults_applied": cfg.get("template_defaults_applied", False),
            "actual_parameters": cfg.get("actual_parameters", {}),
        },
        "legend": {"position": cfg["legend_position"]},
        "curves": [{
            "curve_id": c["curve_id"], "label": c["label"],
            "shape_type": c["shape_type"], "line_style": c["line_style"],
            "line_width": c["line_width"], "marker": c["marker"],
            "curve_index": c["curve_index"], "total_curves": c["total_curves"],
            "data_points": c["data_points"], "pixel_points": c["pixel_points"],
            "bbox_xyxy": c["bbox_xyxy"], "line_color": c["line_color"],
        } for c in curves],
        "quality_check": {},
    }
    ann = P["aschema"].build_mcg_json(payload)
    ann["quality_check"] = P["quality_check"].run_quality_check(ann)
    return ann


# ---------------------------------------------------------------------------
# 批量生成
# ---------------------------------------------------------------------------
def generate(args) -> int:
    t0 = time.time()
    _import_platform(args.platform_root)
    templates = load_templates(args.templates_dir)
    if args.templates:
        keep = set(t for t in args.templates.split(",") if t.strip())
        templates = [t for t in templates if t["template_file"] in keep]
    if not templates:
        print(f"[错误] 无可用模板，检查目录: {args.templates_dir}")
        return 1
    os.makedirs(args.out_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    n_per = args.count // len(templates)
    rem = args.count % len(templates)

    total = failed = warned = 0
    tpl_cnt, ncurves_cnt, shape_cnt, axis_cnt = Counter(), Counter(), Counter(), Counter()
    quality = Counter()

    print(f"[适配器] 模板 {len(templates)} 个 | 目标 {args.count} 张 | "
          f"seed {args.seed} | 输出 {args.out_dir}")

    for tpl_idx, template in enumerate(templates):
        tpl_id = template["template_file"]
        n_here = n_per + (1 if tpl_idx < rem else 0)
        for i in range(n_here):
            idx = args.start_idx + total
            stem = os.path.join(args.out_dir, f"img_{idx:04d}")
            sample_seed = args.seed + total
            try:
                cfg = sample_params(rng, template, sample_seed, i, args)
                curves = make_curves(cfg, sample_seed)
                img, mask, gt, labels, plot_bb, extra = render_chart(cfg, curves)

                # 自检：GT 曲线点必须落在墨迹附近（膨胀 25px 容忍虚线/抗锯齿）
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ink = (gray < 235).astype(np.uint8)
                ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
                bad = tot = 0
                for cpts in gt["curves_px"]:
                    for px, py in cpts:
                        if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
                            tot += 1
                            if ink[int(py), int(px)] == 0:
                                bad += 1
                if tot and bad / tot > 0.10:
                    print(f"[{idx}] 自检警告: {bad}/{tot} GT 点不在墨迹附近")
                    warned += 1

                screenshot = cfg["template_file"] == "low_quality_screenshot"
                if args.degrade:
                    img, applied = degrade(img, rng, screenshot=screenshot,
                                           right_fade=args.right_fade)
                    gt["degradations"] = applied
                gt["image_size"] = [int(img.shape[1]), int(img.shape[0])]

                write_sidecars(stem, curves, img, mask, gt, labels, plot_bb,
                               extra, cfg, args)
                total += 1
                tpl_cnt[tpl_id] += 1
                ncurves_cnt[str(cfg["num_curves"])] += 1
                shape_cnt[str(cfg["curve_shape"])] += 1
                axis_cnt[f"{cfg['x_kind']}/{cfg['y_kind']}"] += 1
                quality["ok"] += 1
                if total % 50 == 0:
                    print(f"  进度 {total}/{args.count} "
                          f"(warn {warned}, fail {failed})")
            except Exception:
                print(f"[{idx}] 生成失败:\n{traceback.format_exc()}")
                failed += 1
                quality["failed"] += 1

    summary = {
        "generator": "mci-dataset-builder-V1 (platform adapter)",
        "platform_root": args.platform_root,
        "target_count": args.count, "total_generated": total,
        "seed": args.seed, "start_idx": args.start_idx,
        "num_curves": args.num_curves,
        "template_counts": dict(tpl_cnt),
        "num_curves_distribution": dict(ncurves_cnt),
        "curve_shape_distribution": dict(shape_cnt),
        "axis_kind_counts": dict(axis_cnt),
        "quality": dict(quality),
        "warned": warned, "failed": failed,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"完成: {total}/{args.count} 张 (warn {warned}, fail {failed})")
    print(f"轴类型分布: {dict(axis_cnt)}")
    print(f"曲线数分布: {dict(ncurves_cnt)}")
    print(f"模板分布: {dict(tpl_cnt)}")
    print(f"输出目录: {args.out_dir}  (用时 {summary['elapsed_s']}s)")
    return 0 if failed == 0 else 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="data/train_platform")
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--start-idx", type=int, default=0)
    ap.add_argument("--platform-root", default=DEFAULT_PLATFORM_ROOT)
    ap.add_argument("--templates-dir", default=None,
                    help="模板 YAML 目录（默认 <platform-root>/templates）")
    ap.add_argument("--num-curves", type=int, default=-1,
                    help="强制曲线数 1..5（1 = 可直接评估的单曲线集）；"
                         "-1 = 按模板概率分布采样")
    ap.add_argument("--templates", default="",
                    help="逗号分隔的模板 id 子集（默认全部）")
    ap.add_argument("--no-log", action="store_true", help="禁用对数轴扩展")
    ap.add_argument("--no-degrade", action="store_true", help="不做图像退化")
    ap.add_argument("--hard-crossing", type=float, default=0.0,
                    help="P2: 曲线交叉/贴近困难样本概率 (0-1, 如 0.4)")
    ap.add_argument("--right-fade", type=float, default=0.0,
                    help="E1: 右端淡出退化概率 (0-1; 训练模型在右端低置信区保持链)")
    ap.add_argument("--yolo", action="store_true", help="额外输出 YOLOv8 标签")
    args = ap.parse_args(argv)
    args.allow_log = not args.no_log
    args.degrade = not args.no_degrade
    if not args.templates_dir:
        args.templates_dir = os.path.join(args.platform_root, "templates")
    return generate(args)


if __name__ == "__main__":
    sys.exit(main())
