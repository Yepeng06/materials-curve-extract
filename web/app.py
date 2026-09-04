"""材料曲线工作台 — 曲线提取 + 数据合成一体化 Web。

运行（mci 环境）:
    conda activate mci
    python web/app.py                 # http://127.0.0.1:8000，自动打开浏览器
    python web/app.py --port 8010 --no-browser
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from pathlib import Path

# 项目根入路径（web/app.py -> baseline/）
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

import uvicorn  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from mci.export.csv_export import write_csv  # noqa: E402
from mci.export.json_export import write_json  # noqa: E402
from mci.export.visualizer import overlay_result  # noqa: E402
from mci.export.redraw import redraw_from_json  # noqa: E402
from mci.export.resample import resample_result_curves  # noqa: E402
from mci.pipeline.extractor import Extractor  # noqa: E402
from mci.schema import AxisKind, AxisRole, AxisSpec, Curve, ExtractionError, ExtractionResult  # noqa: E402
from mci.utils import read_image  # noqa: E402

APP_VERSION = "mci-web-2.1"
RUNS_DIR = _ROOT / "web" / "runs"
SYNTH_RUNS_DIR = _ROOT / "web" / "synthesis_runs"
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg"}
MAX_UPLOAD_MB = 20
VALID_CURVE_TYPES = {"creep", "power", "sigmoid", "relax", "logcurve"}
_TASKS: dict[str, dict] = {}
_TASKS_LOCK = threading.Lock()
# 内置示例图入库（web/examples/，带 labels.json 侧车，stub 可用）；
# 数据目录（gitignore）存在时作为补充来源。
EXAMPLE_GLOBS = [
    (_ROOT / "web" / "examples", "内置示例"),
    (_ROOT / "data" / "synthetic", "合成测试集"),
    (_ROOT / "data" / "eval_platform", "平台模板集"),
]

app = FastAPI(title="材料曲线工作台", version=APP_VERSION)
app.mount("/static", StaticFiles(directory=str(_ROOT / "web" / "static")), name="static")

_segmenter_lock = threading.Lock()
_segmenter_cache: dict = {}


def _resolve_ocr(ocr: str, img_path: str) -> tuple:
    """stub OCR 需要 GT 侧车（<stem>_labels.json）；用户上传的任意图片没有
    侧车 → 自动回退真实 OCR（PaddleOCR）并附提示。"""
    if ocr == "stub":
        labels = os.path.splitext(img_path)[0] + "_labels.json"
        if not os.path.exists(labels):
            return "paddle", ["该图没有标准答案刻度侧车（仅示例图可用），已自动切换真实 OCR（PaddleOCR）"]
    return ocr, []


def _warm_up_ocr() -> None:
    """后台预热 PaddleOCR（首次初始化 10-30s），避免首个真实 OCR 请求卡顿。"""
    def _do():
        try:
            from mci.pipeline.tick_reader import PaddleOCRBackend

            PaddleOCRBackend(lang="en", device="auto", tier="server")._ensure()
            print("[web] PaddleOCR 预热完成")
        except Exception as e:  # 预热失败不影响服务
            print(f"[web] PaddleOCR 预热失败（首次请求时会再尝试）: {type(e).__name__}: {e}")
    threading.Thread(target=_do, daemon=True).start()


def _get_extractor(ocr: str, segmenter: str) -> Extractor:
    """按请求参数构造提取器；U-Net 模型实例跨请求缓存（加载一次）。

    quality_gate=True：真实图鲁棒化分级（A/B/C + 原因码，见
    REAL_ROBUSTNESS_DESIGN.md）。管线默认关闭（合成集回归安全），但 Web
    面向任意上传图，失败必须可解释 —— 这里显式开启。
    """
    extractor = Extractor(ocr_backend=ocr, segmenter=segmenter,
                          config_override={"quality_gate": True})
    if segmenter == "unet":
        with _segmenter_lock:
            key = ("unet",
                   extractor.cfg.get("unet_checkpoint"),
                   int(extractor.cfg.get("unet_size", 512)))
            if key not in _segmenter_cache:
                from mci.pipeline.segmenter import UNetSegmenter

                _segmenter_cache[key] = UNetSegmenter(
                    checkpoint=key[1], size=key[2])
        extractor._segmenter = _segmenter_cache[key]  # 复用已加载模型
    elif segmenter == "multi_unet":
        with _segmenter_lock:
            key = ("multi",
                   extractor.cfg.get("multi_unet_checkpoint",
                                     "models/checkpoints/unet_multi_curve_evalfix.pt"),
                   int(extractor.cfg.get("unet_size", 512)))
            if key not in _segmenter_cache:
                from mci.pipeline.segmenter import MultiUNetSegmenter

                _segmenter_cache[key] = MultiUNetSegmenter(
                    checkpoint=key[1], size=key[2])
        extractor._segmenter = _segmenter_cache[key]
    elif segmenter == "auto":
        # auto: 预注入两个模型实例（跨请求缓存），extractor 内部按
        # 存活通道数自动判定单/多曲线（见 Extractor._extract_auto）。
        with _segmenter_lock:
            k_multi = ("multi",
                       extractor.cfg.get("multi_unet_checkpoint",
                                         "models/checkpoints/unet_multi_curve_evalfix.pt"),
                       int(extractor.cfg.get("unet_size", 512)))
            if k_multi not in _segmenter_cache:
                from mci.pipeline.segmenter import MultiUNetSegmenter

                _segmenter_cache[k_multi] = MultiUNetSegmenter(
                    checkpoint=k_multi[1], size=k_multi[2])
            k_single = ("unet",
                        extractor.cfg.get("unet_checkpoint"),
                        int(extractor.cfg.get("unet_size", 512)))
            if k_single not in _segmenter_cache:
                from mci.pipeline.segmenter import UNetSegmenter

                _segmenter_cache[k_single] = UNetSegmenter(
                    checkpoint=k_single[1], size=k_single[2])
        extractor._multi_segmenter = _segmenter_cache[k_multi]
        extractor._single_segmenter = _segmenter_cache[k_single]
    return extractor


def _clean_runs(max_tasks: int = 300) -> None:
    """启动时清空任务目录（演示版不留历史）；超量时删除最旧任务。"""
    if not RUNS_DIR.exists():
        return
    tasks = sorted(RUNS_DIR.iterdir(), key=lambda p: p.stat().st_mtime)
    for p in tasks[:-max_tasks] if len(tasks) > max_tasks else tasks:
        if p.is_dir():
            import shutil

            shutil.rmtree(p, ignore_errors=True)


# ---------------------------------------------------------------------------
# 页面
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(str(_ROOT / "web" / "templates" / "index.html"))


# ---------------------------------------------------------------------------
# 示例图
# ---------------------------------------------------------------------------
@app.get("/api/examples")
def list_examples(refresh: int = 0, limit: int = 6) -> JSONResponse:
    """示例图列表。

    ``refresh=1``：从每个分组全量池随机抽样（种子随时间轮换），
    实现"换一批"；默认（refresh=0）保持确定性取前 ``limit`` 张，
    兼容既有演示与测试。
    """
    items = []
    for base_dir, label in EXAMPLE_GLOBS:
        if not base_dir.is_dir():
            continue
        names = sorted(
            p for p in os.listdir(base_dir)
            if p.lower().endswith(tuple(ALLOWED_SUFFIXES))
            and not p.endswith("_mask.png")
        )
        if refresh:
            rng = random.Random(int(time.time() * 1000))
            take = min(limit, len(names))
            picked = rng.sample(names, take)
        else:
            picked = names[:limit]
        for name in picked:
            items.append({"name": name, "group": base_dir.name, "group_label": label,
                          "url": f"/api/examples/file?group={base_dir.name}&name={name}"})
    return JSONResponse({"items": items})


@app.get("/api/examples/file")
def example_file(group: str, name: str) -> FileResponse:
    for base_dir, _label in EXAMPLE_GLOBS:
        if base_dir.name != group:
            continue
        p = base_dir / name
        if p.is_file() and p.suffix.lower() in ALLOWED_SUFFIXES \
                and not name.endswith("_mask.png"):
            return FileResponse(str(p), media_type="image/png")
    raise HTTPException(404, "示例图不存在")


# ---------------------------------------------------------------------------
# 提取
# ---------------------------------------------------------------------------
def _run_extraction(image_path: str, ocr: str, segmenter: str,
                    display_name: str, run_dir: Path,
                    points: int = 0) -> JSONResponse:
    """共享提取逻辑：返回 summary（含下载 URL）；失败抛 HTTPException。

    ``points``：导出层点数（4/16/32/64/128…，0=原生密度）。只作用于 CSV
    导出与 summary 统计；result.json 恒为原生点列（导出密度与评估协议解耦，
    见 goal.md 任务 0.1）。
    """
    ocr, ocr_warnings = _resolve_ocr(ocr, image_path)
    # 原图落入 run 目录（示例图直接引用数据目录原图，复制一份保证 /image 可用）
    import shutil

    suffix = Path(image_path).suffix.lower()
    dst_img = run_dir / f"input{suffix}"
    if not dst_img.exists():
        shutil.copy(image_path, dst_img)
    try:
        image = read_image(image_path)
    except ValueError as e:
        raise HTTPException(400, f"无法读取图片: {e}")

    t0 = time.time()
    extractor = _get_extractor(ocr, segmenter)
    try:
        result = extractor.extract(image_path)
    except ExtractionError as e:
        # 结构化失败原因（quality gate）：422 detail 带原因码与人工建议
        detail = {
            "message": f"提取失败: {e}",
            "quality": getattr(e, "quality", "C"),
            "reject_code": getattr(e, "reject_code", None),
            "reject_detail": getattr(e, "reject_detail", None),
        }
        raise HTTPException(422, detail=detail)
    except Exception as e:  # 模型/环境错误 → 500
        import traceback

        traceback.print_exc()
        raise HTTPException(500, f"内部错误: {type(e).__name__}: {e}")
    elapsed = time.time() - t0

    try:
        # 导出层点数重采样（仅 CSV 与 summary 统计；result.json 恒为原生点）
        export_curves = result.curves
        if points and points > 0:
            export_curves = resample_result_curves(
                result.curves, int(points),
                x_log=(result.x_axis.kind is AxisKind.LOG))
        export_result = result if export_curves is result.curves else \
            ExtractionResult(
                image_path=result.image_path, x_axis=result.x_axis,
                y_axis=result.y_axis, curves=export_curves,
                structure=result.structure, meta=result.meta,
                warnings=result.warnings, quality=result.quality,
                status=result.status, reject_code=result.reject_code,
                reject_detail=result.reject_detail)
        write_csv(export_result, str(run_dir / "curves.csv"))
        write_json(result, str(run_dir / "result.json"))
        overlay_result(result, image, str(run_dir / "overlay.png"))
        # 记录本次运行的导出参数，供人工校正重导出时沿用
        with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
            json.dump({"points_param": int(points or 0),
                       "x_log": (result.x_axis.kind is AxisKind.LOG)}, f)
    except Exception as e:
        raise HTTPException(500, f"结果导出失败: {type(e).__name__}: {e}")

    # 重绘图是“锦上添花”产物：即使失败也不影响主提取流程
    try:
        redraw_from_json(str(run_dir / "result.json"), str(run_dir / "redraw.png"))
    except Exception as e:  # noqa: BLE001
        print(f"[warn] redraw 生成失败（不影响主结果）: {type(e).__name__}: {e}")

    tid = run_dir.name
    curves = [{
        "index": i,
        "n_points": len(c.points),
        "n_points_exported": (min(int(points), len(c.points))
                              if points and points > 0 else len(c.points)),
        "color": "#%02x%02x%02x" % tuple(c.color) if c.color else None,
        "legend_label": c.legend_label,
    } for i, c in enumerate(result.curves)]
    timings = result.meta.get("timings", {})
    summary = {
        "task_id": tid,
        "filename": display_name,
        "image_size": [int(image.shape[1]), int(image.shape[0])],
        "n_curves": len(curves),
        "curves": curves,
        "x_axis": result.x_axis.kind.value,
        "y_axis": result.y_axis.kind.value,
        "titles": {k: {kk: vv for kk, vv in v.items() if kk != "center"}
                   for k, v in result.meta.get("titles", {}).items()},
        "warnings": list(result.warnings) + ocr_warnings,
        "elapsed_s": round(elapsed, 2),
        "timings": {k: round(v, 3) for k, v in timings.items()},
        "ocr": ocr,
        "segmenter": segmenter,
        "auto_segmenter": result.meta.get("auto_segmenter"),
        "points_param": int(points or 0),
        "quality": result.quality,
        "status": result.status,
        "reject_code": result.reject_code,
        "reject_detail": result.reject_detail,
        "downloads": {
            "csv": f"/api/runs/{tid}/csv",
            "json": f"/api/runs/{tid}/json",
            "overlay": f"/api/runs/{tid}/overlay",
            "redraw": f"/api/runs/{tid}/redraw",
            "image": f"/api/runs/{tid}/image",
        },
    }
    return JSONResponse(summary)


@app.post("/api/extract")
def extract(
    file: UploadFile = File(...),
    ocr: str = Form("stub"),
    segmenter: str = Form("auto"),
    points: int = Form(0),
):
    if ocr not in ("stub", "paddle"):
        raise HTTPException(400, "ocr 参数须为 stub 或 paddle")
    if segmenter not in ("auto", "unet", "cv", "multi_unet"):
        raise HTTPException(400, "segmenter 参数须为 auto/unet/cv/multi_unet")
    if points < 0 or points > 5000:
        raise HTTPException(400, "points 须为 0（原生）或 4–5000")

    suffix = Path(file.filename or "chart.png").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"仅支持 {sorted(ALLOWED_SUFFIXES)} 图片")
    data = file.file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(400, f"图片超过 {MAX_UPLOAD_MB} MB")

    tid = uuid.uuid4().hex[:12]
    run_dir = RUNS_DIR / tid
    run_dir.mkdir(parents=True, exist_ok=True)
    img_path = run_dir / f"input{suffix}"
    img_path.write_bytes(data)
    return _run_extraction(str(img_path), ocr, segmenter,
                           Path(file.filename or "chart.png").name, run_dir,
                           points=points)


@app.post("/api/extract_example")
def extract_example(
    group: str = Form(...),
    name: str = Form(...),
    ocr: str = Form("stub"),
    segmenter: str = Form("auto"),
    points: int = Form(0),
):
    """示例图专用：直接使用数据目录原图（GT 侧车天然存在，stub 可用）。"""
    if ocr not in ("stub", "paddle"):
        raise HTTPException(400, "ocr 参数须为 stub 或 paddle")
    if segmenter not in ("auto", "unet", "cv", "multi_unet"):
        raise HTTPException(400, "segmenter 参数须为 auto/unet/cv/multi_unet")
    if points < 0 or points > 5000:
        raise HTTPException(400, "points 须为 0（原生）或 4–5000")
    src = None
    for base_dir, _label in EXAMPLE_GLOBS:
        if base_dir.name != group:
            continue
        p = base_dir / name
        if p.is_file() and p.suffix.lower() in ALLOWED_SUFFIXES \
                and not name.endswith("_mask.png"):
            src = p
            break
    if src is None:
        raise HTTPException(404, "示例图不存在")

    tid = uuid.uuid4().hex[:12]
    run_dir = RUNS_DIR / tid
    run_dir.mkdir(parents=True, exist_ok=True)
    return _run_extraction(str(src), ocr, segmenter, name, run_dir,
                           points=points)


# ---------------------------------------------------------------------------
# 结果下载
# ---------------------------------------------------------------------------
def _run_file(tid: str, name: str, media: str, attach: str | None = None):
    p = RUNS_DIR / tid / name
    if not p.is_file():
        raise HTTPException(404, "任务不存在或已清理")
    return FileResponse(str(p), media_type=media,
                        filename=attach, content_disposition_type="attachment" if attach else "inline")


@app.get("/api/runs/{tid}/image")
def run_image(tid: str):
    run_dir = RUNS_DIR / tid
    for name in ("input.png", "input.jpg", "input.jpeg"):
        p = run_dir / name
        if p.is_file():
            return FileResponse(str(p), media_type="image/png")
    raise HTTPException(404, "任务不存在或已清理")


@app.get("/api/runs/{tid}/overlay")
def run_overlay(tid: str):
    return _run_file(tid, "overlay.png", "image/png")


@app.get("/api/runs/{tid}/redraw")
def run_redraw(tid: str):
    return _run_file(tid, "redraw.png", "image/png")


@app.get("/api/runs/{tid}/csv")
def run_csv(tid: str):
    return _run_file(tid, "curves.csv", "text/csv", attach="curves.csv")


@app.get("/api/runs/{tid}/json")
def run_json(tid: str):
    return _run_file(tid, "result.json", "application/json", attach="result.json")


# ---------------------------------------------------------------------------
# 人工校正（结果卡片内嵌编辑器的后端；goal.md 任务 4.4 最小闭环）
# ---------------------------------------------------------------------------
def _axis_from_dict(d: dict, role: AxisRole) -> AxisSpec:
    """Rebuild an AxisSpec from result.json's axis dict (legacy files lack
    ``sign``; fall back to the axis convention +1 x / -1 y)."""
    kind = AxisKind(d.get("kind", "linear"))
    sign = int(d.get("sign", 1 if role is AxisRole.X else -1))
    return AxisSpec(
        role=role, kind=kind,
        slope=float(d["slope"]), intercept=float(d["intercept"]),
        vmin=float(d.get("vmin", 0.0)), vmax=float(d.get("vmax", 1.0)),
        pmin=float(d.get("pmin", 0.0)), pmax=float(d.get("pmax", 1.0)),
        sign=sign, ticks=[], quality=float(d.get("quality", 1.0)),
    )


def _pixel_to_data(axis: AxisSpec, pixel: float) -> float:
    return axis.pixel_to_value(float(pixel))


@app.post("/api/runs/{tid}/correct")
def correct_run(tid: str, payload: dict):
    """Apply manual point corrections from the in-card editor.

    Body: ``{"curves": [{"index": 0, "pixel_points": [[x, y], ...]}, ...],
    "note": optional}``

    The editor sends image-pixel anchor positions; the server converts them
    back to data coordinates with the stored axis mapping, rewrites
    ``result.json`` (points + pixel trace + correction trace), regenerates
    CSV / overlay / redraw, and appends an audit trail (goal.md E4:
    修正轨迹 JSON 落库).
    """
    run_dir = RUNS_DIR / tid
    result_path = run_dir / "result.json"
    if not result_path.is_file():
        raise HTTPException(404, "任务不存在或已清理")

    with open(result_path, encoding="utf-8") as f:
        data = json.load(f)
    try:
        x_axis = _axis_from_dict(data["x_axis"], AxisRole.X)
        y_axis = _axis_from_dict(data["y_axis"], AxisRole.Y)
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(500, f"result.json 轴映射不完整: {e}")

    edits_in = payload.get("curves") or []
    if not edits_in:
        raise HTTPException(400, "payload 缺少 curves")

    curves_out = []
    trace = []
    for item in edits_in:
        try:
            idx = int(item["index"])
        except (KeyError, TypeError, ValueError):
            raise HTTPException(400, "curves[] 项缺少 index")
        if idx < 0 or idx >= len(data["curves"]):
            raise HTTPException(400, f"曲线 index 越界: {idx}")
        px_pts = item.get("pixel_points") or []
        if len(px_pts) < 2:
            raise HTTPException(400, f"曲线 {idx} 至少需要 2 个点")

        old = data["curves"][idx]
        old_pts = old.get("points") or []
        new_pts, new_px = [], []
        for p in px_pts:
            try:
                px, py = float(p[0]), float(p[1])
            except (TypeError, ValueError, IndexError):
                raise HTTPException(400, f"曲线 {idx} 存在非法点: {p}")
            new_pts.append([_pixel_to_data(x_axis, px), _pixel_to_data(y_axis, py)])
            new_px.append([int(round(px)), int(round(py))])
        # keep x ascending (export contract) — reorder both representations
        order = sorted(range(len(new_pts)), key=lambda i: new_pts[i][0])
        new_pts = [new_pts[i] for i in order]
        new_px = [new_px[i] for i in order]

        moved = added = removed = 0
        n_old, n_new = len(old_pts), len(new_pts)
        # Audit diff in PIXEL space (ints) with a 1px tolerance: the
        # pixel->data->pixel round trip adds float noise that would flag
        # untouched points as moved.
        old_px = old.get("pixel_points") or []
        common = min(len(old_px), n_new)
        if common:
            moved = sum(
                1 for i in range(common)
                if abs(old_px[i][0] - new_px[i][0]) > 1
                or abs(old_px[i][1] - new_px[i][1]) > 1)
        added = max(0, n_new - len(old_px))
        removed = max(0, len(old_px) - n_new)

        old["points"] = new_pts
        old["pixel_points"] = new_px
        old["corrected"] = True
        curves_out.append(old)
        trace.append({
            "curve_index": idx, "name": old.get("name"),
            "before_n": n_old, "after_n": n_new,
            "moved": moved, "added": added, "removed": removed,
            "before": old_pts, "after": new_pts,
        })

    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    trace_entry = {
        "ts": ts,
        "reason_code": data.get("reject_code") or data.get("meta", {}).get("reject_code"),
        "note": str(payload.get("note") or ""),
        "operator_ops": int(sum(t["moved"] + t["added"] + t["removed"] for t in trace)),
        "edits": trace,
    }
    data.setdefault("meta", {}).setdefault("corrections", []).append(trace_entry)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    # 审计轨迹独立落盘（goal.md E4），可回放
    with open(run_dir / "corrections.json", "a", encoding="utf-8") as f:
        json.dump(trace_entry, f, ensure_ascii=False)
        f.write("\n")

    # Rebuild a minimal ExtractionResult to regenerate CSV + overlay.
    # (overlay_result only needs plot_bbox + curve pixel traces; tick
    # markers stay empty by design on corrected overlays.)
    structure = None
    if data.get("plot_bbox"):
        from mci.schema import ChartStructure

        structure = ChartStructure(plot_bbox=tuple(data["plot_bbox"]),
                                   x_axis_pixel=0, y_axis_pixel=0)
    curves_obj = [
        Curve(name=c.get("name", f"curve_{i}"),
              points=[(float(p[0]), float(p[1])) for p in c["points"]],
              pixel_points=[(int(p[0]), int(p[1])) for p in c.get("pixel_points", [])],
              color=tuple(c.get("color") or (0, 0, 0)),
              legend_label=c.get("legend_label"))
        for i, c in enumerate(curves_out)
    ]
    img_path = next((run_dir / n for n in ("input.png", "input.jpg", "input.jpeg")
                     if (run_dir / n).is_file()), None)
    # 沿用提取时用户选择的导出点数（run_meta.json 由提取端点写入；
    # 旧任务无此文件时按原生密度导出）。
    points_param = 0
    x_log = (y_axis.kind is AxisKind.LOG)
    meta_path = run_dir / "run_meta.json"
    if meta_path.is_file():
        try:
            with open(meta_path, encoding="utf-8") as f:
                rm = json.load(f)
            points_param = int(rm.get("points_param") or 0)
            x_log = bool(rm.get("x_log", x_log))
        except (ValueError, OSError):
            pass
    export_curves = curves_obj
    if points_param > 0:
        try:
            export_curves = resample_result_curves(
                curves_obj, points_param, x_log=x_log)
        except Exception:  # noqa: BLE001 — 重采样失败则回退原生密度
            export_curves = curves_obj
    rebuilt = ExtractionResult(
        image_path=data.get("image_path", ""), x_axis=x_axis, y_axis=y_axis,
        curves=export_curves, structure=structure,
        meta=data.get("meta", {}), warnings=data.get("warnings", []),
    )
    try:
        write_csv(rebuilt, str(run_dir / "curves.csv"))
        if img_path is not None:
            overlay_result(rebuilt, read_image(str(img_path)),
                           str(run_dir / "overlay.png"))
    except Exception as e:
        raise HTTPException(500, f"校正后重导出失败: {type(e).__name__}: {e}")
    try:
        redraw_from_json(str(result_path), str(run_dir / "redraw.png"))
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 校正后重绘失败（不影响主结果）: {type(e).__name__}: {e}")

    summary = {
        "task_id": tid,
        "filename": Path(data.get("image_path", "")).name or "corrected",
        "n_curves": len(curves_out),
        "curves": [{"index": i, "n_points": len(c["points"]),
                    "n_points_exported": (min(points_param, len(c["points"]))
                                          if points_param > 0 else len(c["points"])),
                    "legend_label": c.get("legend_label")}
                   for i, c in enumerate(curves_out)],
        "points_param": points_param,
        "x_axis": x_axis.kind.value, "y_axis": y_axis.kind.value,
        "quality": data.get("quality", "A"),
        "status": "corrected",
        "operator_ops": trace_entry["operator_ops"],
        "warnings": data.get("warnings", []),
        "downloads": {
            "csv": f"/api/runs/{tid}/csv",
            "json": f"/api/runs/{tid}/json",
            "overlay": f"/api/runs/{tid}/overlay",
            "redraw": f"/api/runs/{tid}/redraw",
            "image": f"/api/runs/{tid}/image",
        },
    }
    return JSONResponse(summary)


def _validate_params(params: dict) -> dict:
    out = {}
    out["count"] = max(1, min(int(params.get("count", 20)), 500))
    out["seed"] = int(params.get("seed", 20260828))
    ctypes = params.get("curve_types") or list(VALID_CURVE_TYPES)
    out["curve_types"] = [c for c in ctypes if c in VALID_CURVE_TYPES] or list(VALID_CURVE_TYPES)
    out["x_axis"] = params.get("x_axis", "both")
    out["y_axis"] = params.get("y_axis", "both")
    out["dpi"] = params.get("dpi", "random")
    out["line_style"] = params.get("line_style", "both")
    out["degrade"] = bool(params.get("degrade", True))
    nc = int(params.get("num_curves", 1))
    out["num_curves"] = max(1, min(nc, 5))
    return out


def _build_cmd(out_dir: Path, p: dict) -> list:
    cmd = [
        "scripts/gen_synthetic.py", "--out-dir", str(out_dir),
        "--count", str(p["count"]), "--seed", str(p["seed"]),
        "--curve-types", ",".join(p["curve_types"]),
        "--x-axis", p["x_axis"], "--y-axis", p["y_axis"],
        "--dpi", str(p["dpi"]), "--line-style", p["line_style"],
        "--num-curves", str(p["num_curves"]),
    ]
    if not p["degrade"]:
        cmd.append("--no-degrade")
    return cmd


def _run_subprocess(cmd: list, tid: str, out_dir: Path) -> None:
    py = sys.executable
    try:
        proc = subprocess.Popen(
            [py, *cmd], cwd=str(_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
    except Exception as e:
        with _TASKS_LOCK:
            _TASKS[tid].update(status="failed", message=f"启动失败: {e}")
        return

    target = int(_TASKS[tid]["target_count"])
    start = time.time()
    while True:
        line = proc.stdout.readline() if proc.stdout else ""
        if line:
            with _TASKS_LOCK:
                _TASKS[tid]["log"] = (_TASKS[tid].get("log", "") + line)[-4000:]
        pngs = glob.glob(str(out_dir / "*.png"))
        done = len([p for p in pngs if not p.endswith("_mask.png") and "_mask_c" not in p])
        elapsed = time.time() - start
        with _TASKS_LOCK:
            _TASKS[tid].update(progress=min(done, target), elapsed_s=round(elapsed, 1))
        if proc.poll() is not None:
            break
        time.sleep(0.5)

    remain = proc.stdout.read() if proc.stdout else ""
    if remain:
        with _TASKS_LOCK:
            _TASKS[tid]["log"] = (_TASKS[tid].get("log", "") + remain)[-4000:]
    rc = proc.returncode
    pngs = glob.glob(str(out_dir / "*.png"))
    done = len([p for p in pngs if not p.endswith("_mask.png") and "_mask_c" not in p])
    with _TASKS_LOCK:
        if rc == 0 and done > 0:
            _TASKS[tid].update(
                status="done", progress=done,
                n_images=done, message=f"生成完成：{done} 张",
            )
        else:
            tail = _TASKS[tid].get("log", "")[-800:]
            _TASKS[tid].update(
                status="failed", progress=done,
                message=f"生成失败（返回码 {rc}）。日志末尾：\n{tail}",
            )


@app.post("/api/generate")
def generate(payload: dict):
    params = _validate_params(payload.get("params", {}))
    tid = uuid.uuid4().hex[:12]
    out_dir = SYNTH_RUNS_DIR / tid
    out_dir.mkdir(parents=True, exist_ok=True)
    with _TASKS_LOCK:
        _TASKS[tid] = {
            "task_id": tid, "params": params,
            "status": "running", "progress": 0, "target_count": params["count"],
            "n_images": 0, "message": "生成中…", "elapsed_s": 0,
            "log": "", "created": time.strftime("%H:%M:%S"),
        }
    cmd = _build_cmd(out_dir, params)
    threading.Thread(target=_run_subprocess, args=(cmd, tid, out_dir),
                     daemon=True).start()
    return JSONResponse({"task_id": tid, "status": "running"})


@app.get("/api/tasks/{tid}")
def task_status(tid: str) -> JSONResponse:
    with _TASKS_LOCK:
        t = _TASKS.get(tid)
        if t is None:
            raise HTTPException(404, "任务不存在")
        return JSONResponse(dict(t))


@app.get("/api/tasks/{tid}/samples")
def task_samples(tid: str, limit: int = 12) -> JSONResponse:
    out_dir = SYNTH_RUNS_DIR / tid
    if not out_dir.is_dir():
        raise HTTPException(404, "任务不存在")
    pngs = sorted(
        p for p in out_dir.glob("*.png")
        if not p.name.endswith("_mask.png") and "_mask_c" not in p.name
    )
    items = [{"name": p.name, "url": f"/api/tasks/{tid}/image/{p.name}"}
             for p in pngs[:limit]]
    return JSONResponse({"items": items, "total": len(pngs)})


@app.get("/api/tasks/{tid}/image/{name}")
def task_image(tid: str, name: str):
    out_dir = SYNTH_RUNS_DIR / tid
    p = out_dir / name
    if not p.is_file() or ".." in name or not name.lower().endswith(".png"):
        raise HTTPException(404, "图片不存在")
    return FileResponse(str(p), media_type="image/png")


@app.get("/api/tasks/{tid}/download")
def task_download(tid: str):
    out_dir = SYNTH_RUNS_DIR / tid
    if not out_dir.is_dir():
        raise HTTPException(404, "任务不存在")
    zip_path = SYNTH_RUNS_DIR / f"{tid}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(out_dir.rglob("*")):
            if f.is_file() and f.name != zip_path.name:
                zf.write(f, f.relative_to(out_dir))
    return FileResponse(str(zip_path), media_type="application/zip",
                        filename=f"unified_data_{tid}.zip",
                        content_disposition_type="attachment")


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = ap.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    SYNTH_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    _clean_runs()
    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.5, lambda: __import__("webbrowser").open(url)).start()
    print(f"[web] {APP_VERSION} -> {url}  (Ctrl+C 退出)")
    _warm_up_ocr()  # 后台预热 PaddleOCR，避免首个真实 OCR 请求等待 10-30s
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
