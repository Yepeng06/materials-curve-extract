"""材料曲线工作台 — 曲线提取 + 数据合成一体化 Web。

运行（mci 环境）:
    conda activate mci
    python web/app.py                 # http://127.0.0.1:8000，自动打开浏览器
    python web/app.py --port 8010 --no-browser
"""
from __future__ import annotations

import argparse
import glob
import os
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
from mci.pipeline.extractor import Extractor  # noqa: E402
from mci.schema import ExtractionError  # noqa: E402
from mci.utils import read_image  # noqa: E402

APP_VERSION = "mci-web-2.0"
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

            PaddleOCRBackend(lang="en", device="auto")._ensure()
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
def list_examples() -> JSONResponse:
    items = []
    for base_dir, label in EXAMPLE_GLOBS:
        if not base_dir.is_dir():
            continue
        names = sorted(
            p for p in os.listdir(base_dir)
            if p.lower().endswith(tuple(ALLOWED_SUFFIXES))
            and not p.endswith("_mask.png")
        )
        for name in names[:6]:
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
                    display_name: str, run_dir: Path) -> JSONResponse:
    """共享提取逻辑：返回 summary（含下载 URL）；失败抛 HTTPException。"""
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
        write_csv(result, str(run_dir / "curves.csv"))
        write_json(result, str(run_dir / "result.json"))
        overlay_result(result, image, str(run_dir / "overlay.png"))
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
    segmenter: str = Form("unet"),
):
    if ocr not in ("stub", "paddle"):
        raise HTTPException(400, "ocr 参数须为 stub 或 paddle")
    if segmenter not in ("unet", "cv", "multi_unet"):
        raise HTTPException(400, "segmenter 参数须为 unet/cv/multi_unet")

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
                           Path(file.filename or "chart.png").name, run_dir)


@app.post("/api/extract_example")
def extract_example(
    group: str = Form(...),
    name: str = Form(...),
    ocr: str = Form("stub"),
    segmenter: str = Form("unet"),
):
    """示例图专用：直接使用数据目录原图（GT 侧车天然存在，stub 可用）。"""
    if ocr not in ("stub", "paddle"):
        raise HTTPException(400, "ocr 参数须为 stub 或 paddle")
    if segmenter not in ("unet", "cv", "multi_unet"):
        raise HTTPException(400, "segmenter 参数须为 unet/cv/multi_unet")
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
    return _run_extraction(str(src), ocr, segmenter, name, run_dir)


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
