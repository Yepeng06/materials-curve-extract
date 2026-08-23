"""T1: four-signal axis-kind diagnostic on data/eval_phased_500.

Quantifies the per-signal error rate of the four candidate signals for the
log/linear axis-kind judgement, compares the existing 3-signal vote with
4-signal fusions, and writes

  data/experiments_t1/axis_kind_stats.json   per-image + aggregated stats
  data/experiments_t1/CONCLUSION.md          findings

Signals (all computed per axis, x and y, on top of detect_structure ->
read_ticks(stub), NO segmentation):

  S1 value sequence   : arithmetic vs geometric consistency of the resolved
                        tick values (reuses the production resolve_values /
                        _value_sequence_vote from axis_kind.py).
  S2 label characters : does any tick-label text carry a log/ln / 10^N /
                        scientific-notation marker (a real-world signal that
                        the synthetic stub labels never exercise).
  S3 tick geometry    : S3b = production minor-tick-density vote
                        (_pixel_spacing_vote); S3a = naive major-gap
                        constancy (rel std of valued-tick pixel gaps).
  S4 gridline spacing : gridlines detected from the image; S4_geo = naive
                        gap constancy, S4_coup = px-per-unit coupling of
                        gridline positions with tick values.

Existing 3-signal combination = judge_axis_kind (value-sequence w=1.0 +
pixel-spacing w=0.8 + title prior w=2.0, tie -> None) and the production
fit_axis decision (judge + R^2 double-fit fallback).

Usage:  python scripts/diag_axis_kind_t1.py [--data-dir data/eval_phased_500]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np

from mci.pipeline.axis_kind import (
    _pixel_spacing_vote,
    _seq_scores,
    _value_sequence_vote,
    judge_axis_kind,
    resolve_values,
)
from mci.pipeline.chart_structure import detect_structure
from mci.pipeline.coordinate_mapper import build_axes
from mci.pipeline.extractor import load_config
from mci.pipeline.tick_reader import StubOCRBackend, read_ticks
from mci.schema import AxisKind, Tick
from mci.utils import ink_mask, read_image

KIND = {AxisKind.LINEAR: "linear", AxisKind.LOG: "log", None: "abstain"}

# --- S2: label-character markers -------------------------------------------
_LOG_LIKE = re.compile(
    r"(?i)\b(log|ln)\b"          # explicit log/ln words
    r"|\^|⁻|¹|²|³|\u2070|\u00B9|\u00B2|\u00B3"  # superscript / caret
    r"|10\s*[xX×]\s*10"          # 1x10^...
    r"|(?:^|[^0-9.])[eE][+-]?[0-9]+"  # scientific e-notation
    r"|\d\s*[xX×]\s*10"          # mantissa x 10
)


def _group_runs(values: List[float], gap: int = 3) -> List[float]:
    if not values:
        return []
    centers: List[float] = []
    run = [values[0]]
    for v in values[1:]:
        if v - run[-1] <= gap:
            run.append(v)
        else:
            centers.append(float(np.mean(run)))
            run = [v]
    centers.append(float(np.mean(run)))
    return centers


# --- S1: value-sequence -----------------------------------------------------
def s1_features(ticks: List[Tick]) -> Dict:
    vals, _ = resolve_values(ticks)
    valued = [v for v in vals if v is not None]
    out: Dict = {"n_valued": len(valued), "lin_score": None, "log_score": None}
    s = _seq_scores(valued)
    if s is not None:
        out["lin_score"] = round(float(s[0]), 6)
        out["log_score"] = round(float(s[1]), 6)
    v = _value_sequence_vote(valued)
    out["vote"] = KIND[v]
    return out


# --- S2: label characters ---------------------------------------------------
def s2_features(ticks: List[Tick]) -> Dict:
    matched: List[str] = []
    for t in ticks:
        if t.text and _LOG_LIKE.search(t.text):
            matched.append(t.text)
    out: Dict = {"n_labels": len(ticks), "n_matched": len(matched),
                 "matched": matched[:5]}
    out["vote"] = "log" if matched else "abstain"
    return out


# --- S3: tick geometry ------------------------------------------------------
def s3_features(ticks: List[Tick], axis: str) -> Dict:
    out: Dict = {}
    v = _pixel_spacing_vote(ticks)
    out["density_vote"] = KIND[v]
    # naive major-gap constancy: valued ticks sorted by pixel
    vp = sorted((t.pixel, t.value) for t in ticks if t.value is not None)
    out["n_valued"] = len(vp)
    if len(vp) >= 3:
        gaps = np.diff([p for p, _ in vp])
        gaps = gaps[gaps > 1e-6]
        if len(gaps) >= 2 and float(np.mean(gaps)) > 1e-6:
            rs = float(np.std(gaps)) / float(np.mean(gaps))
            out["gap_rs"] = round(rs, 6)
            out["geo_vote"] = "linear" if rs < 0.08 else ("log" if rs > 0.15 else "abstain")
        else:
            out["gap_rs"] = None
            out["geo_vote"] = "abstain"
    else:
        out["gap_rs"] = None
        out["geo_vote"] = "abstain"
    return out


# --- S4: gridline detection + spacing ---------------------------------------
def detect_gridlines(img: np.ndarray, structure, x_ticks, y_ticks,
                     frac_thr: float = 0.35, tol_px: float = 6.0,
                     ) -> Tuple[List[float], List[float]]:
    """Detect interior gridlines; returns (vertical, horizontal) line px.

    Synthetic gridlines render SOLID at the major-tick positions (borders
    and the title zone are also solid), so ink-fraction alone cannot tell
    them apart.  A row/column is a gridline when (a) its interior ink
    fraction exceeds ``frac_thr`` and (b) it lies within ``tol_px`` of a
    tick position (valued-tick label centre or detected tick mark).  The
    tick-proximity filter removes frame borders (measured 6-8 px above the
    topmost tick), the title band and legend rows while keeping genuine
    gridlines (measured offset p99 = 5.5 px).
    """
    gray = cv2_gray(img)
    ink = ink_mask(gray).astype(np.float64)
    x0, y0, x1, y1 = structure.plot_bbox
    xa, xb = x0 + 5, x1 - 5
    ya, yb = y0 + 5, structure.x_axis_pixel - 5
    if xb - xa < 40 or yb - ya < 40:
        return [], []
    reg = ink[ya:yb + 1, xa:xb + 1]
    h, w = reg.shape
    col_f = reg.sum(axis=0) / h
    row_f = reg.sum(axis=1) / w
    x_tick_px = [t.pixel for t in x_ticks if t.value is not None] + \
                list(structure.x_ticks_px)
    y_tick_px = [t.pixel for t in y_ticks if t.value is not None] + \
                list(structure.y_ticks_px)
    vcols = [int(c) for c in np.where(col_f > frac_thr)[0] + xa
             if any(abs(c - tp) <= tol_px for tp in x_tick_px)]
    hrows = [int(r) for r in np.where(row_f > frac_thr)[0] + ya
             if any(abs(r - tp) <= tol_px for tp in y_tick_px)]
    return _group_runs(vcols), _group_runs(hrows)


def cv2_gray(img: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _coupled_score(line_px: List[float], tick_px: List[float],
                   tick_vals: List[float], tol: float) -> Tuple[Optional[float], Optional[float]]:
    """px-per-unit constancy for gridlines matched 1:1 to valued ticks.

    Gridlines are assigned to the nearest tick within ``tol`` (greedy,
    ascending distance, one tick per gridline) so a gridline at an unlabeled
    minor position cannot double-match one tick and poison the sequence.
    Returns (lin_score, log_score), smaller = more consistent; None when
    fewer than 3 distinct matched gridlines or degenerate.
    """
    lp_sorted = sorted(line_px)
    if len(lp_sorted) < 3 or len(tick_px) < 3:
        return None, None
    cand = []
    for i, lp in enumerate(lp_sorted):
        for j, tp in enumerate(tick_px):
            d = abs(lp - tp)
            if d <= tol:
                cand.append((d, i, j))
    cand.sort(key=lambda x: x[0])
    used_l: set = set()
    used_t: set = set()
    f: List[Tuple[float, float]] = []
    for _, i, j in cand:
        if i in used_l or j in used_t:
            continue
        used_l.add(i)
        used_t.add(j)
        f.append((lp_sorted[i], tick_vals[j]))
    if len(f) < 3:
        return None, None
    f.sort()
    p = np.asarray([x[0] for x in f], dtype=np.float64)
    v = np.asarray([x[1] for x in f], dtype=np.float64)
    dp = np.diff(p)
    dv = np.diff(v)
    # monotone check on values, then px-per-unit constancy
    lin = float("inf")
    if float(np.min(np.abs(dv))) > 1e-12 * max(1.0, float(np.max(np.abs(v)))) \
            and not (np.any(dv > 0) and np.any(dv < 0)):
        r = dp / np.abs(dv)
        if float(np.mean(r)) > 1e-12:
            lin = float(np.std(r)) / float(np.mean(r))
    log = float("inf")
    pos = v > 0
    if int(pos.sum()) >= 3:
        p_pos = p[pos]
        v_pos = v[pos]
        order = np.argsort(p_pos)
        ldv = np.diff(np.log10(v_pos[order]))
        ldp = np.diff(p_pos[order])
        if float(np.min(np.abs(ldv))) > 1e-9 and not (np.any(ldv > 0) and np.any(ldv < 0)):
            r = ldp / np.abs(ldv)
            if float(np.mean(r)) > 1e-12:
                log = float(np.std(r)) / float(np.mean(r))
    if not np.isfinite(lin) and not np.isfinite(log):
        return None, None
    return (lin if np.isfinite(lin) else None,
            log if np.isfinite(log) else None)


def s4_features(lines: List[float], ticks: List[Tick]) -> Dict:
    out: Dict = {"n_lines": len(lines)}
    if len(lines) < 3:
        out["gap_rs"] = None
        out["geo_vote"] = "abstain"
        out["coup_lin"] = None
        out["coup_log"] = None
        out["coup_vote"] = "abstain"
        return out
    gaps = np.diff(np.asarray(lines, dtype=np.float64))
    gaps = gaps[gaps > 1e-6]
    if len(gaps) >= 2 and float(np.mean(gaps)) > 1e-6:
        rs = float(np.std(gaps)) / float(np.mean(gaps))
        out["gap_rs"] = round(rs, 6)
        out["geo_vote"] = "linear" if rs < 0.08 else ("log" if rs > 0.15 else "abstain")
    else:
        out["gap_rs"] = None
        out["geo_vote"] = "abstain"
    vp = [(t.pixel, t.value) for t in ticks if t.value is not None]
    if len(vp) >= 3:
        med = float(np.median(np.diff(np.asarray(lines, dtype=np.float64))))
        tol = max(8.0, 0.25 * med)
        lin, log = _coupled_score(lines, [p for p, _ in vp], [v for _, v in vp], tol)
        out["coup_lin"] = None if lin is None else round(lin, 6)
        out["coup_log"] = None if log is None else round(log, 6)
        if lin is not None and log is not None:
            out["coup_vote"] = "log" if log < lin else ("linear" if lin < log else "abstain")
        elif lin is not None:
            out["coup_vote"] = "linear"
        elif log is not None:
            out["coup_vote"] = "log"
        else:
            out["coup_vote"] = "abstain"
    else:
        out["coup_lin"] = None
        out["coup_log"] = None
        out["coup_vote"] = "abstain"
    return out


# --- fusions ----------------------------------------------------------------
def vote_fusion(signals: Dict[str, str], weights: Dict[str, float]) -> Optional[str]:
    """Weighted vote over decided signals; None when no signal decided or tie."""
    w_log = w_lin = 0.0
    for name, vote in signals.items():
        if vote not in ("log", "linear"):
            continue
        w = weights.get(name, 1.0)
        if vote == "log":
            w_log += w
        else:
            w_lin += w
    if w_log == w_lin == 0.0:
        return None
    if w_log > w_lin:
        return "log"
    if w_lin > w_log:
        return "linear"
    return None


def majority_fusion(signals: Dict[str, str]) -> Optional[str]:
    return vote_fusion(signals, {k: 1.0 for k in signals})


def veto_fusion(signals: Dict[str, str]) -> Optional[str]:
    """Any decided 'log' wins; otherwise S1 decides; else None."""
    if any(v == "log" for v in signals.values()):
        return "log"
    if signals.get("S1") == "linear":
        return "linear"
    return None


def best_rule(signals: Dict[str, str], weights: Dict[str, float]) -> Optional[str]:
    return vote_fusion(signals, weights)


def accuracy(pairs: List[Tuple[str, Optional[str]]]) -> Dict:
    """pairs = (gt, vote). Error rate over decided votes + abstain share."""
    n = len(pairs)
    dec = [(g, v) for g, v in pairs if v not in (None, "abstain")
           and g in ("log", "linear")]
    n_dec = len(dec)
    err = sum(1 for g, v in dec if v != g)
    c_ll = sum(1 for g, v in dec if g == "log" and v == "linear")   # log as linear
    c_lg = sum(1 for g, v in dec if g == "linear" and v == "log")   # linear as log
    n_abstain = n - n_dec
    return {
        "n": n, "n_decided": n_dec, "n_abstain": n_abstain,
        "err_decided": err,
        "err_rate_decided": round(err / n_dec, 4) if n_dec else None,
        "acc_decided": round(1 - err / n_dec, 4) if n_dec else None,
        "acc_total": round((n_dec - err) / n, 4) if n else None,
        "conf_log_as_linear": c_ll,
        "conf_linear_as_log": c_lg,
        "abstain_rate": round(n_abstain / n, 4) if n else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/eval_phased_500")
    ap.add_argument("--out-dir", default="data/experiments_t1")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    cfg = load_config()

    images = sorted(
        p for p in os.listdir(data_dir)
        if p.endswith(".png") and not p.endswith("_mask.png") and not p.startswith("_")
    )
    if args.limit:
        images = images[: args.limit]

    t0 = time.time()
    rows: List[Dict] = []
    n_ok = n_fail = 0
    for name in images:
        stem = os.path.join(data_dir, os.path.splitext(name)[0])
        meta = json.load(open(stem + "_meta.json", encoding="utf-8"))
        gt = {"x": meta["x_kind"], "y": meta["y_kind"]}
        row: Dict = {"image": name, "gt": gt,
                     "grid": bool(meta.get("grid", False)),
                     "template": meta.get("template_id", "?"),
                     "degradations": meta.get("degradations", [])}
        try:
            img = read_image(stem + ".png")
            structure = detect_structure(img, cfg)
            ocr = StubOCRBackend(stem + "_labels.json")
            x_ticks, y_ticks = read_ticks(img, structure, ocr, cfg)
            vlines, hlines = detect_gridlines(img, structure, x_ticks, y_ticks)
            x0, y0, x1, y1 = structure.plot_bbox
            x_axis, y_axis = build_axes(
                x_ticks, y_ticks,
                x_endpoints=(float(structure.y_axis_pixel), float(x1)),
                y_endpoints=(float(y0), float(structure.x_axis_pixel)),
            )
            per_axis = {}
            for aname, tks, lines, axis_spec in (
                ("x", x_ticks, vlines, x_axis),
                ("y", y_ticks, hlines, y_axis),
            ):
                s1 = s1_features(tks)
                s2 = s2_features(tks)
                s3 = s3_features(tks, aname)
                s4 = s4_features(lines, tks)
                judge_k, _ = judge_axis_kind(tks)
                sig = {
                    "S1": s1["vote"], "S2": s2["vote"],
                    "S3": s3["density_vote"], "S4": s4["coup_vote"],
                }
                sig_geo = {
                    "S1": s1["vote"], "S2": s2["vote"],
                    "S3": s3["geo_vote"], "S4": s4["geo_vote"],
                }
                per_axis[aname] = {
                    "s1": s1, "s2": s2, "s3": s3, "s4": s4,
                    "judge3": KIND[judge_k],
                    "fit_axis": axis_spec.kind.value,
                    "fit_r2": round(axis_spec.quality, 5),
                    "sig": sig, "sig_geo": sig_geo,
                    "n_ticks": len(tks),
                    "maj3": majority_fusion({"S1": s1["vote"], "S3": s3["density_vote"]}),
                    "maj4": majority_fusion(sig),
                    "w4": vote_fusion(sig, {"S1": 1.0, "S2": 0.6, "S3": 0.8, "S4": 0.8}),
                    "veto": veto_fusion(sig),
                    "gt": gt[aname],
                    "err": {
                        n: (v not in (None, "abstain") and v != gt[aname])
                        for n, v in (("S1", s1["vote"]), ("S3", s3["density_vote"]),
                                     ("S4", s4["coup_vote"]),
                                     ("judge3", KIND[judge_k]),
                                     ("fit_axis", axis_spec.kind.value),
                                     ("maj4", majority_fusion(sig)),
                                     ("w4", vote_fusion(sig, {"S1": 1.0, "S2": 0.6, "S3": 0.8, "S4": 0.8})))
                    },
                }
            row["axes"] = per_axis
            row["n_vlines"] = len(vlines)
            row["n_hlines"] = len(hlines)
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {e}"
            n_fail += 1
        rows.append(row)

    # ---------------- aggregation ----------------
    signal_names = {
        "S1": "value-sequence", "S2": "label-chars",
        "S3": "pixel-spacing(minor-density)", "S4": "gridline-spacing(coupled)",
        "S3a": "major-gap-geo", "S4a": "gridline-geo",
        "judge3": "existing-3signal-vote", "fit_axis": "production fit_axis",
        "maj3": "fusion maj(S1,S3)", "maj4": "fusion maj4",
        "w4": "fusion w4", "veto": "fusion veto",
    }

    def collect(sig_fn) -> Dict:
        pairs = {"x": [], "y": []}
        for r in rows:
            if "axes" not in r:
                continue
            for ax in ("x", "y"):
                pairs[ax].append((r["gt"][ax], sig_fn(r["axes"][ax])))
        return pairs

    def pairs_of(ax, key):
        return [(r["gt"][ax], r["axes"][ax].get(key)) for r in rows if "axes" in r]

    def sig_vote(axrec, name):
        """Vote string for a named signal from a per-axis record."""
        if name == "S1":
            return axrec["s1"]["vote"]
        if name == "S2":
            return axrec["s2"]["vote"]
        if name == "S3":
            return axrec["s3"]["density_vote"]
        if name == "S3a":
            return axrec["s3"]["geo_vote"]
        if name == "S4":
            return axrec["s4"]["coup_vote"]
        if name == "S4a":
            return axrec["s4"]["geo_vote"]
        return axrec.get(name)

    summary: Dict = {"per_signal": {}, "fusions": {}, "stratified": {}}
    axes_pairs = {
        name: (lambda a, n=name: [(r["gt"][a], sig_vote(r["axes"][a], n))
                                  for r in rows if "axes" in r])
        for name in ("S1", "S2", "S3", "S3a", "S4", "S4a",
                     "judge3", "fit_axis", "maj3", "maj4", "w4", "veto")
    }
    for name in axes_pairs:
        entry = {}
        for ax in ("x", "y", "both"):
            pairs = axes_pairs[name](ax) if ax != "both" else axes_pairs[name]("x") + axes_pairs[name]("y")
            entry[ax] = accuracy(pairs)
        summary["per_signal"][name] = {"label": signal_names[name], **entry}
    summary["fusions"] = {
        name: {"label": signal_names[name], **summary["per_signal"][name]}
        for name in ("judge3", "fit_axis", "maj3", "maj4", "w4", "veto")
    }

    # stratification by gt axis combination (log-log / linear combos)
    for combo in ("linear/linear", "linear/log", "log/linear", "log/log"):
        xk, yk = combo.split("/")
        sub = [r for r in rows if "axes" in r and r["gt"]["x"] == xk and r["gt"]["y"] == yk]
        entry: Dict = {"n": len(sub)}
        for name in ("S1", "S2", "S3", "S3a", "S4", "S4a", "judge3", "fit_axis", "maj4", "veto"):
            for ax, kind in (("x", xk), ("y", yk)):
                pairs = [(r["gt"][ax], sig_vote(r["axes"][ax], name)) for r in sub]
                entry[f"{name}_{ax}"] = accuracy(pairs)
        summary["stratified"][combo] = entry

    # gridlines: detection sanity (grid flag vs detected)
    det_v = sum(1 for r in rows if "axes" in r and r["n_vlines"] > 0)
    det_h = sum(1 for r in rows if "axes" in r and r["n_hlines"] > 0)
    grid_true = [r for r in rows if "axes" in r and r["grid"]]
    grid_false = [r for r in rows if "axes" in r and not r["grid"]]
    summary["gridline_detection"] = {
        "n": len(rows), "n_ok": n_ok, "n_fail": n_fail,
        "grid_true": len(grid_true),
        "vlines_detected_any": det_v,
        "hlines_detected_any": det_h,
        "grid_true_h_detected": sum(1 for r in grid_true if r["n_hlines"] > 0),
        "grid_true_v_detected": sum(1 for r in grid_true if r["n_vlines"] > 0),
        "grid_false_h_detected": sum(1 for r in grid_false if r["n_hlines"] > 0),
        "grid_false_v_detected": sum(1 for r in grid_false if r["n_vlines"] > 0),
    }

    # ---- best-rule grid search (in-sample diagnostic) ----
    weight_grid = [0.5, 1.0, 1.5, 2.0]
    all_pairs = []
    for r in rows:
        if "axes" not in r:
            continue
        for ax in ("x", "y"):
            all_pairs.append((r["gt"][ax], r["axes"][ax]["sig"]))
    best = None
    for w1 in weight_grid:
        for w2 in weight_grid:
            for w3 in weight_grid:
                for w4 in weight_grid:
                    ws = {"S1": w1, "S2": w2, "S3": w3, "S4": w4}
                    votes = [best_rule(sig, ws) for _, sig in all_pairs]
                    acc = sum(1 for (g, _), v in zip(all_pairs, votes) if v == g) / len(all_pairs)
                    if best is None or acc > best[0]:
                        best = (acc, ws)
    summary["best_rule"] = {
        "acc_total": round(best[0], 4), "weights": best[1],
        "note": "in-sample grid search over {0.5,1.0,1.5,2.0} on the same 500 images",
    }

    out_json = {
        "meta": {
            "task": "T1 four-signal axis-kind diagnostic",
            "dataset": data_dir,
            "n_images": len(images),
            "n_ok": n_ok, "n_fail": n_fail,
            "python": sys.version.split()[0],
            "elapsed_s": round(time.time() - t0, 1),
        },
        "summary": summary,
        "per_image": rows,
    }
    json_path = os.path.join(out_dir, "axis_kind_stats.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=1, ensure_ascii=False)

    # ---------------- console summary ----------------
    print("=" * 78)
    print(f"T1 four-signal axis-kind diagnostic  dataset={data_dir}  ok={n_ok} fail={n_fail}")
    print("=" * 78)
    print(f"{'signal':<34}{'axis':<6}{'err/dec':<10}{'err_rate':<10}{'log->lin':<10}{'lin->log':<10}{'abstain':<9}")
    for name in ("S1", "S2", "S3", "S3a", "S4", "S4a"):
        e = summary["per_signal"][name]
        for ax in ("x", "y", "both"):
            a = e[ax]
            print(f"{signal_names[name]:<34}{ax:<6}{a['err_decided']}/{a['n_decided']:<8}"
                  f"{a['err_rate_decided'] if a['err_rate_decided'] is not None else float('nan'):<10.4f}"
                  f"{a['conf_log_as_linear']:<10}{a['conf_linear_as_log']:<10}{a['n_abstain']:<9}")
    print("-" * 78)
    print(f"{'combination':<34}{'axis':<6}{'err/dec':<10}{'err_rate':<10}{'log->lin':<10}{'lin->log':<10}{'acc_total':<10}")
    for name in ("judge3", "fit_axis", "maj3", "maj4", "w4", "veto"):
        e = summary["per_signal"][name]
        for ax in ("x", "y", "both"):
            a = e[ax]
            print(f"{signal_names[name]:<34}{ax:<6}{a['err_decided']}/{a['n_decided']:<8}"
                  f"{a['err_rate_decided'] if a['err_rate_decided'] is not None else float('nan'):<10.4f}"
                  f"{a['conf_log_as_linear']:<10}{a['conf_linear_as_log']:<10}{a['acc_total']:<10.4f}")
    print("-" * 78)
    print("gridline detection:", json.dumps(summary["gridline_detection"], ensure_ascii=False))
    print("best in-sample rule:", json.dumps(summary["best_rule"], ensure_ascii=False))
    print("stratified (err_rate_decided, x/y):")
    for combo, e in summary["stratified"].items():
        line = []
        for name in ("S1", "S3", "S4", "fit_axis", "maj4"):
            for ax in ("x", "y"):
                a = e[f"{name}_{ax}"]
                line.append(f"{name}{ax}={a['err_rate_decided'] if a['err_rate_decided'] is not None else float('nan')}")
        print(f"  {combo:<15} n={e['n']:<4} " + " ".join(line))
    print(f"\nwrote {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
