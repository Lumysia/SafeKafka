"""SOTA ablation: temporal models vs. the per-frame YOLO baseline.

Trains and evaluates every registered temporal model *kind* under identical
conditions (same manifest splits, same frame sampling, window=16, same epochs /
optimizer, same metric code) and emits one comparison table. The heavy lifting
is delegated to the existing single-purpose scripts so they stay the single
source of truth:

    train_temporal      -> runs/ablation_<kind>/best.pt
    evaluate_temporal   -> eval_<kind>.json    (offline binary AP / F1 / top-1)
    evaluate_streaming  -> stream_<kind>.json  (alert P/R, false-alert rate, latency)

This script adds the cross-model bits the per-script tools don't: trainable/total
parameter counts, measured inference latency (ms/frame through the live
TemporalInfer path), the YOLO baseline row (read from eval_results.json), and the
aggregated ablation_results.json + ready-to-paste ablation_results.md table.

Each kind is isolated: if one fails (e.g. a Kinetics-400 weight download with no
internet), its row is recorded as "unavailable" and the rest of the table still
completes.

Usage:
    python -m scripts.ablation --data clips_manifest.csv --epochs 30
    python -m scripts.ablation --data clips_manifest.csv --smoke --models head mvit hiera

Controlled across all rows: manifest splits, sample_frames selection, window=16,
epochs, seed, AdamW, and the identical evaluate_temporal/evaluate_streaming code.
Inherent (documented, not hidden) differences: input resolution (112 for
R(2+1)D-18, 224 for the rest) and fine-tune regime (see the Regime column).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

from safestream.common.encoding import detect_device
from safestream.temporal.model import MODEL_KINDS

DISPLAY = {
    "head": "head (ResNet18+GRU)",
    "video": "video (R(2+1)D-18)",
    "mvit": "MViTv2-S",
    "swin3d": "Video Swin-T",
    "hiera": "Hiera-B",
}
REGIME = {
    "head": "frozen encoder + GRU head",
    "video": "full fine-tune",
    "mvit": "frozen backbone (linear probe)",
    "swin3d": "frozen backbone (linear probe)",
    "hiera": "frozen backbone (linear probe)",
}


def _run(cmd: List[str]) -> None:
    """Run a sub-step, streaming its output; raise on non-zero exit."""
    print("\n$ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(cmd)}")


def _time_infer(infer, device: str) -> float:
    """Wall-clock ms per frame through an ``.infer(camera_id, frame_bgr)`` wrapper.

    Identical protocol for every model (temporal or YOLO) so the latency column
    is comparable: 60 dummy BGR frames (seeded), warm up on the first 10 (which
    also fills any sliding buffer and absorbs model build / cuDNN autotune),
    ``torch.cuda.synchronize()`` on CUDA, then time ``.infer`` over the
    remaining 50 and return the per-frame mean in milliseconds.
    """
    import numpy as np
    import torch

    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (240, 320, 3), dtype=np.uint8) for _ in range(60)]
    cam = "lat-probe"
    cuda = device.startswith("cuda")
    for f in frames[:10]:  # warm-up (also fills the K-frame buffer)
        infer.infer(cam, f)
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for f in frames[10:]:
        infer.infer(cam, f)
    if cuda:
        torch.cuda.synchronize()
    n = len(frames) - 10
    return 1000.0 * (time.perf_counter() - t0) / n


def _measure(weights: str, device: str) -> Dict[str, float]:
    """Parameter counts + inference latency through the live TemporalInfer path."""
    from safestream.detector.temporal import TemporalInfer

    infer = TemporalInfer(weights, device=device)
    model = infer.model
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "params_total": int(total),
        "params_trainable": int(trainable),
        "latency_ms_per_frame": _time_infer(infer, device),
    }


def _measure_yolo(weights: str, device: str) -> float:
    """Per-frame YOLO latency through the same wrapper used for the streaming row.

    Reuses ``YoloFrameInfer`` (one ``model(frame)`` + softmax read, representative
    of the deployed detector) and the shared ``_time_infer`` protocol, so the
    baseline's latency is apples-to-apples with the temporal rows. Param counts
    are intentionally omitted: YOLO isn't trained in the ablation, so a
    trainable/total split is meaningless here."""
    from scripts.evaluate_streaming import YoloFrameInfer

    infer = YoloFrameInfer(weights, device=device)
    return _time_infer(infer, device)


def _smart_streaming(stream_json: Dict) -> Dict:
    for r in stream_json.get("results", []):
        if r.get("aggregator") == "smart":
            return r
    return {}


def _run_kind(kind: str, data: str, epochs: int, device: str, *,
              objective: str, target_recall: float, exit_ratio: float,
              reuse_weights: bool) -> Dict:
    """Train + evaluate one kind; return its result row (raises on failure).

    Streaming is evaluated with per-model threshold calibration: the smart
    aggregator's enter/exit thresholds are calibrated on the val split and
    reported on test, so the smart columns are apples-to-apples across models.
    """
    py = sys.executable
    name = f"ablation_{kind}"
    weights = str(Path("runs") / name / "best.pt")
    eval_out = f"eval_{kind}.json"
    stream_out = f"stream_{kind}.json"

    if reuse_weights and Path(weights).exists():
        print(f"(reuse-weights) using existing {weights}; skipping train_temporal")
    else:
        _run([py, "-m", "scripts.train_temporal", "--model", kind, "--name", name,
              "--window", "16", "--epochs", str(epochs), "--data", data,
              "--device", device])
    _run([py, "-m", "scripts.evaluate_temporal", "--weights", weights,
          "--data", data, "--out", eval_out, "--device", device])
    _run([py, "-m", "scripts.evaluate_streaming", "--weights", weights,
          "--data", data, "--out", stream_out, "--device", device,
          "--split", "test", "--calibrate-split", "val",
          "--cal-objective", objective,
          "--cal-target-recall", str(target_recall),
          "--cal-exit-ratio", str(exit_ratio)])

    offline = json.loads(Path(eval_out).read_text(encoding="utf-8"))
    streaming = json.loads(Path(stream_out).read_text(encoding="utf-8"))
    measured = _measure(weights, detect_device(device))

    cal = streaming.get("calibration", {})
    return {
        "model": DISPLAY.get(kind, kind),
        "kind": kind,
        "status": "ok",
        "regime": REGIME.get(kind, ""),
        "params_total": measured["params_total"],
        "params_trainable": measured["params_trainable"],
        "latency_ms_per_frame": measured["latency_ms_per_frame"],
        "offline": {
            "binary_unsafe_ap": offline.get("binary_unsafe_ap"),
            "f1": offline.get("f1"),
            "top1_accuracy_8class": offline.get("top1_accuracy_8class"),
            "macro_ap_8class": offline.get("macro_ap_8class"),
        },
        "streaming_smart": _smart_streaming(streaming),
        "streaming_cal_enter": cal.get("chosen", {}).get("enter_threshold"),
        "streaming_cal": cal,
        "weights": weights,
    }


def _baseline_row(path: str) -> Optional[Dict]:
    p = Path(path)
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return {
        "model": "YOLOv8 (per-frame)",
        "kind": "yolo",
        "status": "ok",
        "regime": "per-frame classifier",
        "params_total": None,
        "params_trainable": None,
        "latency_ms_per_frame": None,
        "offline": {
            "binary_unsafe_ap": d.get("binary_unsafe_ap"),
            "f1": d.get("f1"),
            "top1_accuracy_8class": d.get("top1_accuracy_8class"),
            "macro_ap_8class": d.get("macro_ap_8class"),
        },
        "streaming_smart": {},  # filled by _attach_baseline_streaming when weights given
        "source": str(p),
    }


def _attach_baseline_streaming(row: Dict, weights: str, data: str, device: str, *,
                               objective: str, target_recall: float,
                               exit_ratio: float) -> None:
    """Run the per-frame YOLO baseline through evaluate_streaming (calibrated on
    val, reported on test) so the baseline gets a streaming row directly
    comparable to the temporal models. Mutates ``row`` in place."""
    py = sys.executable
    out = "stream_yolo.json"
    _run([py, "-m", "scripts.evaluate_streaming", "--model-type", "yolo",
          "--weights", weights, "--data", data, "--out", out, "--device", device,
          "--split", "test", "--calibrate-split", "val",
          "--cal-objective", objective, "--cal-target-recall", str(target_recall),
          "--cal-exit-ratio", str(exit_ratio)])
    streaming = json.loads(Path(out).read_text(encoding="utf-8"))
    cal = streaming.get("calibration", {})
    row["streaming_smart"] = _smart_streaming(streaming)
    row["streaming_cal_enter"] = cal.get("chosen", {}).get("enter_threshold")
    row["streaming_cal"] = cal


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------
def _f(v, nd: int = 3) -> str:
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) and not isinstance(v, bool) else "—"


def _params_cell(row: Dict) -> str:
    t, tot = row.get("params_trainable"), row.get("params_total")
    if t is None or tot is None:
        return "—"
    return f"{t / 1e6:.2f}M / {tot / 1e6:.1f}M"


def _pr_cell(stream: Dict) -> str:
    if not stream:
        return "—"
    return f"{_f(stream.get('alert_precision'))} / {_f(stream.get('alert_recall'))}"


def _render_md(rows: List[Dict], cfg: Dict) -> str:
    header = (
        "| Model | Params (train/total) | Offline unsafe AP | F1 | 8-cls top-1 "
        "| Smart false-alert rate | Smart enter-thr (cal) | Alert P/R "
        "| Latency ms/frame | Regime |"
    )
    sep = "|" + "|".join(["---"] * 10) + "|"
    lines = [
        "# SOTA ablation — temporal safety-behaviour models",
        "",
        f"Window=16, epochs={cfg['epochs']}, data=`{cfg['data']}`"
        + ("  _(smoke run)_" if cfg["smoke"] else "")
        + ".",
        "",
        "**Controlled:** same manifest splits & `sample_frames` selection, window=16, "
        "epochs, seed, AdamW, and identical `evaluate_temporal`/`evaluate_streaming` "
        "metric code.  Smart thresholds are calibrated per model on the val split to "
        "alert-recall >= 0.95, so each smart column is that model's val operating point "
        "applied to test (apples-to-apples across models).  ",
        "**Inherent differences (see Regime):** input resolution (112 for R(2+1)D-18, "
        "224 for the rest) and fine-tune regime.",
        "",
        header,
        sep,
    ]
    for r in rows:
        if r.get("status") != "ok":
            lines.append(
                f"| {r['model']} | unavailable — {r.get('error', '')} "
                "| — | — | — | — | — | — | — | "
                f"{r.get('regime', '')} |"
            )
            continue
        off = r.get("offline", {})
        st = r.get("streaming_smart", {})
        lines.append(
            f"| {r['model']} | {_params_cell(r)} "
            f"| {_f(off.get('binary_unsafe_ap'))} "
            f"| {_f(off.get('f1'))} "
            f"| {_f(off.get('top1_accuracy_8class'))} "
            f"| {_f(st.get('false_alert_rate'))} "
            f"| {_f(r.get('streaming_cal_enter'), 2)} "
            f"| {_pr_cell(st)} "
            f"| {_f(r.get('latency_ms_per_frame'), 2)} "
            f"| {r.get('regime', '')} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="clips_manifest.csv")
    p.add_argument("--models", nargs="+", default=list(MODEL_KINDS),
                   choices=list(MODEL_KINDS))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--device", default="auto")
    p.add_argument("--baseline", default="eval_results.json",
                   help="YOLO per-frame metrics JSON to seed the baseline row")
    p.add_argument("--baseline-weights", default="previous_weights/best.pt",
                   help="YOLO weights to also run through evaluate_streaming so the "
                        "baseline gets a calibrated streaming row. Empty = skip.")
    p.add_argument("--out-json", default="ablation_results.json")
    p.add_argument("--out-md", default="ablation_results.md")
    p.add_argument("--smoke", action="store_true",
                   help="Fast end-to-end check: override epochs to 2.")
    p.add_argument("--reuse-weights", action="store_true",
                   help="Skip training for kinds whose runs/ablation_<kind>/best.pt "
                        "already exists (still re-runs eval/streaming/measure). "
                        "Regenerates the calibrated table without retraining.")
    # Per-model streaming threshold calibration (passed through to evaluate_streaming).
    p.add_argument("--cal-objective", default="target-recall",
                   choices=["target-recall"],
                   help="Smart-threshold calibration objective.")
    p.add_argument("--cal-target-recall", type=float, default=0.95,
                   help="Min alert-recall a calibrated threshold must reach on val.")
    p.add_argument("--cal-exit-ratio", type=float, default=0.6,
                   help="exit_threshold = enter_threshold * this ratio.")
    args = p.parse_args()

    epochs = 2 if args.smoke else args.epochs
    print(f"Ablation: models={args.models} epochs={epochs} data={args.data} "
          f"device={args.device} smoke={args.smoke}")

    rows: List[Dict] = []
    base = _baseline_row(args.baseline)
    if base is not None:
        if args.baseline_weights and Path(args.baseline_weights).exists():
            print("\n" + "=" * 72 + "\nyolo (per-frame baseline streaming)\n" + "=" * 72)
            try:
                _attach_baseline_streaming(
                    base, args.baseline_weights, args.data, args.device,
                    objective=args.cal_objective,
                    target_recall=args.cal_target_recall,
                    exit_ratio=args.cal_exit_ratio,
                )
            except Exception as e:  # don't let a baseline streaming failure sink the run
                traceback.print_exc()
                print(f"(baseline streaming unavailable: {e})")
            try:
                base["latency_ms_per_frame"] = _measure_yolo(
                    args.baseline_weights, detect_device(args.device))
            except Exception as e:  # latency failure shouldn't sink the run either
                traceback.print_exc()
                print(f"(baseline latency unavailable: {e})")
        rows.append(base)
    else:
        print(f"(no baseline at {args.baseline}; skipping YOLO row)")

    for kind in args.models:
        print("\n" + "=" * 72 + f"\n{kind}\n" + "=" * 72)
        try:
            rows.append(_run_kind(
                kind, args.data, epochs, args.device,
                objective=args.cal_objective,
                target_recall=args.cal_target_recall,
                exit_ratio=args.cal_exit_ratio,
                reuse_weights=args.reuse_weights,
            ))
        except Exception as e:  # isolate per-kind failures
            traceback.print_exc()
            rows.append({
                "model": DISPLAY.get(kind, kind), "kind": kind,
                "status": "unavailable", "error": str(e),
                "regime": REGIME.get(kind, ""),
            })

    cfg = {"epochs": epochs, "window": 16, "models": args.models,
           "data": args.data, "smoke": args.smoke, "device": args.device,
           "reuse_weights": args.reuse_weights,
           "cal_objective": args.cal_objective,
           "cal_target_recall": args.cal_target_recall,
           "cal_exit_ratio": args.cal_exit_ratio}
    Path(args.out_json).write_text(
        json.dumps({"config": cfg, "rows": rows}, indent=2), encoding="utf-8")
    Path(args.out_md).write_text(_render_md(rows, cfg), encoding="utf-8")

    print("\n" + _render_md(rows, cfg))
    print(f"\nWrote {args.out_json} and {args.out_md}")
    n_ok = sum(r.get("status") == "ok" and r.get("kind") != "yolo" for r in rows)
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())
