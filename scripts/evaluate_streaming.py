"""Streaming evaluation: does smarter aggregation actually beat the naive one?

Replays each held-out clip frame-by-frame through a per-frame inference backend
(``--model-type temporal`` = a TemporalInfer checkpoint, the default;
``--model-type yolo`` = the per-frame YOLOv8 baseline) and feeds the resulting
per-frame messages into TWO aggregators built from the same stream:

    naive  : SafetyAggregator(use_prob=False)  -> count-ratio + single threshold
    smart  : SafetyAggregator(use_prob=True)   -> EWMA + hysteresis on unsafe_prob

For each clip we record whether each aggregator raised an alert and, for unsafe
clips, how many frames it took (detection latency). Reported per aggregator:

    alert precision / recall   (clip flagged unsafe == any alert fired)
    false-alert rate           (fraction of SAFE clips that fired)
    mean detection latency     (frames-to-first-alert on unsafe clips)

This is the headline figure: smart aggregation should cut the false-alert rate
at equal/again recall and lower the latency. No Kafka required — pure in-process.

Per-model threshold calibration
-------------------------------
Different models emit ``unsafe_prob`` on different scales, so a single global
``enter_threshold`` (from settings) is not apples-to-apples across models. Pass
``--calibrate-split val`` to first sweep the smart aggregator's enter/exit
thresholds on a held-out split and pick a per-model operating point (default:
the lowest false-alert rate among thresholds reaching alert-recall ≥ 0.95), then
report on ``--split`` (test) at that calibrated point. Only the smart
aggregator's enter/exit thresholds move; every other aggregator parameter stays
fixed. With no calibration flags the behaviour is identical to before.
"""
from __future__ import annotations

import argparse
import json
import sys
from statistics import mean
from typing import Callable, Dict, List, Optional

from safestream.aggregator.aggregator import SafetyAggregator
from safestream.common.encoding import detect_device
from safestream.common.labels import classify_label
from safestream.detector.temporal import TemporalInfer
from safestream.settings import get_settings
from safestream.temporal.dataset import read_manifest

from scripts.prepare_dataset import sample_frames, _norm_class


class YoloFrameInfer:
    """Per-frame YOLOv8 classifier exposing the same ``.infer()`` contract as
    :class:`~safestream.detector.temporal.TemporalInfer`, so the per-frame YOLO
    baseline can be replayed through the identical streaming pipeline.

    ``unsafe_prob`` is the softmax mass over classes whose name maps to *unsafe*
    (``classify_label``) — exactly how ``scripts/evaluate.py`` computes it — so the
    baseline's streaming row is comparable to its offline metrics and to the
    temporal models. There is no sliding window: each frame is classified
    independently, so ``camera_id`` is ignored. ultralytics/torch are imported
    lazily here, mirroring the detector.
    """

    def __init__(self, weights: str, device: str = "cpu"):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        try:
            self.model.to(device)
        except Exception:
            pass
        self.names = self.model.names

    def infer(self, camera_id: str, frame_bgr) -> tuple:
        preds = self.model(frame_bgr, verbose=False)
        probs = getattr(preds[0], "probs", None) if preds else None
        if probs is None:
            return "unknown", "other", 0.0, 0.0
        top = int(probs.top1)
        label = self.names.get(top, str(top))
        conf = float(probs.top1conf)
        try:
            vec = [float(x) for x in probs.data.tolist()]
        except Exception:
            vec = []
        u_prob = sum(
            p for cid, p in enumerate(vec)
            if classify_label(self.names.get(cid, str(cid))) == "unsafe"
        )
        return label, classify_label(label), conf, float(u_prob)


def _make_aggregators(s, *, enter_threshold: Optional[float] = None,
                      exit_threshold: Optional[float] = None) -> Dict[str, SafetyAggregator]:
    """Build the {naive, smart} aggregator pair.

    When ``enter_threshold``/``exit_threshold`` are None this is identical to the
    historical behaviour (thresholds from settings). When set, *only* the smart
    aggregator's enter/exit thresholds change; every other parameter — including
    the naive aggregator — comes from settings unchanged.
    """
    common = dict(
        window_seconds=s.agg_window_seconds,
        unsafe_ratio_alert=s.agg_unsafe_ratio_alert,
        min_window_obs=s.agg_min_window_obs,
        alert_cooldown_seconds=s.agg_alert_cooldown_seconds,
    )
    enter = s.agg_enter_threshold if enter_threshold is None else float(enter_threshold)
    exit_t = s.agg_exit_threshold if exit_threshold is None else float(exit_threshold)
    return {
        "naive": SafetyAggregator(use_prob=False, **common),
        "smart": SafetyAggregator(
            use_prob=True,
            ewma_halflife=s.agg_ewma_halflife,
            enter_threshold=enter,
            exit_threshold=exit_t,
            min_dwell=s.agg_min_dwell,
            **common,
        ),
    }


def _summarise(name: str, fired: List[bool], truth: List[bool],
               latency: List[Optional[int]]) -> Dict:
    tp = sum(f and t for f, t in zip(fired, truth))
    fp = sum(f and not t for f, t in zip(fired, truth))
    fn = sum((not f) and t for f, t in zip(fired, truth))
    n_safe = sum(not t for t in truth)
    fa_rate = fp / n_safe if n_safe else None
    lat = [l for f, t, l in zip(fired, truth, latency) if f and t and l is not None]
    return {
        "aggregator": name,
        "alert_precision": tp / (tp + fp) if (tp + fp) else None,
        "alert_recall": tp / (tp + fn) if (tp + fn) else None,
        "false_alert_rate": fa_rate,
        "mean_detection_latency_frames": mean(lat) if lat else None,
        "confusion_clips": {"tp": tp, "fp": fp, "fn": fn},
    }


def _replay_split(model, rows, *, frames_per_clip: int,
                  fps: float, cam_prefix: str = "clip") -> List[Dict]:
    """Run inference once over every *scorable* clip in a split.

    ``model`` is any object with a ``TemporalInfer``-style
    ``.infer(cam, frame) -> (label, cat, conf, unsafe_prob)`` method (the temporal
    model or :class:`YoloFrameInfer` for the per-frame baseline).

    Returns one record per safe/unsafe clip (clips classifying as "other" are
    skipped, exactly as the inline replay did):

        {video_path, cam, truth_unsafe, frames:[{timestamp, unsafe_prob, cat,
                                                 label, conf}]}

    ``cam=f"{cam_prefix}-{ci}"`` and ``base_ts=ci*10_000`` reproduce the inline
    replay's per-clip key + time isolation, so feeding these cached sequences
    through the aggregators is byte-identical to the original loop. ``ci`` is the
    enumerate index over *all* rows (including skipped "other" clips), matching
    the historical numbering. The ``cam_prefix`` lets the calibration (val) and
    reporting (test) replays run on the same TemporalInfer without their
    per-camera frame buffers carrying over between passes.
    """
    dt = 1.0 / fps
    clips: List[Dict] = []
    for ci, (video_path, label) in enumerate(rows):
        cat0 = classify_label(_norm_class(label))
        if cat0 == "other":
            continue  # only safe/unsafe clips are scorable
        truth_unsafe = cat0 == "unsafe"
        cam = f"{cam_prefix}-{ci}"  # fresh key -> per-clip buffer + aggregator state
        base_ts = ci * 10_000.0     # keep clips far apart on the time axis
        frames = sample_frames(video_path, frames_per_clip)
        seq: List[Dict] = []
        for fi, frame in enumerate(frames):
            label_t, cat, conf, u_prob = model.infer(cam, frame)
            seq.append({
                "timestamp": base_ts + fi * dt,
                "unsafe_prob": u_prob,
                "cat": cat,
                "label": label_t,
                "conf": conf,
            })
        clips.append({
            "video_path": video_path,
            "cam": cam,
            "truth_unsafe": truth_unsafe,
            "frames": seq,
        })
        print(f"[{len(clips)}] {video_path}  "
              f"truth={'UNSAFE' if truth_unsafe else 'safe'}  frames={len(seq)}")
    return clips


def _score_sequences(clips: List[Dict],
                     aggregators_factory: Callable[[], Dict[str, SafetyAggregator]]
                     ) -> List[Dict]:
    """Feed cached per-clip sequences through a fresh aggregator pair and
    summarise. ``aggregators_factory`` must return a fresh {naive, smart} dict
    each call: state is per-camera and the clip keys repeat across sweep passes,
    so each pass needs its own aggregators.
    """
    aggs = aggregators_factory()
    fired: Dict[str, List[bool]] = {k: [] for k in aggs}
    latency: Dict[str, List[Optional[int]]] = {k: [] for k in aggs}
    truth: List[bool] = []

    for clip in clips:
        truth.append(clip["truth_unsafe"])
        cam = clip["cam"]
        first: Dict[str, Optional[int]] = {k: None for k in aggs}
        for fi, fr in enumerate(clip["frames"]):
            msg = {
                "camera_id": cam,
                "timestamp": fr["timestamp"],
                "safe_count": 1 if fr["cat"] == "safe" else 0,
                "unsafe_count": 1 if fr["cat"] == "unsafe" else 0,
                "unsafe_prob": fr["unsafe_prob"],
                "detections": [{"label": fr["label"], "category": fr["cat"],
                                "conf": fr["conf"]}],
            }
            for k, agg in aggs.items():
                _, alert = agg.update(msg)
                if alert is not None and first[k] is None:
                    first[k] = fi
        for k in aggs:
            fired[k].append(first[k] is not None)
            latency[k].append(first[k])

    return [_summarise(k, fired[k], truth, latency[k]) for k in aggs]


def _build_grid(step: float) -> List[float]:
    """Candidate enter-thresholds, excluding the degenerate 0.0 / 1.0 endpoints
    (always-fire / never-fire)."""
    if not (0.0 < step < 1.0):
        raise ValueError(f"--cal-grid-step must be in (0, 1), got {step}")
    n = int(round(1.0 / step))
    return [round(i * step, 6) for i in range(1, n)]


def _sweep_thresholds(cal_clips: List[Dict], s, *, grid: List[float],
                      objective: str, target_recall: float,
                      exit_ratio: float) -> Dict:
    """Sweep smart enter-thresholds over ``grid`` on the calibration clips and
    pick an operating point.

    objective ``target-recall``: among thresholds whose alert-recall ≥
    ``target_recall``, choose the lowest false-alert rate, tie-broken by the
    highest threshold. If none reach the target, fall back to the highest-recall
    threshold (lowest FA / highest threshold) and set ``fallback_used``.

    Returns ``{chosen:{enter_threshold, exit_threshold}, val_operating_point,
    sweep:[...], fallback_used}``.
    """
    if objective != "target-recall":
        raise ValueError(f"unknown --cal-objective {objective!r}")

    sweep: List[Dict] = []
    for enter in grid:
        exit_t = max(0.0, min(enter * exit_ratio, enter))  # clamp 0 <= exit <= enter
        rows = _score_sequences(
            cal_clips,
            lambda e=enter, x=exit_t: _make_aggregators(s, enter_threshold=e,
                                                        exit_threshold=x),
        )
        smart = next(r for r in rows if r["aggregator"] == "smart")
        sweep.append({
            "enter_threshold": enter,
            "exit_threshold": exit_t,
            "alert_recall": smart["alert_recall"],
            "false_alert_rate": smart["false_alert_rate"],
            "alert_precision": smart["alert_precision"],
        })

    def _key(r):  # lowest FA, tie-break highest enter_threshold
        fa = r["false_alert_rate"]
        return (fa if fa is not None else float("inf"), -r["enter_threshold"])

    qualifying = [r for r in sweep
                  if r["alert_recall"] is not None and r["alert_recall"] >= target_recall]
    fallback_used = not qualifying
    if qualifying:
        chosen = min(qualifying, key=_key)
    else:
        recalls = [r["alert_recall"] for r in sweep if r["alert_recall"] is not None]
        max_recall = max(recalls) if recalls else None
        pool = ([r for r in sweep if r["alert_recall"] == max_recall]
                if max_recall is not None else list(sweep))
        chosen = min(pool, key=_key)

    return {
        "chosen": {
            "enter_threshold": chosen["enter_threshold"],
            "exit_threshold": chosen["exit_threshold"],
        },
        "val_operating_point": dict(chosen),
        "sweep": sweep,
        "fallback_used": fallback_used,
    }


def main() -> int:
    s = get_settings()
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="runs/temporal/best.pt")
    p.add_argument("--model-type", default="temporal", choices=["temporal", "yolo"],
                   help="temporal = TemporalInfer checkpoint; yolo = per-frame "
                        "YOLOv8 classification weights (the baseline).")
    p.add_argument("--data", default="clips_manifest.csv")
    p.add_argument("--split", default="test")
    p.add_argument("--frames-per-clip", type=int, default=64,
                   help="Frames streamed per clip (evenly sampled)")
    p.add_argument("--fps", type=float, default=4.0,
                   help="Synthetic frame rate -> message timestamps / EWMA decay")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="eval_streaming.json")
    # --- per-model calibration (optional; absent => historical behaviour) -----
    p.add_argument("--calibrate-split", default=None,
                   help="Held-out split to calibrate the smart enter/exit "
                        "thresholds on (e.g. val). When unset, settings "
                        "thresholds are used and output is unchanged.")
    p.add_argument("--cal-objective", default="target-recall",
                   choices=["target-recall"],
                   help="Threshold-selection objective.")
    p.add_argument("--cal-target-recall", type=float, default=0.95,
                   help="Minimum alert-recall a threshold must reach to qualify.")
    p.add_argument("--cal-exit-ratio", type=float, default=0.6,
                   help="exit_threshold = enter_threshold * this ratio.")
    p.add_argument("--cal-grid-step", type=float, default=0.05,
                   help="Enter-threshold sweep step over (0, 1).")
    args = p.parse_args()

    device = detect_device(args.device)
    if args.model_type == "yolo":
        model = YoloFrameInfer(args.weights, device=device)
    else:
        model = TemporalInfer(args.weights, device=device)

    rows, _ = read_manifest(args.data, args.split)
    if not rows:
        print(f"No clips in split {args.split!r}", file=sys.stderr)
        return 2

    print(f"Replaying split {args.split!r} ({len(rows)} clips) ...")
    clips = _replay_split(model, rows, frames_per_clip=args.frames_per_clip,
                          fps=args.fps)

    calibration: Optional[Dict] = None
    if args.calibrate_split:
        cal_rows, _ = read_manifest(args.data, args.calibrate_split)
        if not cal_rows:
            print(f"No clips in calibration split {args.calibrate_split!r}",
                  file=sys.stderr)
            return 2
        print(f"Replaying calibration split {args.calibrate_split!r} "
              f"({len(cal_rows)} clips) ...")
        cal_clips = _replay_split(model, cal_rows,
                                  frames_per_clip=args.frames_per_clip,
                                  fps=args.fps, cam_prefix="cal")
        sweep = _sweep_thresholds(
            cal_clips, s, grid=_build_grid(args.cal_grid_step),
            objective=args.cal_objective, target_recall=args.cal_target_recall,
            exit_ratio=args.cal_exit_ratio,
        )
        enter = sweep["chosen"]["enter_threshold"]
        exit_t = sweep["chosen"]["exit_threshold"]
        vop = sweep["val_operating_point"]
        if sweep["fallback_used"]:
            print(f"WARNING: no threshold reached recall {args.cal_target_recall} "
                  f"on {args.calibrate_split!r}; fell back to max-recall / "
                  f"lowest-FA (enter={enter}).", file=sys.stderr)
        print(f"Calibrated smart thresholds on {args.calibrate_split!r}: "
              f"enter={enter} exit={exit_t}  "
              f"(val recall={vop['alert_recall']}, FA={vop['false_alert_rate']})")
        results = _score_sequences(
            clips,
            lambda: _make_aggregators(s, enter_threshold=enter, exit_threshold=exit_t),
        )
        calibration = {
            "calibrate_split": args.calibrate_split,
            "objective": args.cal_objective,
            "target_recall": args.cal_target_recall,
            "exit_ratio": args.cal_exit_ratio,
            "chosen": sweep["chosen"],
            "val_operating_point": vop,
            "fallback_used": sweep["fallback_used"],
            "sweep": sweep["sweep"],
        }
    else:
        results = _score_sequences(clips, lambda: _make_aggregators(s))

    report: Dict = {
        "clips_scored": len(clips),
        "fps": args.fps,
        "results": results,
    }
    if calibration is not None:
        report["calibration"] = calibration

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\n" + json.dumps(report["results"], indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
