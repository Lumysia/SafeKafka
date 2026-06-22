"""Recompute the offline detector/classifier metrics from the shipped weights.

This reproduces the numbers the dashboard shows under `offline_detector_metrics`
so they are not just hard-coded constants. It mirrors the *exact* production
inference path used by safestream.detector (read preds[0].probs -> top1 ->
model.names -> classify_label), so eval == serving.

The dataset is the YOLO classification tree produced by scripts/prepare_dataset.py:

    <data>/<split>/<class_dir>/*.jpg

The class directory name is the ground-truth 8-class label. Binary truth (and the
binary prediction) come from safestream.common.labels.classify_label, the same
mapping the detector and dashboard use, so a class is scored only if it maps to
'safe' or 'unsafe' (classes mapping to 'other' are skipped).

Usage:
    python -m scripts.evaluate --weights previous_weights/best.pt --data yolo_dataset
    python -m scripts.evaluate --data yolo_dataset --split test --out eval_results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from safestream.common.encoding import detect_device
from safestream.common.labels import classify_label
from safestream.settings import get_settings

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
logger = logging.getLogger("evaluate")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _canon(name: str) -> str:
    """Canonical 8-class key for identity comparison: drop a leading 'N_' index
    prefix and lowercase, so the model's 'Safe_Walkway_Violation' matches the
    dataset directory '0_safe_walkway_violation'. (Binary safe/unsafe mapping
    still goes through classify_label on the raw name.)"""
    return re.sub(r"^\d+_", "", name.strip()).lower()


def _iter_images(split_dir: Path):
    """Yield (image_path, class_dir_name) for every image in the split tree."""
    for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for img in sorted(cls_dir.iterdir()):
            if img.suffix.lower() in IMG_EXTS:
                yield img, cls_dir.name


def _binary_metrics(tp: int, tn: int, fp: int, fn: int) -> Dict[str, Optional[float]]:
    """Same formulas as dashboard._metrics_payload (positive class = unsafe)."""
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall)
        else None
    )
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else None
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def _average_precisions(
    class_names: List[str],
    true_idx: List[int],
    prob_rows: List[List[float]],
    unsafe_score: List[float],
    binary_true: List[int],
) -> Dict[str, Optional[float]]:
    """Honest classification AP (PR-AUC), replacing the bogus bbox-mAP rows.

    Returns binary unsafe AP and 8-class macro one-vs-rest AP. Computed only if
    scikit-learn is importable; otherwise both are None.
    """
    try:
        import numpy as np
        from sklearn.metrics import average_precision_score
        from sklearn.preprocessing import label_binarize
    except Exception:
        logger.warning("scikit-learn not available; skipping AP metrics.")
        return {"binary_unsafe_ap": None, "macro_ap_8class": None}

    binary_ap: Optional[float] = None
    if any(binary_true) and not all(binary_true):
        binary_ap = float(average_precision_score(binary_true, unsafe_score))

    macro_ap: Optional[float] = None
    n_classes = len(class_names)
    if n_classes > 1 and len(set(true_idx)) > 1:
        y_true = label_binarize(true_idx, classes=list(range(n_classes)))
        y_score = np.asarray(prob_rows)
        if y_true.shape == y_score.shape:
            macro_ap = float(
                average_precision_score(y_true, y_score, average="macro")
            )
    return {"binary_unsafe_ap": binary_ap, "macro_ap_8class": macro_ap}


def main() -> int:
    s = get_settings()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default="previous_weights/best.pt",
                   help="Path to YOLOv8 classification weights (.pt)")
    p.add_argument("--data", default="yolo_dataset",
                   help="Classification dataset root (contains the split dirs)")
    p.add_argument("--split", default="test", help="Split subdirectory to evaluate")
    p.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    p.add_argument("--conf", type=float, default=s.detector_conf,
                   help="Confidence threshold (defaults to the detector's)")
    p.add_argument("--out", default="eval_results.json",
                   help="Where to write the metrics JSON")
    args = p.parse_args()

    split_dir = Path(args.data).expanduser().resolve() / args.split
    if not split_dir.is_dir():
        logger.error("split directory not found: %s", split_dir)
        return 2

    device = detect_device(args.device)
    logger.info("Evaluating %s on %s (device=%s, conf=%.2f)",
                args.weights, split_dir, device, args.conf)

    from ultralytics import YOLO

    model = YOLO(args.weights)
    try:
        model.to(device)
    except Exception as e:
        logger.warning("model.to(%s) failed: %s -- using default device", device, e)

    # Stable class-index ordering for the macro-AP one-vs-rest matrix. Canonical
    # names so model classes and dataset dirs share one index space.
    eval_class_names = sorted({_canon(n) for n in model.names.values()})
    cls_to_idx = {name: i for i, name in enumerate(eval_class_names)}

    tp = tn = fp = fn = 0
    eightclass_correct = 0
    eightclass_total = 0
    skipped_other = 0
    skipped_unreadable = 0
    samples = 0

    true_idx: List[int] = []
    prob_rows: List[List[float]] = []
    unsafe_score: List[float] = []
    binary_true: List[int] = []

    for img_path, true_cls_dir in _iter_images(split_dir):
        truth = classify_label(true_cls_dir)
        if truth not in {"safe", "unsafe"}:
            skipped_other += 1
            continue

        preds = model(str(img_path), conf=args.conf, verbose=False)
        probs = getattr(preds[0], "probs", None) if preds else None
        if probs is None:
            skipped_unreadable += 1
            continue

        top1conf = float(probs.top1conf)
        pred_name = model.names.get(int(probs.top1), str(int(probs.top1)))
        pred_cat = classify_label(pred_name) if top1conf >= args.conf else "other"

        # 8-class top-1 accuracy: predicted class name vs true directory name
        # (canonicalized so naming format differences don't count as misses).
        eightclass_total += 1
        if _canon(pred_name) == _canon(true_cls_dir):
            eightclass_correct += 1

        # Binary confusion (positive class = unsafe). A below-threshold/"other"
        # prediction counts as the negative outcome it most resembles: not unsafe.
        pred_unsafe = pred_cat == "unsafe"
        if truth == "unsafe" and pred_unsafe:
            tp += 1
        elif truth == "safe" and not pred_unsafe:
            tn += 1
        elif truth == "safe" and pred_unsafe:
            fp += 1
        else:  # truth unsafe, predicted not unsafe
            fn += 1

        # Score arrays for AP. probs.data is the full softmax vector.
        try:
            vec = [float(x) for x in probs.data.tolist()]
        except Exception:
            vec = []
        row = [0.0] * len(eval_class_names)
        u_score = 0.0
        for cls_id, prob in enumerate(vec):
            name = model.names.get(cls_id, str(cls_id))
            canon = _canon(name)
            if canon in cls_to_idx:
                row[cls_to_idx[canon]] = prob
            if classify_label(name) == "unsafe":
                u_score += prob
        prob_rows.append(row)
        unsafe_score.append(u_score)
        true_idx.append(cls_to_idx.get(_canon(true_cls_dir), -1))
        binary_true.append(1 if truth == "unsafe" else 0)
        samples += 1

    if samples == 0:
        logger.error("No scorable images found under %s", split_dir)
        return 2

    binary = _binary_metrics(tp, tn, fp, fn)
    ap = _average_precisions(eval_class_names, true_idx, prob_rows,
                             unsafe_score, binary_true)
    top1_8class = eightclass_correct / eightclass_total if eightclass_total else None

    result = {
        "note": (
            f"Computed with {args.weights} on {args.data}/{args.split} "
            f"({samples} sampled {args.split} frames). Classification AP "
            f"(PR-AUC), not bbox IoU mAP."
        ),
        "positive_class": "unsafe",
        "binary_unsafe_ap": ap["binary_unsafe_ap"],
        "macro_ap_8class": ap["macro_ap_8class"],
        "precision": binary["precision"],
        "recall": binary["recall"],
        "f1": binary["f1"],
        "accuracy": binary["accuracy"],
        "top1_accuracy_8class": top1_8class,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "samples": samples,
        "skipped_other_class": skipped_other,
        "skipped_unreadable": skipped_unreadable,
    }

    out_path = Path(args.out).expanduser().resolve()
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print()
    print("=" * 64)
    print(f"Offline evaluation ({samples} frames, positive class = unsafe)")
    print("-" * 64)
    print(f"  confusion        TP={tp} FP={fp} FN={fn} TN={tn}")

    def _fmt(v: Optional[float]) -> str:
        return f"{v:.3f}" if isinstance(v, float) else "n/a"

    print(f"  precision        {_fmt(binary['precision'])}")
    print(f"  recall           {_fmt(binary['recall'])}")
    print(f"  f1               {_fmt(binary['f1'])}")
    print(f"  accuracy         {_fmt(binary['accuracy'])}")
    print(f"  8-class top-1    {_fmt(top1_8class)}")
    print(f"  binary unsafe AP {_fmt(ap['binary_unsafe_ap'])}")
    print(f"  8-class macro AP {_fmt(ap['macro_ap_8class'])}")
    print("-" * 64)
    print(f"Wrote {out_path}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
