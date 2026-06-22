"""Clip-level evaluation of a trained temporal model.

Emits the *same* metric schema as scripts/evaluate.py (binary unsafe
precision/recall/f1/accuracy, binary unsafe AP, 8-class top-1) so the temporal
model is directly comparable to the per-frame YOLOv8 classifier baseline:

    # baseline (per-frame classifier)
    python -m scripts.evaluate --weights previous_weights/best.pt --data yolo_dataset

    # temporal model (this script)
    python -m scripts.evaluate_temporal --weights runs/temporal/best.pt --data clips_manifest.csv

Each test clip is scored once on a K-frame window (the unit the model was
trained on). Binary truth/prediction use safestream.common.labels.classify_label,
exactly like the baseline.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader

from safestream.common.encoding import detect_device
from safestream.common.labels import classify_label
from safestream.temporal.dataset import ClipWindowDataset
from safestream.temporal.model import load_checkpoint, unsafe_prob as _unsafe_prob
# Reuse the baseline's metric helpers so the numbers are computed identically.
from scripts.evaluate import _average_precisions, _binary_metrics

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
logger = logging.getLogger("evaluate_temporal")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="runs/temporal/best.pt")
    p.add_argument("--data", default="clips_manifest.csv")
    p.add_argument("--split", default="test")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="eval_results_temporal.json")
    args = p.parse_args()

    device = detect_device(args.device)
    model, config = load_checkpoint(args.weights, device=device)
    class_names: List[str] = config["class_names"]
    unsafe_idx = config["unsafe_idx"]
    logger.info("Loaded %s (kind=%s, %d classes) on %s",
                args.weights, config["kind"], len(class_names), device)

    ds = ClipWindowDataset(args.data, args.split, window=config["window"],
                           img_size=config["img_size"], class_names=class_names,
                           train=False)
    if len(ds) == 0:
        logger.error("No clips in split %r of %s", args.split, args.data)
        return 2
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers)

    tp = tn = fp = fn = 0
    correct8 = total8 = 0
    true_idx: List[int] = []
    prob_rows: List[List[float]] = []
    unsafe_score: List[float] = []
    binary_true: List[int] = []

    for clips, targets in loader:
        clips = clips.to(device)
        with torch.no_grad():
            probs = torch.softmax(model(clips), dim=1).cpu()
        u_scores = _unsafe_prob(probs, unsafe_idx)
        for i in range(probs.shape[0]):
            t = int(targets[i].item())
            pred = int(probs[i].argmax().item())
            truth = classify_label(class_names[t])
            pred_name = class_names[pred]
            total8 += 1
            correct8 += int(pred == t)
            pred_unsafe = classify_label(pred_name) == "unsafe"
            if truth == "unsafe" and pred_unsafe:
                tp += 1
            elif truth == "safe" and not pred_unsafe:
                tn += 1
            elif truth == "safe" and pred_unsafe:
                fp += 1
            else:
                fn += 1
            prob_rows.append([float(x) for x in probs[i].tolist()])
            unsafe_score.append(float(u_scores[i].item()))
            true_idx.append(t)
            binary_true.append(1 if truth == "unsafe" else 0)

    binary = _binary_metrics(tp, tn, fp, fn)
    ap = _average_precisions(class_names, true_idx, prob_rows, unsafe_score, binary_true)
    result = {
        "note": f"Temporal model {args.weights} on {args.data}/{args.split} "
                f"({total8} clips). Clip-level classification AP, not bbox mAP.",
        "model_kind": config["kind"],
        "positive_class": "unsafe",
        "binary_unsafe_ap": ap["binary_unsafe_ap"],
        "macro_ap_8class": ap["macro_ap_8class"],
        "precision": binary["precision"],
        "recall": binary["recall"],
        "f1": binary["f1"],
        "accuracy": binary["accuracy"],
        "top1_accuracy_8class": correct8 / total8 if total8 else None,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "clips": total8,
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
