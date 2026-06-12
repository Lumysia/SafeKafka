"""Train a YOLOv8 classifier on a YOLO classification dataset.

The dataset is a directory tree (train/<class>/*.jpg, val/<class>/*.jpg, ...)
as produced by scripts/prepare_dataset.py.

Usage:
    python -m scripts.train_yolo --data path/to/yolo_dataset --epochs 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from safestream.common.encoding import detect_device


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True,
                   help="Path to the classification dataset root directory")
    p.add_argument("--weights", default="yolov8m-cls.pt",
                   help="Starting weights / model size (a *-cls model)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=224,
                   help="224 is standard for classification; raise if you have VRAM")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="auto",
                   help="auto | mps | cuda | cpu")
    p.add_argument("--project", default="runs")
    p.add_argument("--name", default="safestream_yolov8m")
    args = p.parse_args()

    data_path = Path(args.data).expanduser().resolve()
    if not data_path.exists():
        print(f"dataset directory not found: {data_path}", file=sys.stderr)
        return 2

    device = detect_device(args.device)
    print(f"Training on device: {device}")

    from ultralytics import YOLO

    model = YOLO(args.weights)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
        patience=10,
        lr0=0.01,
        lrf=0.001,
        augment=True,
        verbose=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nDone. Best weights: {best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
