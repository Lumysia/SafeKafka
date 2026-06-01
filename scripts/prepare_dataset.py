"""Turn the downloaded Voxel51 Safe_and_Unsafe_Behaviours clips into a
YOLOv8-ready dataset on disk.

What it does:
  1. Loads the (already cached) FiftyOne dataset
  2. Extracts the *first frame* of every clip as a JPG
  3. Writes a YOLO-format label file per image (one full-frame box tagged
     with the clip's class — same approach as the project notebook)
  4. Splits 70 / 20 / 10 into train / val / test
  5. Emits dataset.yaml that the YOLOv8 trainer accepts

Usage:
    python -m scripts.prepare_dataset
    python -m scripts.prepare_dataset --out my_yolo_dataset --max-samples 100
"""
from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
from pathlib import Path

import cv2

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
logger = logging.getLogger("prepare_dataset")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="Voxel51/Safe_and_Unsafe_Behaviours",
                   help="Hugging Face dataset id (must already be downloaded)")
    p.add_argument("--out", default="yolo_dataset",
                   help="Output directory for the YOLO dataset")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Optional cap on clips to use (faster iterations)")
    p.add_argument("--train", type=float, default=0.70)
    p.add_argument("--val", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    try:
        import fiftyone as fo
        from fiftyone.utils.huggingface import load_from_hub
    except ImportError:
        logger.error("FiftyOne is not installed. Run: pip install fiftyone")
        return 2

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    logger.info("Loading dataset %s ...", args.name)
    kwargs = {}
    if args.max_samples:
        kwargs["max_samples"] = args.max_samples
    ds = load_from_hub(args.name, **kwargs)
    logger.info("Got %d clips.", len(ds))

    # ------------------------------------------------------------------
    # 2. Collect the label vocabulary
    # ------------------------------------------------------------------
    label_field = "ground_truth"
    classes = sorted({
        s[label_field].label
        for s in ds
        if getattr(s, label_field, None) is not None
        and hasattr(s[label_field], "label")
    })
    cls_to_id = {c: i for i, c in enumerate(classes)}
    logger.info("Classes (%d):", len(classes))
    for c, i in cls_to_id.items():
        logger.info("  [%2d] %s", i, c)

    # ------------------------------------------------------------------
    # 3. Output layout
    # ------------------------------------------------------------------
    out = Path(args.out).expanduser().resolve()
    if out.exists():
        logger.info("Removing existing output dir: %s", out)
        shutil.rmtree(out)
    for split in ("train", "val", "test"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 4. Deterministic shuffle + split
    # ------------------------------------------------------------------
    samples = list(ds)
    rnd = random.Random(args.seed)
    rnd.shuffle(samples)

    n = len(samples)
    n_train = int(args.train * n)
    n_val = int(args.val * n)
    splits = {
        "train": samples[:n_train],
        "val":   samples[n_train:n_train + n_val],
        "test":  samples[n_train + n_val:],
    }
    for sp, items in splits.items():
        logger.info("Split %s: %d clips", sp, len(items))

    # ------------------------------------------------------------------
    # 5. Extract first frame + write YOLO label
    # ------------------------------------------------------------------
    # A YOLOv8 label line is:
    #   <class_id> <cx> <cy> <w> <h>   (all in [0,1], normalized)
    # We use a single full-frame box because this dataset only has
    # clip-level Classification labels (no bounding boxes). YOLO trained
    # this way learns the dominant behaviour of the frame as a whole.
    full_frame_box = " 0.5 0.5 1.0 1.0"

    written = 0
    skipped = 0
    for sp, items in splits.items():
        img_dir = out / sp / "images"
        lbl_dir = out / sp / "labels"
        for s in items:
            gt = getattr(s, label_field, None)
            if gt is None or not hasattr(gt, "label"):
                skipped += 1
                continue
            cid = cls_to_id.get(gt.label)
            if cid is None:
                skipped += 1
                continue

            cap = cv2.VideoCapture(s.filepath)
            ok, frame_bgr = cap.read()
            cap.release()
            if not ok or frame_bgr is None:
                skipped += 1
                continue

            stem = Path(s.filepath).stem
            img_path = img_dir / f"{stem}.jpg"
            lbl_path = lbl_dir / f"{stem}.txt"
            cv2.imwrite(str(img_path), frame_bgr)
            lbl_path.write_text(f"{cid}{full_frame_box}\n")
            written += 1

    logger.info("Wrote %d image/label pairs (skipped %d).", written, skipped)

    # ------------------------------------------------------------------
    # 6. dataset.yaml
    # ------------------------------------------------------------------
    yaml_lines = [
        f"# SafeStream-Kafka — auto-generated by scripts/prepare_dataset.py",
        f"path: {out}",
        "train: train/images",
        "val:   val/images",
        "test:  test/images",
        "",
        f"nc: {len(classes)}",
        f"names: {classes}",
        "",
    ]
    yaml_path = out / "dataset.yaml"
    yaml_path.write_text("\n".join(yaml_lines))

    print()
    print("=" * 72)
    print(f"YOLO dataset ready at: {out}")
    print(f"dataset.yaml         : {yaml_path}")
    print()
    print("Next — train YOLOv8 on your Mac M1 (uses MPS automatically):")
    print()
    print(f'  python -m scripts.train_yolo --data "{yaml_path}" \\')
    print(f'      --epochs 50 --imgsz 640 --batch 16 --device auto')
    print()
    print("Then use the trained weights with the detector:")
    print()
    print('  python -m safestream.detector \\')
    print('      --weights runs/safestream_yolov8m/weights/best.pt')
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
