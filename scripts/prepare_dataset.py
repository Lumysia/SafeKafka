"""Turn the Voxel51 Safe_and_Unsafe_Behaviours clips into a YOLOv8
*classification*-ready dataset on disk.

This dataset only has clip-level labels (the whole clip is one behaviour, no
bounding boxes), so we train YOLOv8 in classification mode rather than detection
mode. The on-disk layout Ultralytics classification expects is a directory tree:

    <out>/train/<class>/<image>.jpg
    <out>/val/<class>/<image>.jpg
    <out>/test/<class>/<image>.jpg

What it does:
  1. Resolves a source of clips:
       - default: the (already cached) FiftyOne / HuggingFace hub dataset
       - --src <dir>: local class-folder videos (<src>/train|test/<class>/*.mp4)
  2. Samples N evenly spaced frames from every clip (not just the first frame,
     so the model actually sees the behaviour)
  3. Writes each frame as a JPG under <out>/<split>/<class>/

Usage:
    # default: pull clips from the hub dataset
    python -m scripts.prepare_dataset --out yolo_dataset --frames-per-clip 12

    # or use local class-folder videos
    python -m scripts.prepare_dataset --src path/to/videos --out yolo_dataset
"""
from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
logger = logging.getLogger("prepare_dataset")

# A clip is described by (video path, class name, split name).
ClipSpec = Tuple[str, str, str]


def _norm_class(name: str) -> str:
    """Normalize a class label into a safe directory / model-class name.

    Spaces -> underscores so labels like '2_opened_panel cover' match the
    'opened_panel_cover' keyword in safestream.common.labels.
    """
    return name.strip().replace(" ", "_")


def collect_from_hub(name: str, max_samples: int | None,
                     train: float, val: float, seed: int) -> List[ClipSpec]:
    """Load the hub dataset and build a deterministic train/val/test split."""
    try:
        from fiftyone.utils.huggingface import load_from_hub
    except ImportError:
        logger.error("FiftyOne is not installed. Run: pip install fiftyone "
                     "(or use --src to read local video dirs instead)")
        raise SystemExit(2)

    logger.info("Loading dataset %s ...", name)
    # overwrite=True so re-runs don't collide with a dataset of the same name
    # already cached in the local FiftyOne database.
    kwargs = {"overwrite": True}
    if max_samples:
        kwargs["max_samples"] = max_samples
    ds = load_from_hub(name, **kwargs)
    logger.info("Got %d clips.", len(ds))

    label_field = "ground_truth"
    samples = [
        s for s in ds
        if getattr(s, label_field, None) is not None
        and hasattr(s[label_field], "label")
    ]
    rnd = random.Random(seed)
    rnd.shuffle(samples)

    n = len(samples)
    n_train = int(train * n)
    n_val = int(val * n)
    bins = {
        "train": samples[:n_train],
        "val":   samples[n_train:n_train + n_val],
        "test":  samples[n_train + n_val:],
    }
    clips: List[ClipSpec] = []
    for split, items in bins.items():
        for s in items:
            clips.append((s.filepath, s[label_field].label, split))
    return clips


def collect_from_local(src: Path, val: float, seed: int) -> List[ClipSpec]:
    """Walk <src>/train|test/<class>/*.mp4 and carve a val split out of train."""
    clips: List[ClipSpec] = []

    # test -> test (verbatim)
    test_root = src / "test"
    if test_root.is_dir():
        for cls_dir in sorted(p for p in test_root.iterdir() if p.is_dir()):
            for vid in sorted(cls_dir.glob("*.mp4")):
                clips.append((str(vid), cls_dir.name, "test"))

    # train -> train + carved val (stratified per class, deterministic)
    train_root = src / "train"
    if not train_root.is_dir():
        logger.error("No train/ directory under %s", src)
        raise SystemExit(2)

    rnd = random.Random(seed)
    for cls_dir in sorted(p for p in train_root.iterdir() if p.is_dir()):
        vids = sorted(cls_dir.glob("*.mp4"))
        rnd.shuffle(vids)
        n_val = int(val * len(vids))
        val_vids = set(vids[:n_val])
        for vid in vids:
            split = "val" if vid in val_vids else "train"
            clips.append((str(vid), cls_dir.name, split))
    return clips


def sample_frames(video_path: str, n_frames: int) -> List["np.ndarray"]:
    """Read a clip and return up to n_frames evenly spaced BGR frames."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        # Unknown length (some codecs); fall back to reading everything.
        frames_all = []
        while True:
            ok, f = cap.read()
            if not ok or f is None:
                break
            frames_all.append(f)
        cap.release()
        if not frames_all:
            return []
        idxs = np.linspace(0, len(frames_all) - 1, min(n_frames, len(frames_all)))
        return [frames_all[int(round(i))] for i in idxs]

    wanted = set(int(round(i)) for i in np.linspace(0, total - 1, min(n_frames, total)))
    out: List[np.ndarray] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if idx in wanted:
            out.append(frame)
        idx += 1
    cap.release()
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="Voxel51/Safe_and_Unsafe_Behaviours",
                   help="Hugging Face dataset id (used unless --src is given)")
    p.add_argument("--src", default=None,
                   help="Local dataset root with train/ and test/ class folders "
                        "(overrides the hub dataset)")
    p.add_argument("--out", default="yolo_dataset",
                   help="Output directory for the YOLO classification dataset")
    p.add_argument("--frames-per-clip", type=int, default=12,
                   help="Evenly spaced frames to extract per clip")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Optional cap on hub clips to use (faster iterations)")
    p.add_argument("--train", type=float, default=0.70,
                   help="Train fraction (hub split only)")
    p.add_argument("--val", type=float, default=0.20,
                   help="Val fraction (hub split, and carved from local train)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.out).expanduser().resolve()

    # ------------------------------------------------------------------
    # 1. Resolve clips
    # ------------------------------------------------------------------
    if args.src:
        src = Path(args.src).expanduser().resolve()
        if src == out or out in src.parents:
            logger.error("--out (%s) must be separate from --src (%s)", out, src)
            return 2
        logger.info("Reading local clips from %s", src)
        clips = collect_from_local(src, args.val, args.seed)
    else:
        clips = collect_from_hub(args.name, args.max_samples,
                                 args.train, args.val, args.seed)

    if not clips:
        logger.error("No clips found.")
        return 2

    classes = sorted({_norm_class(c) for _, c, _ in clips})
    logger.info("Classes (%d): %s", len(classes), ", ".join(classes))
    per_split = defaultdict(int)
    for _, _, sp in clips:
        per_split[sp] += 1
    for sp in ("train", "val", "test"):
        logger.info("Split %s: %d clips", sp, per_split[sp])

    # ------------------------------------------------------------------
    # 2. Fresh output tree
    # ------------------------------------------------------------------
    if out.exists():
        logger.info("Removing existing output dir: %s", out)
        shutil.rmtree(out)
    for sp in ("train", "val", "test"):
        for c in classes:
            (out / sp / c).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 3. Extract frames
    # ------------------------------------------------------------------
    written = 0
    skipped = 0
    for video_path, raw_cls, split in clips:
        cls = _norm_class(raw_cls)
        frames = sample_frames(video_path, args.frames_per_clip)
        if not frames:
            skipped += 1
            continue
        stem = Path(video_path).stem
        dest = out / split / cls
        for i, frame in enumerate(frames):
            cv2.imwrite(str(dest / f"{stem}_f{i:02d}.jpg"), frame)
            written += 1

    logger.info("Wrote %d frames (skipped %d unreadable clips).", written, skipped)

    print()
    print("=" * 72)
    print(f"YOLO classification dataset ready at: {out}")
    print()
    print("Next — train a YOLOv8 classifier:")
    print()
    print(f'  python -m scripts.train_yolo --data "{out}" \\')
    print(f'      --epochs 50 --imgsz 224 --batch 16 --device auto')
    print()
    print("Then use the trained weights with the detector:")
    print()
    print('  python -m safestream.detector \\')
    print('      --weights runs/safestream_yolov8m/weights/best.pt')
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
