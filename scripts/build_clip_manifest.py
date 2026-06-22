"""Emit a clip manifest (video_path,label,split) for temporal training.

Unlike scripts/prepare_dataset.py this does NOT extract frames — it keeps the
clips as mp4 and just records which split each belongs to, so the temporal
Dataset can decode K-frame windows on the fly. The split logic is reused
verbatim from prepare_dataset (collect_from_hub / collect_from_local) so the
temporal and per-frame pipelines train/evaluate on the same partition.

Usage:
    # local class-folder videos (<src>/train|test/<class>/*.mp4)
    python -m scripts.build_clip_manifest --src path/to/videos --out clips_manifest.csv

    # or pull from the hub dataset
    python -m scripts.build_clip_manifest --out clips_manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path

from scripts.prepare_dataset import collect_from_hub, collect_from_local

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
logger = logging.getLogger("build_clip_manifest")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="Voxel51/Safe_and_Unsafe_Behaviours",
                   help="Hugging Face dataset id (used unless --src is given)")
    p.add_argument("--src", default=None,
                   help="Local dataset root with train/ and test/ class folders")
    p.add_argument("--out", default="clips_manifest.csv")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--train", type=float, default=0.70)
    p.add_argument("--val", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.src:
        clips = collect_from_local(Path(args.src).expanduser().resolve(),
                                   args.val, args.seed)
    else:
        clips = collect_from_hub(args.name, args.max_samples,
                                 args.train, args.val, args.seed)
    if not clips:
        logger.error("No clips found.")
        return 2

    out = Path(args.out).expanduser().resolve()
    per_split: defaultdict[str, int] = defaultdict(int)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_path", "label", "split"])
        for video_path, label, split in clips:
            w.writerow([video_path, label, split])
            per_split[split] += 1

    logger.info("Wrote %d clips to %s", len(clips), out)
    for sp in ("train", "val", "test"):
        logger.info("  split %s: %d clips", sp, per_split[sp])
    print(f"\nNext: python -m scripts.train_temporal --data {out} --model head")
    return 0


if __name__ == "__main__":
    sys.exit(main())
