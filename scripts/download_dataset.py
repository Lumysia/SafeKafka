"""Download the Voxel51 Safe_and_Unsafe_Behaviours dataset locally and print
the directory where the .mp4 clips live, so you can pass it to
`safestream.producer --video-dir <dir>`.

Usage:
    python -m scripts.download_dataset
    python -m scripts.download_dataset --max-samples 100
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
logger = logging.getLogger("download_dataset")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="Voxel51/Safe_and_Unsafe_Behaviours",
                   help="Hugging Face dataset id")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Optional cap on clips to download for faster setup")
    args = p.parse_args()

    try:
        import fiftyone as fo
        from fiftyone.utils.huggingface import load_from_hub
    except ImportError:
        logger.error("FiftyOne is not installed. Run: pip install fiftyone fiftyone-db")
        return 2

    logger.info("Loading %s (this may take a few minutes the first time)...",
                args.name)
    kwargs = {}
    if args.max_samples:
        kwargs["max_samples"] = args.max_samples
    ds = load_from_hub(args.name, **kwargs)

    paths = [Path(s.filepath) for s in ds]
    if not paths:
        logger.error("Dataset loaded but no clips found.")
        return 3

    # Find the common parent directory
    common = Path(paths[0].parent)
    for p in paths[1:]:
        while common not in p.parents and common != p.parent:
            common = common.parent
            if str(common) in ("/", str(Path.home())):
                break

    counts: Counter = Counter()
    for s in ds:
        gt = getattr(s, "ground_truth", None)
        if gt is not None and hasattr(gt, "label"):
            counts[gt.label] += 1

    print()
    print("=" * 72)
    print(f"Downloaded {len(paths)} clips.")
    print(f"Clips directory: {common}")
    print("Label breakdown:")
    for k, v in counts.most_common():
        print(f"  {k:<40} {v:>5}")
    print()
    print("Next step — start the live producer against the dataset:")
    print()
    print(f'  python -m safestream.producer --video-dir "{common}" --loop --realtime')
    print()
    print("Or split it across two synthetic cameras for a multi-camera demo:")
    print()
    print(f'  python -m safestream.producer --video-dir "{common}" --loop \\')
    print(f'      --cameras cam-01,cam-02 --shuffle --realtime')
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
