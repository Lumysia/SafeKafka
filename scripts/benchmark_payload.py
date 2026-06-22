"""Measure the on-the-wire cost of SafeStream's text-encoded JPEG frame messages.

This quantifies the network overhead of moving frames through Kafka as
base64-in-JSON payloads (the reviewer's "network overhead caused by passing
text-encoded JPEGs"). It is fully offline -- no broker is needed. It rebuilds the
*exact* producer payload (safestream/producer/__main__.py) for a sample of frames
from demo_videos/ and reports, per frame and aggregated:

    raw_bytes        -- uncompressed BGR ndarray size (what a naive raw pipe sends)
    jpeg_bytes       -- cv2 JPEG-encoded size at the producer's quality
    b64_bytes        -- base64 text of the JPEG (cost of making it text-safe)
    msg_bytes        -- full json.dumps(payload) message published to cctv-frames
    lz4_bytes        -- msg after librdkafka's configured lz4 compression
    b64_overhead     -- b64_bytes / jpeg_bytes (base64's ~1.33x expansion)
    json_overhead    -- msg_bytes / b64_bytes (the JSON envelope around the image)
    lz4_ratio        -- lz4_bytes / msg_bytes (how much the broker claws back)

Run:
    python -m scripts.benchmark_payload
    python -m scripts.benchmark_payload --frames 200 --video-dir demo_videos
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import List

import cv2
import lz4.frame

from safestream.common.encoding import encode_frame_jpeg
from safestream.settings import get_settings

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm")


def _collect_videos(root: Path) -> List[Path]:
    vids: List[Path] = []
    for ext in VIDEO_EXTS:
        vids.extend(root.rglob(f"*{ext}"))
    return sorted(vids)


def _sample_frames(videos: List[Path], n: int):
    """Yield up to n BGR frames spread across the available clips."""
    if not videos:
        return
    per_clip = max(1, n // len(videos))
    emitted = 0
    for vp in videos:
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        step = max(1, total // per_clip) if total else 1
        idx = 0
        taken = 0
        while emitted < n and taken < per_clip:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if idx % step == 0:
                yield frame
                emitted += 1
                taken += 1
            idx += 1
        cap.release()
        if emitted >= n:
            break


def main() -> int:
    s = get_settings()
    ap = argparse.ArgumentParser(description="SafeStream frame-payload network-overhead benchmark")
    ap.add_argument("--video-dir", default="demo_videos", help="Directory of clips to sample")
    ap.add_argument("--frames", type=int, default=200, help="Number of frames to measure")
    ap.add_argument("--quality", type=int, default=s.producer_jpeg_quality,
                    help="JPEG quality (defaults to PRODUCER_JPEG_QUALITY)")
    ap.add_argument("--out", default="systems_payload.json")
    args = ap.parse_args()

    root = Path(args.video_dir).expanduser().resolve()
    videos = _collect_videos(root)
    if not videos:
        print(f"ERROR: no videos found under {root}", file=sys.stderr)
        return 2
    print(f"Sampling up to {args.frames} frames from {len(videos)} clip(s) "
          f"under {root} (jpeg quality={args.quality})")

    raw, jpeg, b64, msg, lz4b, dims = [], [], [], [], [], []
    for frame in _sample_frames(videos, args.frames):
        h, w = frame.shape[:2]
        b64_str = encode_frame_jpeg(frame, args.quality)
        # exact producer payload shape (producer/__main__.py:185-191)
        payload = {
            "camera_id": s.producer_camera_id,
            "frame_id": 0,
            "source": "demo_videos/cam_01/0_tr1.mp4",
            "timestamp": 1.7e9,
            "image_b64": b64_str,
        }
        msg_bytes = json.dumps(payload).encode("utf-8")
        jpeg_len = (len(b64_str) * 3) // 4  # decoded JPEG byte length from b64 length
        raw.append(frame.nbytes)
        jpeg.append(jpeg_len)
        b64.append(len(b64_str.encode("ascii")))
        msg.append(len(msg_bytes))
        lz4b.append(len(lz4.frame.compress(msg_bytes)))
        dims.append((w, h))

    n = len(msg)
    if n == 0:
        print("ERROR: decoded 0 frames", file=sys.stderr)
        return 2

    def stats(xs):
        return {
            "mean": statistics.mean(xs),
            "min": min(xs),
            "max": max(xs),
            "median": statistics.median(xs),
        }

    mean_raw = statistics.mean(raw)
    mean_jpeg = statistics.mean(jpeg)
    mean_b64 = statistics.mean(b64)
    mean_msg = statistics.mean(msg)
    mean_lz4 = statistics.mean(lz4b)

    result = {
        "frames_measured": n,
        "jpeg_quality": args.quality,
        "frame_resolution_wxh": f"{dims[0][0]}x{dims[0][1]}",
        "bytes": {
            "raw_bgr": stats(raw),
            "jpeg": stats(jpeg),
            "base64": stats(b64),
            "json_message": stats(msg),
            "lz4_compressed_message": stats(lz4b),
        },
        "ratios": {
            "jpeg_vs_raw": mean_jpeg / mean_raw,
            "base64_overhead_vs_jpeg": mean_b64 / mean_jpeg,
            "json_envelope_vs_base64": mean_msg / mean_b64,
            "lz4_ratio_vs_message": mean_lz4 / mean_msg,
            "wire_vs_jpeg": mean_lz4 / mean_jpeg,
        },
        "fps_4_bandwidth_KBps": {
            "json_uncompressed": mean_msg * s.producer_analytics_fps / 1024.0,
            "lz4_on_wire": mean_lz4 * s.producer_analytics_fps / 1024.0,
        },
    }

    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")

    # ASCII-only console output (cp1252 terminal).
    print("\n=== Frame payload size (mean over %d frames) ===" % n)
    print("  raw BGR ndarray      : %9.0f bytes  (%6.1f KB)" % (mean_raw, mean_raw / 1024))
    print("  JPEG (q=%d)          : %9.0f bytes  (%6.1f KB)" % (args.quality, mean_jpeg, mean_jpeg / 1024))
    print("  base64 text          : %9.0f bytes  (%6.1f KB)" % (mean_b64, mean_b64 / 1024))
    print("  full JSON message    : %9.0f bytes  (%6.1f KB)" % (mean_msg, mean_msg / 1024))
    print("  lz4-compressed (wire): %9.0f bytes  (%6.1f KB)" % (mean_lz4, mean_lz4 / 1024))
    print("\n=== Overhead ratios ===")
    print("  JPEG vs raw          : %.4f  (JPEG keeps %.1f%% of raw)" %
          (mean_jpeg / mean_raw, 100 * mean_jpeg / mean_raw))
    print("  base64 vs JPEG       : %.4f  (+%.1f%% to make it text)" %
          (mean_b64 / mean_jpeg, 100 * (mean_b64 / mean_jpeg - 1)))
    print("  JSON envelope vs b64 : %.4f" % (mean_msg / mean_b64))
    print("  lz4 wire vs message  : %.4f  (broker reclaims %.1f%%)" %
          (mean_lz4 / mean_msg, 100 * (1 - mean_lz4 / mean_msg)))
    print("\n=== Bandwidth at %.0f analytics FPS (one camera) ===" % s.producer_analytics_fps)
    print("  uncompressed JSON    : %6.1f KB/s" % (mean_msg * s.producer_analytics_fps / 1024))
    print("  lz4 on the wire      : %6.1f KB/s" % (mean_lz4 * s.producer_analytics_fps / 1024))
    print("\nWrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
