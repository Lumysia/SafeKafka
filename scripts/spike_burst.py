"""Backpressure demonstration: blast frames into cctv-frames faster than the
detector can drain them, so consumer lag builds up and then recovers.

The normal producer is decode-bound and cannot exceed the detector's inference
budget, so to reproduce a "frame-rate spike beyond the GPU budget" we encode a
single (downscaled) frame once and republish it as fast as the broker accepts,
with the exact cctv-frames payload shape. Run `scripts.measure_lag --samples N`
alongside this to record the detector-group lag rising and then draining once the
burst stops.

    python -m scripts.spike_burst --count 1500 --size 640
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import cv2
import numpy as np

from safestream.common.encoding import encode_frame_jpeg
from safestream.common.kafka_clients import make_producer
from safestream.settings import get_settings


def main() -> int:
    s = get_settings()
    ap = argparse.ArgumentParser(description="Backpressure burst producer")
    ap.add_argument("--count", type=int, default=1500, help="Frames to blast")
    ap.add_argument("--size", type=int, default=640, help="Square frame size (small => cheap)")
    ap.add_argument("--camera-id", default="spike-cam")
    args = ap.parse_args()

    # One synthetic frame, encoded once.
    frame = (np.random.rand(args.size, args.size, 3) * 255).astype(np.uint8)
    b64 = encode_frame_jpeg(frame, s.producer_jpeg_quality)
    msg_bytes = len(json.dumps({"image_b64": b64}).encode())
    print("Blasting %d frames of ~%d KB each to %s (camera_id=%s) as fast as possible"
          % (args.count, msg_bytes // 1024, s.topic_frames, args.camera_id))

    producer = make_producer()
    t0 = time.time()
    for i in range(args.count):
        payload = {
            "camera_id": args.camera_id,
            "frame_id": i,
            "source": "spike",
            "timestamp": time.time(),
            "image_b64": b64,
        }
        producer.produce(s.topic_frames, key=args.camera_id, value=json.dumps(payload))
        producer.poll(0)
        if (i + 1) % 250 == 0:
            print("  queued %d (%.0f msg/s)" % (i + 1, (i + 1) / (time.time() - t0)))
    producer.flush(30)
    dt = time.time() - t0
    print("Sent %d frames in %.1fs (%.0f msg/s offered)" % (args.count, dt, args.count / dt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
