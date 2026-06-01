"""SafeStream detector: cctv-frames -> YOLOv8 -> safety-detections.

On Apple Silicon the model runs on MPS automatically (set DETECTOR_DEVICE=cpu
in .env to override).
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from typing import Any, Dict, List

from confluent_kafka import KafkaError

from safestream.common.encoding import decode_frame_jpeg, detect_device
from safestream.common.kafka_clients import (
    make_consumer,
    make_producer,
    safe_decode,
)
from safestream.common.labels import classify_label
from safestream.settings import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s :: %(message)s",
)
logger = logging.getLogger("safestream.detector")


def parse_args() -> argparse.Namespace:
    s = get_settings()
    p = argparse.ArgumentParser(description="SafeStream Kafka detection consumer")
    p.add_argument("--weights", default=s.detector_weights,
                   help="Path to YOLOv8 weights (.pt)")
    p.add_argument("--device", default=s.detector_device,
                   help="auto | mps | cuda | cpu")
    p.add_argument("--conf", type=float, default=s.detector_conf)
    p.add_argument("--group-id", default="safestream-detector")
    return p.parse_args()


_STOP = False


def _install_signal_handlers() -> None:
    def _handler(signum, _frame):
        global _STOP
        logger.info("Signal %s received, shutting down...", signum)
        _STOP = True

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def _delivery_report(err, _msg):
    if err is not None:
        logger.warning("Delivery failed: %s", err)


def run() -> int:
    args = parse_args()
    s = get_settings()
    _install_signal_handlers()

    device = detect_device(args.device)
    logger.info("YOLO device: %s", device)

    # Import lazily so the producer/aggregator services don't pay torch import cost
    from ultralytics import YOLO

    logger.info("Loading YOLO weights: %s", args.weights)
    model = YOLO(args.weights)
    try:
        model.to(device)
    except Exception as e:
        logger.warning("model.to(%s) failed: %s -- falling back to default", device, e)

    consumer = make_consumer(args.group_id)
    producer = make_producer()
    consumer.subscribe([s.topic_frames])
    logger.info("Subscribed to %s -> publishing to %s",
                s.topic_frames, s.topic_detections)

    n = 0
    last_log = time.time()
    try:
        while not _STOP:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    logger.warning("Consumer error: %s", msg.error())
                continue

            raw = safe_decode(msg)
            if raw is None:
                continue
            try:
                payload: Dict[str, Any] = json.loads(raw)
            except Exception:
                logger.warning("Bad JSON, skipping")
                continue

            frame = decode_frame_jpeg(payload.get("image_b64", ""))
            if frame is None:
                logger.warning("Failed to decode frame, skipping")
                continue

            # Inference
            preds = model(frame, conf=args.conf, verbose=False)
            boxes = preds[0].boxes if preds else None
            detections: List[Dict[str, Any]] = []
            safe_c = unsafe_c = 0
            if boxes is not None and len(boxes) > 0:
                for b in boxes:
                    cls_id = int(b.cls[0])
                    label = model.names.get(cls_id, str(cls_id))
                    cat = classify_label(label)
                    if cat == "safe":
                        safe_c += 1
                    elif cat == "unsafe":
                        unsafe_c += 1
                    detections.append(
                        {
                            "label": label,
                            "category": cat,
                            "conf": float(b.conf[0]),
                            "bbox": [float(x) for x in b.xyxy[0].tolist()],
                        }
                    )

            out = {
                "camera_id": payload.get("camera_id"),
                "frame_id": payload.get("frame_id"),
                "source": payload.get("source"),
                "timestamp": payload.get("timestamp", time.time()),
                "detections": detections,
                "safe_count": safe_c,
                "unsafe_count": unsafe_c,
                "total_detections": len(detections),
            }
            producer.produce(
                s.topic_detections,
                key=str(payload.get("camera_id")),
                value=json.dumps(out),
                on_delivery=_delivery_report,
            )
            producer.poll(0)
            n += 1
            if time.time() - last_log > 5.0:
                logger.info(
                    "Processed %d frames (last: cam=%s safe=%d unsafe=%d)",
                    n, payload.get("camera_id"), safe_c, unsafe_c,
                )
                last_log = time.time()
    finally:
        logger.info("Closing consumer (processed=%d)", n)
        try:
            consumer.close()
        except Exception:
            pass
        producer.flush(5)
    return 0


if __name__ == "__main__":
    sys.exit(run())
