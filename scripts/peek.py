"""Tail any SafeStream topic and pretty-print each record.

Usage:
    python3 scripts/peek.py                       # defaults to safety-alerts
    python3 scripts/peek.py safety-alerts
    python3 scripts/peek.py safety-detections
    python3 scripts/peek.py cctv-frames           # base64 payload is hidden
    python3 scripts/peek.py safety-alerts --from-beginning
    python3 scripts/peek.py safety-alerts --limit 20

Each run uses a brand-new consumer group, so it won't interfere with the
real aggregator's offsets and won't 'steal' messages from other consumers.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Optional

from confluent_kafka import Consumer, KafkaError

from safestream.common.kafka_clients import make_consumer
from safestream.settings import get_settings


def _shorten(value: str, max_len: int = 800) -> str:
    if len(value) <= max_len:
        return value
    return value[:max_len] + f"  …[{len(value) - max_len} more chars truncated]"


def main() -> int:
    s = get_settings()
    default_topic = s.topic_alerts

    p = argparse.ArgumentParser(description="Tail a SafeStream Kafka topic.")
    p.add_argument("topic", nargs="?", default=default_topic,
                   help=f"Topic name (default: {default_topic})")
    p.add_argument("--from-beginning", action="store_true",
                   help="Read history from the start of the topic")
    p.add_argument("--limit", type=int, default=0,
                   help="Stop after N messages (0 = run forever)")
    p.add_argument("--raw", action="store_true",
                   help="Don't pretty-print JSON; show raw bytes")
    args = p.parse_args()

    # Unique throwaway group + offset reset policy.
    group_id = f"peek-{int(time.time() * 1000)}"
    extra = {
        "auto.offset.reset": "earliest" if args.from_beginning else "latest",
        "enable.auto.commit": False,
    }
    consumer: Optional[Consumer] = None
    try:
        consumer = make_consumer(group_id, extra=extra)
        consumer.subscribe([args.topic])
        print(f"Tailing topic: {args.topic}  "
              f"(from {'beginning' if args.from_beginning else 'latest'}, "
              f"Ctrl-C to stop)\n")

        seen = 0
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                print(f"!! consumer error: {msg.error()}", file=sys.stderr)
                continue

            key = msg.key().decode("utf-8", errors="replace") if msg.key() else "-"
            raw_val = msg.value().decode("utf-8", errors="replace") if msg.value() else ""

            print(f"── #{seen + 1}  partition={msg.partition()}  offset={msg.offset()}"
                  f"  key={key}")
            if args.raw:
                print(_shorten(raw_val))
            else:
                try:
                    obj = json.loads(raw_val)
                    # Strip giant base64 image payloads if present
                    if isinstance(obj, dict) and "image_b64" in obj:
                        n = len(obj["image_b64"])
                        obj["image_b64"] = f"<{n} chars of base64 image, hidden>"
                    print(json.dumps(obj, indent=2, default=str))
                except Exception:
                    print(_shorten(raw_val))
            print()

            seen += 1
            if args.limit and seen >= args.limit:
                print(f"Reached --limit {args.limit}, stopping.")
                break
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        if consumer is not None:
            try:
                consumer.close()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())