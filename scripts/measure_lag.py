"""Measure Kafka consumer-group lag for the SafeStream pipeline.

Lag for a (group, topic) pair = sum over partitions of
(high_watermark - committed_offset). A steady-state lag near zero means the
consumer keeps up with the producer; a growing lag means the stage is the
bottleneck and frames are buffering in the broker (backpressure).

Reuses the project's Kafka config (safestream/common/kafka_clients.py) so it
talks to the same Confluent Cloud / local broker as the services.

Run once:
    python -m scripts.measure_lag

Watch during a load/backpressure test (writes a time series to JSON):
    python -m scripts.measure_lag --samples 30 --interval 2 --out systems_lag.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Dict, List, Tuple

from confluent_kafka import Consumer, TopicPartition

from safestream.common.kafka_clients import consumer_config
from safestream.settings import get_settings

# (consumer group id, topic) pairs used by the pipeline.
def _default_targets() -> List[Tuple[str, str]]:
    s = get_settings()
    return [
        ("safestream-detector", s.topic_frames),       # detector reads cctv-frames
        ("safestream-dashboard", s.topic_detections),   # dashboard reads safety-detections
        ("safestream-aggregator", s.topic_detections),  # standalone aggregator reads safety-detections
    ]


def _group_lag(group_id: str, topic: str) -> Dict:
    """Return lag info for one consumer group on one topic."""
    # A short-lived consumer in the *target* group so committed() returns that
    # group's offsets. We never subscribe, so we don't disturb its assignment.
    cfg = consumer_config(group_id, {"enable.auto.commit": False})
    c = Consumer(cfg)
    try:
        md = c.list_topics(topic, timeout=10)
        tmd = md.topics.get(topic)
        if tmd is None or tmd.error is not None:
            return {"group": group_id, "topic": topic, "error": "topic-unavailable", "total_lag": None}
        parts = [TopicPartition(topic, p) for p in tmd.partitions]
        committed = c.committed(parts, timeout=10)
        per_partition = []
        total_lag = 0
        any_committed = False
        for tp in committed:
            lo, hi = c.get_watermark_offsets(tp, timeout=10, cached=False)
            if tp.offset is None or tp.offset < 0:
                # No committed offset for this group/partition yet.
                lag = None
            else:
                any_committed = True
                lag = max(0, hi - tp.offset)
                total_lag += lag
            per_partition.append({
                "partition": tp.partition,
                "committed": tp.offset if tp.offset is not None and tp.offset >= 0 else None,
                "low": lo,
                "high": hi,
                "lag": lag,
            })
        return {
            "group": group_id,
            "topic": topic,
            "partitions": len(parts),
            "total_lag": total_lag if any_committed else None,
            "has_committed_offsets": any_committed,
            "detail": per_partition,
        }
    finally:
        c.close()


def _snapshot(targets: List[Tuple[str, str]]) -> Dict:
    rows = [_group_lag(g, t) for g, t in targets]
    return {"t": time.time(), "groups": rows}


def _print_snapshot(snap: Dict) -> None:
    print("  %-26s %-20s %6s %10s" % ("group", "topic", "parts", "lag"))
    for r in snap["groups"]:
        lag = r.get("total_lag")
        lag_s = "n/a" if lag is None else str(lag)
        print("  %-26s %-20s %6s %10s" % (
            r["group"], r["topic"], r.get("partitions", "?"), lag_s))


def main() -> int:
    ap = argparse.ArgumentParser(description="SafeStream Kafka consumer-lag probe")
    ap.add_argument("--samples", type=int, default=1, help="Number of snapshots (watch mode if >1)")
    ap.add_argument("--interval", type=float, default=2.0, help="Seconds between snapshots")
    ap.add_argument("--out", default=None, help="Write the time series to this JSON file")
    args = ap.parse_args()

    targets = _default_targets()
    series = []
    try:
        for i in range(args.samples):
            snap = _snapshot(targets)
            series.append(snap)
            ts = time.strftime("%H:%M:%S", time.localtime(snap["t"]))
            print("[%s] sample %d/%d" % (ts, i + 1, args.samples))
            _print_snapshot(snap)
            if i < args.samples - 1:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        pass

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"series": series}, fh, indent=2)
        print("\nWrote %s (%d samples)" % (args.out, len(series)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
