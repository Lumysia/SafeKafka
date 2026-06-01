"""Standalone aggregator service.

For the dashboard, the same SafetyAggregator class is instantiated inside the
FastAPI process so the WebSocket endpoint can read snapshots directly. This
standalone entry point is useful when you want the aggregator to run as its
own service (e.g. a separate container) without the dashboard.
"""
from __future__ import annotations

import json
import logging
import signal
import sys
import time

from confluent_kafka import KafkaError

from safestream.aggregator.aggregator import SafetyAggregator
from safestream.common.kafka_clients import (
    make_consumer,
    make_producer,
    safe_decode,
)
from safestream.settings import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s :: %(message)s",
)
logger = logging.getLogger("safestream.aggregator")

_STOP = False


def _install_signal_handlers() -> None:
    def _handler(signum, _frame):
        global _STOP
        logger.info("Signal %s received, shutting down...", signum)
        _STOP = True

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def run() -> int:
    s = get_settings()
    _install_signal_handlers()

    agg = SafetyAggregator(
        window_seconds=s.agg_window_seconds,
        unsafe_ratio_alert=s.agg_unsafe_ratio_alert,
        min_window_obs=s.agg_min_window_obs,
    )

    consumer = make_consumer("safestream-aggregator")
    producer = make_producer()
    consumer.subscribe([s.topic_detections])
    logger.info(
        "Subscribed to %s -> publishing alerts to %s (window=%.1fs, ratio>=%.2f)",
        s.topic_detections, s.topic_alerts,
        s.agg_window_seconds, s.agg_unsafe_ratio_alert,
    )

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
                payload = json.loads(raw)
            except Exception:
                continue
            snap, alert = agg.update(payload)
            if alert is not None:
                logger.warning(
                    "ALERT cam=%s ratio=%.2f unsafe=%d/%d severity=%s",
                    alert["camera_id"], alert["rolling_ratio"],
                    alert["rolling_unsafe"], alert["rolling_total"],
                    alert["severity"],
                )
                producer.produce(
                    s.topic_alerts,
                    key=alert["camera_id"],
                    value=json.dumps(alert),
                )
                producer.poll(0)
            n += 1
            if time.time() - last_log > 5.0:
                shot = agg.snapshot()
                logger.info(
                    "Aggregated %d records | per-camera: %s",
                    n,
                    {c: f"safe={v['cumulative_safe']} unsafe={v['cumulative_unsafe']}"
                     for c, v in shot.items()},
                )
                last_log = time.time()
    finally:
        logger.info("Closing aggregator (records=%d)", n)
        try:
            consumer.close()
        except Exception:
            pass
        producer.flush(5)
    return 0


if __name__ == "__main__":
    sys.exit(run())
