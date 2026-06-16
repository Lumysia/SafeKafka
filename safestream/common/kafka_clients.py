"""Factory helpers for `confluent-kafka` Producer / Consumer / AdminClient.

The SASL_SSL config block is used for Confluent Cloud; if `USE_LOCAL_BROKER`
is set the helpers fall back to a plaintext config that works against the
docker-compose broker.
"""
from __future__ import annotations

import logging
import uuid
from typing import Optional

from confluent_kafka import Consumer, Producer
from confluent_kafka.admin import AdminClient

from safestream.settings import get_settings

logger = logging.getLogger(__name__)


def _base_config() -> dict:
    s = get_settings()
    cfg: dict = {
        "bootstrap.servers": s.bootstrap_servers,
        "client.id": f"safestream-{uuid.uuid4().hex[:8]}",
    }
    if s.use_local_broker:
        cfg["security.protocol"] = "PLAINTEXT"
    else:
        cfg.update(
            {
                "security.protocol": "SASL_SSL",
                "sasl.mechanisms": "PLAIN",
                "sasl.username": s.api_key,
                "sasl.password": s.api_secret,
            }
        )
    return cfg


def producer_config(extra: Optional[dict] = None) -> dict:
    cfg = _base_config()
    cfg.update(
        {
            # Reasonable defaults for video frame payloads (~50–200 KB each)
            "compression.type": "lz4",
            "linger.ms": 5,
            "acks": "all",
            "enable.idempotence": True,
            # Allow large frames
            "message.max.bytes": 5 * 1024 * 1024,
        }
    )
    if extra:
        cfg.update(extra)
    return cfg


def consumer_config(group_id: str, extra: Optional[dict] = None) -> dict:
    cfg = _base_config()
    cfg.update(
        {
            "group.id": group_id,
            "auto.offset.reset": get_settings().kafka_auto_offset_reset,
            "enable.auto.commit": True,
            "session.timeout.ms": 45_000,
            # Allow large frames
            "fetch.max.bytes": 50 * 1024 * 1024,
            "max.partition.fetch.bytes": 10 * 1024 * 1024,
        }
    )
    if extra:
        cfg.update(extra)
    return cfg


def admin_config() -> dict:
    return _base_config()


def make_producer(extra: Optional[dict] = None) -> Producer:
    return Producer(producer_config(extra))


def make_consumer(group_id: str, extra: Optional[dict] = None) -> Consumer:
    return Consumer(consumer_config(group_id, extra))


def make_admin() -> AdminClient:
    return AdminClient(admin_config())


def safe_decode(msg) -> Optional[str]:
    """Decode a confluent-kafka message value; return None on errors."""
    try:
        v = msg.value()
        return v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v
    except Exception:
        logger.exception("Failed to decode message value")
        return None
