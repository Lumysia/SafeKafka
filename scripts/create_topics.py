"""Create the three SafeStream topics on the configured Kafka cluster.

Usage:
    python -m scripts.create_topics            # default partitions=3
    python -m scripts.create_topics --partitions 6 --replicas 3
"""
from __future__ import annotations

import argparse
import logging
import sys

from confluent_kafka.admin import NewTopic

from safestream.common.kafka_clients import make_admin
from safestream.settings import get_settings

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
logger = logging.getLogger("create_topics")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--partitions", type=int, default=3)
    p.add_argument("--replicas", type=int, default=None,
                   help="Replication factor. Defaults to 3 for Confluent Cloud, "
                        "1 for local broker.")
    args = p.parse_args()

    s = get_settings()
    replicas = args.replicas if args.replicas is not None else (
        1 if s.use_local_broker else 3
    )

    admin = make_admin()
    topics = [s.topic_frames, s.topic_detections, s.topic_alerts]
    new_topics = [
        NewTopic(t, num_partitions=args.partitions, replication_factor=replicas)
        for t in topics
    ]
    logger.info(
        "Creating topics on %s (partitions=%d, replicas=%d): %s",
        s.bootstrap_servers, args.partitions, replicas, topics,
    )
    futures = admin.create_topics(new_topics, request_timeout=20)
    rc = 0
    for topic, f in futures.items():
        try:
            f.result()
            logger.info("Created: %s", topic)
        except Exception as e:
            msg = str(e)
            if "already exists" in msg or "TOPIC_ALREADY_EXISTS" in msg:
                logger.info("Already exists: %s", topic)
            else:
                logger.error("Failed to create %s: %s", topic, e)
                rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
