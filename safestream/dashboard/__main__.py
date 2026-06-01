"""FastAPI dashboard for SafeStream-Kafka.

Runs a SafetyAggregator in the same process, drives it from a background
thread that consumes the safety-detections topic, and exposes:

  GET  /                 -> static HTML page
  GET  /api/snapshot     -> current per-camera totals (JSON)
  GET  /api/alerts       -> most recent alerts (JSON)
  WS   /ws               -> push snapshots + alerts at ~1 Hz

Start: python -m safestream.dashboard
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import List

import uvicorn
from confluent_kafka import KafkaError
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

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
logger = logging.getLogger("safestream.dashboard")

# ---------------------------------------------------------------------------
# Shared aggregator + background consumer
# ---------------------------------------------------------------------------
SETTINGS = get_settings()
AGG = SafetyAggregator(
    window_seconds=SETTINGS.agg_window_seconds,
    unsafe_ratio_alert=SETTINGS.agg_unsafe_ratio_alert,
    min_window_obs=SETTINGS.agg_min_window_obs,
)
STOP_EVENT = threading.Event()


def _kafka_loop() -> None:
    """Background thread: consume detections, update the aggregator,
    re-publish alerts to the safety-alerts topic."""
    consumer = make_consumer("safestream-dashboard")
    producer = make_producer()
    consumer.subscribe([SETTINGS.topic_detections])
    logger.info(
        "Dashboard consumer subscribed to %s (window=%.1fs, ratio>=%.2f)",
        SETTINGS.topic_detections,
        SETTINGS.agg_window_seconds,
        SETTINGS.agg_unsafe_ratio_alert,
    )
    n = 0
    try:
        while not STOP_EVENT.is_set():
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
            _snap, alert = AGG.update(payload)
            if alert is not None:
                logger.warning(
                    "ALERT cam=%s ratio=%.2f severity=%s",
                    alert["camera_id"], alert["rolling_ratio"], alert["severity"],
                )
                producer.produce(
                    SETTINGS.topic_alerts,
                    key=alert["camera_id"],
                    value=json.dumps(alert),
                )
                producer.poll(0)
            n += 1
    finally:
        logger.info("Dashboard consumer stopping (records=%d)", n)
        try:
            consumer.close()
        except Exception:
            pass
        producer.flush(5)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="SafeStream-Kafka Dashboard")

INDEX_HTML = (Path(__file__).parent / "static" / "index.html").read_text(
    encoding="utf-8"
)


@app.on_event("startup")
def _startup() -> None:
    t = threading.Thread(target=_kafka_loop, daemon=True, name="kafka-loop")
    t.start()
    logger.info("Background Kafka thread started.")


@app.on_event("shutdown")
def _shutdown() -> None:
    STOP_EVENT.set()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.get("/api/snapshot")
def api_snapshot() -> JSONResponse:
    return JSONResponse(
        {
            "settings": {
                "window_seconds": SETTINGS.agg_window_seconds,
                "unsafe_ratio_alert": SETTINGS.agg_unsafe_ratio_alert,
                "topic_detections": SETTINGS.topic_detections,
                "topic_alerts": SETTINGS.topic_alerts,
            },
            "cameras": AGG.snapshot(),
        }
    )


@app.get("/api/alerts")
def api_alerts(limit: int = 25) -> JSONResponse:
    return JSONResponse({"alerts": AGG.recent_alerts(limit=limit)})


@app.get("/api/history")
def api_history(limit: int = 100) -> JSONResponse:
    return JSONResponse({"history": AGG.recent_history(limit=limit)})


class _Hub:
    def __init__(self) -> None:
        self.clients: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self.clients.append(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            if ws in self.clients:
                self.clients.remove(ws)

    async def broadcast(self, payload: dict) -> None:
        msg = json.dumps(payload, default=str)
        async with self._lock:
            stale: List[WebSocket] = []
            for c in self.clients:
                try:
                    await c.send_text(msg)
                except Exception:
                    stale.append(c)
            for s in stale:
                if s in self.clients:
                    self.clients.remove(s)


HUB = _Hub()


@app.on_event("startup")
async def _start_broadcaster() -> None:
    async def broadcaster() -> None:
        while True:
            await asyncio.sleep(1.0)
            await HUB.broadcast(
                {
                    "ts": time.time(),
                    "cameras": AGG.snapshot(),
                    "recent_alerts": AGG.recent_alerts(limit=10),
                }
            )

    asyncio.create_task(broadcaster())


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await HUB.connect(ws)
    try:
        # Send an immediate snapshot on connect
        await ws.send_text(
            json.dumps(
                {
                    "ts": time.time(),
                    "cameras": AGG.snapshot(),
                    "recent_alerts": AGG.recent_alerts(limit=10),
                },
                default=str,
            )
        )
        while True:
            # We don't expect input — just keep the socket open
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await HUB.disconnect(ws)


def main() -> None:
    uvicorn.run(
        "safestream.dashboard.__main__:app",
        host=SETTINGS.dashboard_host,
        port=SETTINGS.dashboard_port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
