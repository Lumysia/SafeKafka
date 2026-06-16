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
from typing import Dict, List

import cv2
import uvicorn
from confluent_kafka import KafkaError
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from safestream.aggregator.aggregator import SafetyAggregator
from safestream.common.encoding import decode_frame_jpeg
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
    alert_cooldown_seconds=SETTINGS.agg_alert_cooldown_seconds,
)
STOP_EVENT = threading.Event()

_frame_lock = threading.Lock()
LATEST_FRAMES: Dict[str, str] = {}       # camera_id → image_b64
LATEST_DETECTIONS: Dict[str, List] = {}  # camera_id → detections[]
VIOLATION_FRAMES: Dict[str, bytes] = {}  # camera_id → annotated JPEG bytes of last violation
_last_frame_ts: Dict[str, float] = {}    # throttle tracker

FRAME_DISPLAY_INTERVAL = 1.0 / max(SETTINGS.dashboard_frame_fps, 0.1)


def _settings_payload() -> dict:
    return {
        "window_seconds": SETTINGS.agg_window_seconds,
        "unsafe_ratio_alert": SETTINGS.agg_unsafe_ratio_alert,
        "min_window_obs": SETTINGS.agg_min_window_obs,
        "alert_cooldown_seconds": SETTINGS.agg_alert_cooldown_seconds,
        "topic_detections": SETTINGS.topic_detections,
        "topic_frames": SETTINGS.topic_frames,
        "topic_alerts": SETTINGS.topic_alerts,
        "detector_weights": SETTINGS.detector_weights,
        "detector_device": SETTINGS.detector_device,
        "detector_conf": SETTINGS.detector_conf,
        "producer_analytics_fps": SETTINGS.producer_analytics_fps,
        "dashboard_frame_fps": SETTINGS.dashboard_frame_fps,
    }


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
            cam = payload.get("camera_id", "unknown")
            _snap, alert = AGG.update(payload)
            with _frame_lock:
                LATEST_DETECTIONS[cam] = payload.get("detections", [])
            unsafe_dets = [d for d in payload.get("detections", []) if d.get("category") == "unsafe"]
            if unsafe_dets:
                with _frame_lock:
                    b64 = LATEST_FRAMES.get(cam)
                if b64:
                    vframe = decode_frame_jpeg(b64)
                    if vframe is not None:
                        for det in unsafe_dets:
                            bbox = det.get("bbox")
                            if bbox and len(bbox) == 4:
                                x1, y1, x2, y2 = (int(c) for c in bbox)
                                cv2.rectangle(vframe, (x1, y1), (x2, y2), (50, 50, 220), 2)
                                lbl = f"{det.get('label', '')} {det.get('conf', 0):.2f}"
                                cv2.putText(vframe, lbl, (x1, max(y1 - 6, 10)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 220), 1, cv2.LINE_AA)
                        ok, buf = cv2.imencode(".jpg", vframe, [cv2.IMWRITE_JPEG_QUALITY, 75])
                        if ok:
                            with _frame_lock:
                                VIOLATION_FRAMES[cam] = buf.tobytes()
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


def _frames_loop() -> None:
    """Background thread: consume cctv-frames, store latest frame per camera (~1 fps)."""
    consumer = make_consumer("safestream-dashboard-frames")
    consumer.subscribe([SETTINGS.topic_frames])
    try:
        while not STOP_EVENT.is_set():
            msg = consumer.poll(1.0)
            if msg is None or msg.error():
                continue
            raw = safe_decode(msg)
            if raw is None:
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                continue
            cam = payload.get("camera_id", "unknown")
            now = time.time()
            with _frame_lock:
                if now - _last_frame_ts.get(cam, 0) >= FRAME_DISPLAY_INTERVAL:
                    LATEST_FRAMES[cam] = payload.get("image_b64", "")
                    _last_frame_ts[cam] = now
    finally:
        try:
            consumer.close()
        except Exception:
            pass


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
    t2 = threading.Thread(target=_frames_loop, daemon=True, name="frames-loop")
    t2.start()
    logger.info("Background Kafka threads started.")


@app.on_event("shutdown")
def _shutdown() -> None:
    STOP_EVENT.set()


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML, headers={"Cache-Control": "no-store"})


@app.get("/api/snapshot")
def api_snapshot() -> JSONResponse:
    return JSONResponse(
        {
            "settings": _settings_payload(),
            "cameras": AGG.snapshot(),
        }
    )


@app.get("/api/alerts")
def api_alerts(limit: int = 25) -> JSONResponse:
    return JSONResponse({"alerts": AGG.recent_alerts(limit=limit)})


@app.get("/api/history")
def api_history(limit: int = 100) -> JSONResponse:
    return JSONResponse({"history": AGG.recent_history(limit=limit)})


@app.get("/api/frame/{camera_id}")
def api_frame(camera_id: str) -> Response:
    with _frame_lock:
        b64 = LATEST_FRAMES.get(camera_id)
        dets = list(LATEST_DETECTIONS.get(camera_id, []))
    if not b64:
        return Response(status_code=204)

    frame = decode_frame_jpeg(b64)
    if frame is None:
        return Response(status_code=204)

    for det in dets:
        bbox = det.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = (int(c) for c in bbox)
        color = (50, 50, 220) if det.get("category") == "unsafe" else (50, 200, 80)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{det.get('label', '')} {det.get('conf', 0):.2f}"
        cv2.putText(frame, label, (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        return Response(status_code=500)
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/api/stream/{camera_id}")
def api_stream(camera_id: str) -> StreamingResponse:
    boundary = "frame"
    delay = 1.0 / max(SETTINGS.dashboard_frame_fps, 0.1)

    def frames():
        last_image = None
        while not STOP_EVENT.is_set():
            with _frame_lock:
                b64 = LATEST_FRAMES.get(camera_id)
                dets = list(LATEST_DETECTIONS.get(camera_id, []))
            if b64 and b64 != last_image:
                frame = decode_frame_jpeg(b64)
                if frame is not None:
                    for det in dets:
                        bbox = det.get("bbox")
                        if not bbox or len(bbox) < 4:
                            continue
                        x1, y1, x2, y2 = (int(c) for c in bbox)
                        color = (50, 50, 220) if det.get("category") == "unsafe" else (50, 200, 80)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{det.get('label', '')} {det.get('conf', 0):.2f}"
                        cv2.putText(frame, label, (x1, max(y1 - 6, 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ok:
                        data = buf.tobytes()
                        last_image = b64
                        yield (
                            f"--{boundary}\r\n"
                            "Content-Type: image/jpeg\r\n"
                            f"Content-Length: {len(data)}\r\n\r\n"
                        ).encode("ascii") + data + b"\r\n"
            time.sleep(delay)

    return StreamingResponse(
        frames(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/violation-frame/{camera_id}")
def api_violation_frame(camera_id: str) -> Response:
    with _frame_lock:
        data = VIOLATION_FRAMES.get(camera_id)
    if not data:
        return Response(status_code=204)
    return Response(content=data, media_type="image/jpeg")


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
            now = time.time()
            await HUB.broadcast(
                {
                    "ts": now,
                    "settings": _settings_payload(),
                    "cameras": AGG.snapshot(),
                    "recent_alerts": AGG.recent_alerts(limit=10),
                    "frame_ages": {cam: now - ts for cam, ts in _last_frame_ts.items()},
                }
            )

    asyncio.create_task(broadcaster())


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await HUB.connect(ws)
    try:
        # Send an immediate snapshot on connect
        now = time.time()
        await ws.send_text(
            json.dumps(
                {
                    "ts": now,
                    "settings": _settings_payload(),
                    "cameras": AGG.snapshot(),
                    "recent_alerts": AGG.recent_alerts(limit=10),
                    "frame_ages": {cam: now - ts for cam, ts in _last_frame_ts.items()},
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
