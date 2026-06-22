"""Central configuration loaded from environment / .env."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Load .env once at import-time. The file is optional: if it doesn't exist
# the environment is left untouched.
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
if _ENV_FILE.exists():
    load_dotenv(_ENV_FILE)


def _bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


class Settings:
    """Plain config object. Keeping it as a class rather than a Pydantic model
    avoids the cold-import cost of pydantic-settings in short-lived CLI calls.
    """

    # Kafka / Confluent Cloud
    bootstrap_servers: str = os.environ.get(
        "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
    )
    api_key: str = os.environ.get("KAFKA_API_KEY", "")
    api_secret: str = os.environ.get("KAFKA_API_SECRET", "")
    use_local_broker: bool = _bool("USE_LOCAL_BROKER", False)
    kafka_auto_offset_reset: str = os.environ.get("KAFKA_AUTO_OFFSET_RESET", "earliest")

    # Topic names
    topic_frames: str = os.environ.get("TOPIC_FRAMES", "cctv-frames")
    topic_detections: str = os.environ.get("TOPIC_DETECTIONS", "safety-detections")
    topic_alerts: str = os.environ.get("TOPIC_ALERTS", "safety-alerts")

    # Producer
    producer_camera_id: str = os.environ.get("PRODUCER_CAMERA_ID", "cam-01")
    producer_analytics_fps: float = float(
        os.environ.get("PRODUCER_ANALYTICS_FPS", "4")
    )
    producer_jpeg_quality: int = int(os.environ.get("PRODUCER_JPEG_QUALITY", "80"))

    # Detector
    detector_weights: str = os.environ.get("DETECTOR_WEIGHTS", "yolov8m.pt")
    detector_device: str = os.environ.get("DETECTOR_DEVICE", "auto")
    detector_conf: float = float(os.environ.get("DETECTOR_CONF", "0.25"))
    # frame  -> per-frame YOLOv8 classifier (legacy default)
    # temporal -> sliding-window temporal model emitting a calibrated unsafe_prob
    detector_mode: str = os.environ.get("DETECTOR_MODE", "frame")
    temporal_weights: str = os.environ.get(
        "TEMPORAL_WEIGHTS", "runs/temporal/best.pt"
    )
    temporal_window: int = int(os.environ.get("TEMPORAL_WINDOW", "16"))

    # Aggregator
    agg_window_seconds: float = float(os.environ.get("AGG_WINDOW_SECONDS", "60"))
    agg_unsafe_ratio_alert: float = float(
        os.environ.get("AGG_UNSAFE_RATIO_ALERT", "0.30")
    )
    agg_min_window_obs: int = int(os.environ.get("AGG_MIN_WINDOW_OBS", "5"))
    agg_alert_cooldown_seconds: float = float(
        os.environ.get("AGG_ALERT_COOLDOWN_SECONDS", "0")
    )
    # Confidence-aware smoothing for the alert decision (used when detections
    # carry a continuous `unsafe_prob`, i.e. the temporal detector). EWMA has a
    # time half-life; hysteresis enters ALERT at enter_threshold (held min_dwell
    # observations) and clears below exit_threshold to stop alert flapping.
    agg_use_prob: bool = _bool("AGG_USE_PROB", True)
    agg_ewma_halflife: float = float(os.environ.get("AGG_EWMA_HALFLIFE", "10"))
    agg_enter_threshold: float = float(os.environ.get("AGG_ENTER_THRESHOLD", "0.5"))
    agg_exit_threshold: float = float(os.environ.get("AGG_EXIT_THRESHOLD", "0.30"))
    agg_min_dwell: int = int(os.environ.get("AGG_MIN_DWELL", "3"))
    # Demo toggle: when false the dashboard bypasses the rolling-window/threshold/
    # cooldown logic and fires one naive alert per unsafe frame, so you can see the
    # alert noise the aggregator normally suppresses.
    agg_enabled: bool = _bool("AGG_ENABLED", True)

    # Dashboard
    dashboard_host: str = os.environ.get("DASHBOARD_HOST", "127.0.0.1")
    dashboard_port: int = int(os.environ.get("DASHBOARD_PORT", "8000"))
    dashboard_frame_fps: float = float(os.environ.get("DASHBOARD_FRAME_FPS", "8"))

    def summary(self) -> str:
        lines = [
            f"bootstrap_servers   = {self.bootstrap_servers}",
            f"use_local_broker    = {self.use_local_broker}",
            f"topic_frames        = {self.topic_frames}",
            f"topic_detections    = {self.topic_detections}",
            f"topic_alerts        = {self.topic_alerts}",
            f"detector_device     = {self.detector_device}",
            f"agg_window_seconds  = {self.agg_window_seconds}",
            f"agg_unsafe_ratio    = {self.agg_unsafe_ratio_alert}",
            f"agg_enabled         = {self.agg_enabled}",
        ]
        return "\n".join(lines)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
