"""SafetyAggregator: per-camera rolling window + cumulative totals.

The dashboard server creates one of these and shares it with the Kafka
consumer thread. The aggregator is thread-safe — calls into `update()` and
`snapshot()` are protected by an internal lock.
"""
from __future__ import annotations

import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple


@dataclass
class _Window:
    events: Deque[Tuple[float, int, int]] = field(default_factory=deque)


@dataclass
class _Totals:
    safe: int = 0
    unsafe: int = 0
    frames: int = 0
    label_counts: Dict[str, int] = field(default_factory=dict)
    safe_label_counts: Dict[str, int] = field(default_factory=dict)
    last_detection: Optional[Dict[str, Any]] = None
    last_violation: Optional[Dict[str, Any]] = None


class SafetyAggregator:
    """Stateful rolling-window + cumulative aggregator."""

    def __init__(
        self,
        window_seconds: float = 60.0,
        unsafe_ratio_alert: float = 0.30,
        high_ratio: float = 0.60,
        min_window_obs: int = 5,
        max_history: int = 5000,
        alert_cooldown_seconds: float = 0.0,
    ):
        self.window = float(window_seconds)
        self.ratio_alert = float(unsafe_ratio_alert)
        self.high_ratio = float(high_ratio)
        self.min_window_obs = int(min_window_obs)
        self.max_history = int(max_history)
        self.alert_cooldown_seconds = float(alert_cooldown_seconds)

        self._windows: Dict[str, _Window] = defaultdict(_Window)
        self._totals: Dict[str, _Totals] = defaultdict(_Totals)
        self._history: Deque[Dict[str, Any]] = deque(maxlen=max_history)
        self._alerts: Deque[Dict[str, Any]] = deque(maxlen=max_history)
        self._last_alert_ts: Dict[str, float] = {}
        self._lock = threading.Lock()

    def update(self, msg: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """Ingest one detection-summary record."""
        cam = str(msg.get("camera_id", "unknown"))
        ts = float(msg.get("timestamp"))
        s = int(msg.get("safe_count", 0))
        u = int(msg.get("unsafe_count", 0))

        with self._lock:
            w = self._windows[cam]
            w.events.append((ts, s, u))

            cutoff = ts - self.window
            while w.events and w.events[0][0] < cutoff:
                w.events.popleft()

            roll_safe = sum(e[1] for e in w.events)
            roll_unsafe = sum(e[2] for e in w.events)
            roll_total = roll_safe + roll_unsafe
            roll_ratio = (roll_unsafe / roll_total) if roll_total > 0 else 0.0

            tot = self._totals[cam]
            tot.safe += s
            tot.unsafe += u
            tot.frames += 1

            detections = msg.get("detections", [])
            if detections:
                det = detections[0]
                tot.last_detection = {
                    "label": det.get("label", "unknown"),
                    "category": det.get("category", "other"),
                    "conf": float(det.get("conf", 0)),
                    "timestamp": ts,
                    "source": msg.get("source"),
                    "frame_id": msg.get("frame_id"),
                }

            for det in detections:
                cat = det.get("category")
                lbl = det.get("label", "unknown")

                if cat == "unsafe":
                    tot.label_counts[lbl] = tot.label_counts.get(lbl, 0) + 1
                    conf = float(det.get("conf", 0))
                    tot.last_violation = {
                        "label": lbl,
                        "conf": conf,
                        "timestamp": ts,
                    }

                elif cat == "safe":
                    tot.safe_label_counts[lbl] = tot.safe_label_counts.get(lbl, 0) + 1

            snap = {
                "camera_id": cam,
                "timestamp": ts,
                "cumulative_safe": tot.safe,
                "cumulative_unsafe": tot.unsafe,
                "cumulative_frames": tot.frames,
                "rolling_safe": roll_safe,
                "rolling_unsafe": roll_unsafe,
                "rolling_total": roll_total,
                "rolling_ratio": roll_ratio,
            }
            self._history.append(snap)

            alert: Optional[Dict[str, Any]] = None
            if (
                roll_total >= self.min_window_obs
                and roll_ratio >= self.ratio_alert
                and ts - self._last_alert_ts.get(cam, -float("inf")) >= self.alert_cooldown_seconds
            ):
                alert = {
                    "camera_id": cam,
                    "timestamp": ts,
                    "rolling_unsafe": roll_unsafe,
                    "rolling_total": roll_total,
                    "rolling_ratio": roll_ratio,
                    "severity": "HIGH" if roll_ratio >= self.high_ratio else "WARN",
                    "violation_label": tot.last_violation["label"] if tot.last_violation else None,
                }
                self._alerts.append(alert)
                self._last_alert_ts[cam] = ts

            return snap, alert

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Per-camera cumulative + dashboard summary state."""
        with self._lock:
            out: Dict[str, Dict[str, Any]] = {}

            for cam, tot in self._totals.items():
                w = self._windows[cam]

                roll_safe = sum(e[1] for e in w.events)
                roll_unsafe = sum(e[2] for e in w.events)
                roll_total = roll_safe + roll_unsafe
                roll_ratio = (roll_unsafe / roll_total) if roll_total > 0 else 0.0

                out[cam] = {
                    "cumulative_safe": tot.safe,
                    "cumulative_unsafe": tot.unsafe,
                    "cumulative_frames": tot.frames,
                    "cumulative_ratio": (
                        tot.unsafe / (tot.safe + tot.unsafe)
                        if (tot.safe + tot.unsafe) > 0
                        else 0.0
                    ),
                    "rolling_safe": roll_safe,
                    "rolling_unsafe": roll_unsafe,
                    "rolling_total": roll_total,
                    "rolling_ratio": roll_ratio,
                    "violation_counts": dict(
                        sorted(tot.label_counts.items(), key=lambda x: x[1], reverse=True)
                    ),
                    "safe_label_counts": dict(
                        sorted(tot.safe_label_counts.items(), key=lambda x: x[1], reverse=True)
                    ),
                    "last_detection": dict(tot.last_detection) if tot.last_detection else None,
                    "last_violation": dict(tot.last_violation) if tot.last_violation else None,
                }

            # 1. Top Unsafe Behaviors across all cameras
            top_unsafe: Dict[str, int] = {}
            for tot in self._totals.values():
                for label, count in tot.label_counts.items():
                    top_unsafe[label] = top_unsafe.get(label, 0) + count

            out["_top_unsafe_behaviors"] = dict(
                sorted(top_unsafe.items(), key=lambda x: x[1], reverse=True)[:5]
            )

            # 2. Alert History, newest 20 alerts
            out["_alert_history"] = list(self._alerts)[-20:]

            # 3. Trend Chart data, last 60 detection events
            out["_trend_chart"] = list(self._history)[-60:]

            return out

    def recent_alerts(self, limit: int = 25) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._alerts)[-limit:]

    def recent_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._history)[-limit:]