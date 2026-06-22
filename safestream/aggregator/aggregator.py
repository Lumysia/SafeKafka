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
        use_prob: bool = True,
        ewma_halflife: float = 10.0,
        enter_threshold: float = 0.5,
        exit_threshold: float = 0.30,
        min_dwell: int = 3,
    ):
        self.window = float(window_seconds)
        self.ratio_alert = float(unsafe_ratio_alert)
        self.high_ratio = float(high_ratio)
        self.min_window_obs = int(min_window_obs)
        self.max_history = int(max_history)
        self.alert_cooldown_seconds = float(alert_cooldown_seconds)

        # Confidence-aware smoothing (engaged per-message only when the record
        # carries a continuous `unsafe_prob`, e.g. from the temporal detector).
        self.use_prob = bool(use_prob)
        self.ewma_halflife = float(ewma_halflife)
        self.enter_threshold = float(enter_threshold)
        self.exit_threshold = float(exit_threshold)
        self.min_dwell = int(min_dwell)

        self._windows: Dict[str, _Window] = defaultdict(_Window)
        self._totals: Dict[str, _Totals] = defaultdict(_Totals)
        self._history: Deque[Dict[str, Any]] = deque(maxlen=max_history)
        self._alerts: Deque[Dict[str, Any]] = deque(maxlen=max_history)
        self._alerts_emitted = 0  # lifetime count (not capped like _alerts)
        self._last_alert_ts: Dict[str, float] = {}

        # Per-camera EWMA + hysteresis state for the smoothed alert path.
        self._ewma: Dict[str, float] = {}
        self._ewma_ts: Dict[str, float] = {}
        self._in_alert: Dict[str, bool] = {}
        self._dwell: Dict[str, int] = {}

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

            cooldown_ok = (
                ts - self._last_alert_ts.get(cam, -float("inf"))
                >= self.alert_cooldown_seconds
            )
            violation_label = tot.last_violation["label"] if tot.last_violation else None

            raw_prob = msg.get("unsafe_prob")
            alert: Optional[Dict[str, Any]] = None

            if self.use_prob and raw_prob is not None:
                # --- Confidence-aware EWMA + hysteresis path ------------------
                # Smooth the per-frame unsafe probability with a time-decayed
                # EWMA, then fire on a rising edge (sustained >= enter_threshold
                # for min_dwell obs) and clear below exit_threshold. This emits
                # one alert per unsafe *episode* instead of per noisy frame.
                prob = float(raw_prob)
                prev = self._ewma.get(cam)
                if prev is None:
                    score = prob
                else:
                    dt = max(0.0, ts - self._ewma_ts.get(cam, ts))
                    decay = (
                        0.5 ** (dt / self.ewma_halflife)
                        if self.ewma_halflife > 0 else 0.0
                    )
                    score = decay * prev + (1.0 - decay) * prob
                self._ewma[cam] = score
                self._ewma_ts[cam] = ts

                if not self._in_alert.get(cam, False):
                    if score >= self.enter_threshold:
                        self._dwell[cam] = self._dwell.get(cam, 0) + 1
                    else:
                        self._dwell[cam] = 0
                    if self._dwell.get(cam, 0) >= self.min_dwell and cooldown_ok:
                        self._in_alert[cam] = True
                        alert = {
                            "camera_id": cam,
                            "timestamp": ts,
                            "rolling_unsafe": roll_unsafe,
                            "rolling_total": roll_total,
                            "rolling_ratio": roll_ratio,
                            "smoothed_score": score,
                            "severity": "HIGH" if score >= self.high_ratio else "WARN",
                            "violation_label": violation_label,
                        }
                elif score < self.exit_threshold:
                    # Recovered: arm the next episode.
                    self._in_alert[cam] = False
                    self._dwell[cam] = 0
            else:
                # --- Legacy count-ratio path (unchanged behaviour) -----------
                if (
                    roll_total >= self.min_window_obs
                    and roll_ratio >= self.ratio_alert
                    and cooldown_ok
                ):
                    alert = {
                        "camera_id": cam,
                        "timestamp": ts,
                        "rolling_unsafe": roll_unsafe,
                        "rolling_total": roll_total,
                        "rolling_ratio": roll_ratio,
                        "severity": "HIGH" if roll_ratio >= self.high_ratio else "WARN",
                        "violation_label": violation_label,
                    }

            if alert is not None:
                self._alerts.append(alert)
                self._alerts_emitted += 1
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

            # 4. Lifetime count of alerts emitted across all cameras
            out["_alerts_emitted"] = self._alerts_emitted

            return out

    def record_external_alert(self, alert: Dict[str, Any]) -> None:
        """Append an alert produced outside the normal window logic.

        Used by the dashboard's naive (AGG_ENABLED=false) demo mode so per-frame
        alerts still surface in the UI's alert feed / history.
        """
        with self._lock:
            self._alerts.append(alert)
            self._alerts_emitted += 1

    def recent_alerts(self, limit: int = 25) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._alerts)[-limit:]

    def recent_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._history)[-limit:]