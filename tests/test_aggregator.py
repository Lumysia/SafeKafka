"""Unit tests for SafetyAggregator — no Kafka or YOLO required."""
from __future__ import annotations

import time

from safestream.aggregator.aggregator import SafetyAggregator
from safestream.common.labels import classify_label


def _msg(cam, ts, safe, unsafe):
    return {
        "camera_id": cam,
        "timestamp": ts,
        "safe_count": safe,
        "unsafe_count": unsafe,
    }


def test_classify_label_basic():
    assert classify_label("wearing_helmet") == "safe"
    assert classify_label("no_helmet") == "unsafe"
    assert classify_label("opened_panel_cover") == "unsafe"
    assert classify_label("safety_vest") == "safe"
    assert classify_label("no_safety_vest") == "unsafe"
    assert classify_label("person") == "other"
    assert classify_label("") == "other"


def test_cumulative_totals_accumulate():
    agg = SafetyAggregator(window_seconds=60, unsafe_ratio_alert=0.5,
                           min_window_obs=2)
    t0 = time.time()
    agg.update(_msg("cam-01", t0,     2, 0))
    agg.update(_msg("cam-01", t0 + 1, 1, 1))
    agg.update(_msg("cam-02", t0 + 2, 0, 3))
    shot = agg.snapshot()
    assert shot["cam-01"]["cumulative_safe"] == 3
    assert shot["cam-01"]["cumulative_unsafe"] == 1
    assert shot["cam-01"]["cumulative_frames"] == 2
    assert shot["cam-02"]["cumulative_unsafe"] == 3


def test_rolling_window_evicts_old_events():
    agg = SafetyAggregator(window_seconds=10, unsafe_ratio_alert=0.5,
                           min_window_obs=1)
    t0 = 1000.0
    # Old event outside the window
    agg.update(_msg("cam-01", t0,        5, 0))
    # Recent event
    agg.update(_msg("cam-01", t0 + 100,  0, 4))
    shot = agg.snapshot()
    # Only the recent event should still be in the rolling window
    assert shot["cam-01"]["rolling_safe"] == 0
    assert shot["cam-01"]["rolling_unsafe"] == 4
    # Cumulative still counts both
    assert shot["cam-01"]["cumulative_safe"] == 5
    assert shot["cam-01"]["cumulative_unsafe"] == 4


def test_alert_fires_when_ratio_crosses_threshold():
    agg = SafetyAggregator(window_seconds=60, unsafe_ratio_alert=0.30,
                           high_ratio=0.60, min_window_obs=3)
    t0 = time.time()
    _, a1 = agg.update(_msg("cam-01", t0,     5, 0))    # not enough obs
    _, a2 = agg.update(_msg("cam-01", t0 + 1, 4, 0))    # still safe
    # rolling totals after this call: safe=9, unsafe=6 → ratio=0.40 → WARN
    _, a3 = agg.update(_msg("cam-01", t0 + 2, 0, 6))
    assert a1 is None and a2 is None
    assert a3 is not None
    assert a3["severity"] == "WARN"
    assert 0.30 <= a3["rolling_ratio"] < 0.60
    assert a3["camera_id"] == "cam-01"


def test_alert_escalates_to_high():
    agg = SafetyAggregator(window_seconds=60, unsafe_ratio_alert=0.30,
                           high_ratio=0.60, min_window_obs=3)
    t0 = time.time()
    agg.update(_msg("cam-01", t0,     0, 4))
    agg.update(_msg("cam-01", t0 + 1, 0, 4))
    _, a = agg.update(_msg("cam-01", t0 + 2, 1, 7))     # ratio = 15/16
    assert a is not None and a["severity"] == "HIGH"


def test_alert_does_not_fire_below_min_obs():
    agg = SafetyAggregator(window_seconds=60, unsafe_ratio_alert=0.30,
                           min_window_obs=5)
    t0 = time.time()
    _, a = agg.update(_msg("cam-01", t0, 0, 1))
    assert a is None


def test_alert_cooldown_limits_repeated_alerts():
    agg = SafetyAggregator(window_seconds=60, unsafe_ratio_alert=0.30,
                           min_window_obs=1, alert_cooldown_seconds=5)
    t0 = time.time()
    _, a1 = agg.update(_msg("cam-01", t0, 0, 1))
    _, a2 = agg.update(_msg("cam-01", t0 + 1, 0, 1))
    _, a3 = agg.update(_msg("cam-01", t0 + 5, 0, 1))
    assert a1 is not None
    assert a2 is None
    assert a3 is not None
