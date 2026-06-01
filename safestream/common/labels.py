"""Map a YOLO class label to one of {safe, unsafe, other}.

The mapping is data, not code: tweak the keyword tables to match your
fine-tuned detector's class names without touching the consumers.
"""
from __future__ import annotations

UNSAFE_KEYWORDS = (
    "unsafe",
    "no_helmet",
    "no_hardhat",
    "no_vest",
    "no_safety",
    "fall",
    "danger",
    "violation",
    "opened_panel_cover",
    "opened panel cover",
    "unauthorized_intervention",
    "unauthorized intervention",
    "carrying_overload_with_forklift",
    "carrying overload",
)

SAFE_KEYWORDS = (
    "safe",
    "wearing_helmet",
    "wearing_vest",
    "hardhat",
    "safety_vest",
    "compliant",
)


def classify_label(label: str) -> str:
    """Return 'safe', 'unsafe', or 'other'.

    A label that contains an *unsafe* keyword wins over a *safe* one (so
    'no_safety_vest' is correctly classified as unsafe even though it
    contains the substring 'safety').
    """
    if not label:
        return "other"
    l = label.lower()
    if any(k in l for k in UNSAFE_KEYWORDS):
        return "unsafe"
    if any(k in l for k in SAFE_KEYWORDS):
        return "safe"
    # Fall back: bare 'unsafe' or 'safe' substring
    if "unsafe" in l:
        return "unsafe"
    if "safe" in l:
        return "safe"
    return "other"
