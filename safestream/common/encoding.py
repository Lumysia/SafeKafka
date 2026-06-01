"""Frame ↔ Kafka payload helpers."""
from __future__ import annotations

import base64
from typing import Optional

import cv2
import numpy as np


def encode_frame_jpeg(frame_bgr: np.ndarray, quality: int = 80) -> str:
    """Encode a BGR frame as a base64 JPEG string."""
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def decode_frame_jpeg(b64_str: str) -> Optional[np.ndarray]:
    """Decode a base64 JPEG string back to a BGR ndarray."""
    if not b64_str:
        return None
    try:
        buf = base64.b64decode(b64_str)
        arr = np.frombuffer(buf, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def detect_device(requested: str = "auto") -> str:
    """Return the device string to pass to Ultralytics.

    Order on Apple Silicon: MPS → CPU. On Linux/NVIDIA: CUDA → CPU.
    Explicit values ('mps', 'cuda', 'cpu') are honoured verbatim.
    """
    requested = (requested or "auto").lower()
    if requested != "auto":
        return requested
    try:
        import torch  # local import: keeps non-detector services fast

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"
