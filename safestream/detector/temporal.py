"""Streaming temporal inference for the detector.

Keeps a per-camera sliding buffer of the last K frames and runs the temporal
model on every incoming frame, emitting a calibrated continuous ``unsafe_prob``
plus the usual top label/category/conf. Until a camera's buffer fills, the
earliest frame is repeated (warm-up) so a prediction is available immediately.

torch/torchvision are imported lazily by the caller's process only when
DETECTOR_MODE=temporal, mirroring the detector's lazy YOLO import.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, Tuple

import numpy as np
import torch

from safestream.common.labels import classify_label
from safestream.temporal.dataset import preprocess_frame
from safestream.temporal.model import load_checkpoint, unsafe_prob as _unsafe_prob


class TemporalInfer:
    def __init__(self, weights: str, device: str = "cpu", window: int | None = None):
        self.device = device
        self.model, self.config = load_checkpoint(weights, device=device)
        self.class_names = self.config["class_names"]
        self.unsafe_idx = self.config["unsafe_idx"]
        self.img_size = self.config["img_size"]
        # Allow an override but default to the trained window length.
        self.window = int(window or self.config["window"])
        self._buffers: Dict[str, Deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=self.window)
        )

    def infer(self, camera_id: str, frame_bgr: "np.ndarray") -> Tuple[str, str, float, float]:
        """-> (label, category, conf, unsafe_prob)."""
        buf = self._buffers[camera_id]
        buf.append(preprocess_frame(frame_bgr, self.img_size))

        frames = list(buf)
        if len(frames) < self.window:  # warm-up: pad with the earliest frame
            frames = [frames[0]] * (self.window - len(frames)) + frames

        clip = torch.from_numpy(np.stack(frames)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(clip), dim=1)

        u_prob = float(_unsafe_prob(probs, self.unsafe_idx)[0].item())
        top_idx = int(probs[0].argmax().item())
        conf = float(probs[0, top_idx].item())
        label = self.class_names[top_idx]
        return label, classify_label(label), conf, u_prob
