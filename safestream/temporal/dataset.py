"""Clip-window dataset for temporal training, decoded straight from the mp4s.

A manifest CSV (``video_path,label,split`` — see scripts/build_clip_manifest.py)
points at the original Safe_and_Unsafe_Behaviours clips. Each __getitem__ decodes
K frames from one clip and returns ``(frames (K,C,H,W) float tensor, class_idx)``.

`sample_frames` from scripts/prepare_dataset.py is reused so frame selection is
identical to the per-frame pipeline; training adds a small temporal jitter so the
model sees different sub-windows across epochs.

Decoding every mp4 in full just to keep ~16 frames is CPU-bound and dominates
training time (the GPU sits idle waiting for it). To avoid re-decoding the same
clips every epoch, the resized RGB frames are cached to disk as ``.npy`` the first
time a clip is read; later epochs (and reruns) load straight from the cache.
Augmentation (sub-window jitter + horizontal flip) is applied per-sample on the
cached frames, so training variety is unchanged. Delete the cache dir to rebuild.
"""
from __future__ import annotations

import csv
import hashlib
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from safestream.temporal.model import IMAGENET_MEAN, IMAGENET_STD

# Reuse the exact frame sampler used to build the classification dataset.
from scripts.prepare_dataset import sample_frames, _norm_class

_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
_STD = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)


def _resize_rgb(frame_bgr: "np.ndarray", img_size: int) -> "np.ndarray":
    """BGR uint8 HxWx3 -> resized RGB uint8 (img_size, img_size, 3). Cache-friendly."""
    img = cv2.resize(frame_bgr, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _normalize_rgb(rgb_uint8: "np.ndarray") -> "np.ndarray":
    """Resized RGB uint8 HxWx3 -> normalised float32 CHW (ImageNet stats)."""
    img = rgb_uint8.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return (img - _MEAN) / _STD


def preprocess_frame(frame_bgr: "np.ndarray", img_size: int) -> "np.ndarray":
    """BGR uint8 HxWx3 -> normalised float32 CHW (ImageNet stats)."""
    return _normalize_rgb(_resize_rgb(frame_bgr, img_size))


def read_manifest(path: str, split: str) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Return ([(video_path, class_name)], sorted_class_names) for one split."""
    rows: List[Tuple[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == split:
                rows.append((r["video_path"], _norm_class(r["label"])))
    classes = sorted({c for _, c in rows})
    return rows, classes


class ClipWindowDataset(Dataset):
    def __init__(
        self,
        manifest: str,
        split: str,
        window: int = 16,
        img_size: int = 224,
        class_names: List[str] | None = None,
        train: bool = False,
        seed: int = 42,
        cache_dir: str | None = None,
    ):
        self.rows, found = read_manifest(manifest, split)
        # Use a fixed class ordering (e.g. the train split's) across all splits.
        self.class_names = class_names if class_names is not None else found
        self.cls_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.window = int(window)
        self.img_size = int(img_size)
        self.train = bool(train)
        self._rng = random.Random(seed)

        # Drop rows whose class isn't in the fixed ordering (defensive).
        self.rows = [(v, c) for v, c in self.rows if c in self.cls_to_idx]

        # Number of frames decoded per clip (over-sample in train for jitter room).
        self._n = self.window + (4 if self.train else 0)
        # Disk cache of decoded+resized frames, keyed by (img_size, n). Default
        # location sits next to the manifest so reruns reuse it.
        base = Path(cache_dir) if cache_dir else Path(manifest).resolve().parent / ".clipcache"
        self._cache_dir = base / f"s{self.img_size}_n{self._n}"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.rows)

    def _cache_path(self, video_path: str) -> Path:
        # Hash the absolute path so distinct clips never collide.
        key = hashlib.sha1(str(Path(video_path).resolve()).encode("utf-8")).hexdigest()[:16]
        return self._cache_dir / f"{key}.npy"

    def _frames_uint8(self, video_path: str) -> "np.ndarray":
        """Return cached (n, img_size, img_size, 3) RGB uint8 frames for a clip,
        decoding and caching on first miss."""
        cp = self._cache_path(video_path)
        if cp.exists():
            try:
                return np.load(cp)
            except Exception:
                cp.unlink(missing_ok=True)  # corrupt cache entry -> re-decode
        frames = sample_frames(video_path, self._n)
        if not frames:
            arr = np.zeros((self._n, self.img_size, self.img_size, 3), np.uint8)
        else:
            rgb = [_resize_rgb(f, self.img_size) for f in frames]
            if len(rgb) < self._n:  # pad short clips by repeating the last frame
                rgb = rgb + [rgb[-1]] * (self._n - len(rgb))
            arr = np.stack(rgb).astype(np.uint8)
        # Atomic-ish write so concurrent workers don't read a half-written file.
        # np.save appends ".npy", so name the temp file accordingly.
        tmp = cp.with_name(f"{cp.stem}.{np.random.randint(1 << 30)}.tmp.npy")
        np.save(tmp, arr)
        tmp.replace(cp)
        return arr

    def _load_window(self, video_path: str) -> "np.ndarray":
        frames = self._frames_uint8(video_path)  # (n, H, W, 3) RGB uint8
        start = 0
        if self.train and len(frames) > self.window:
            start = self._rng.randint(0, len(frames) - self.window)
        window = frames[start:start + self.window]
        if self.train and self._rng.random() < 0.5:
            window = window[:, :, ::-1, :]  # horizontal flip (mirror width axis)
        return np.stack([_normalize_rgb(f) for f in window])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path, cls = self.rows[idx]
        clip = self._load_window(video_path)
        return torch.from_numpy(clip), self.cls_to_idx[cls]
