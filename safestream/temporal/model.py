"""Temporal safety-behaviour models + checkpoint helpers.

Two interchangeable models, both consuming a window of frames shaped
``(B, K, C, H, W)`` and emitting class logits ``(B, num_classes)``:

- ``EmbeddingTemporalModel`` (``kind="head"``): a frozen/fine-tuneable 2D CNN
  frame encoder (torchvision ResNet-18 by default) + a small GRU/attention
  temporal head. Cheap enough to run per-frame in the streaming detector.
- ``VideoModel`` (``kind="video"``): a torchvision 3D-conv video network
  (R(2+1)D-18) fine-tuned on the clips — the stronger "temporal SOTA" ablation.

`unsafe_prob` is the softmax mass on the classes that map to the *unsafe*
category via :func:`safestream.common.labels.classify_label`, so the streaming
contract stays identical to the per-frame detector.

torchvision is already a transitive dependency of ultralytics, so neither model
adds a new heavy install.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from safestream.common.labels import classify_label

# ImageNet normalisation (both the 2D encoder and the video net expect it).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_IMG_SIZE = {"head": 224, "video": 112, "mvit": 224, "swin3d": 224, "hiera": 224}

# Every registered kind. CLIs import this instead of hard-coding the list.
MODEL_KINDS = ("head", "video", "mvit", "swin3d", "hiera")


# ---------------------------------------------------------------------------
# torchvision loaders (robust across the weights= / pretrained= API change)
# ---------------------------------------------------------------------------
def _resnet18(pretrained: bool) -> Tuple[nn.Module, int]:
    import torchvision

    try:  # new API
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        net = torchvision.models.resnet18(weights=weights)
    except (AttributeError, TypeError):  # old API
        net = torchvision.models.resnet18(pretrained=pretrained)
    feat_dim = net.fc.in_features
    net.fc = nn.Identity()  # expose the 512-d penultimate embedding
    return net, feat_dim


def _r2plus1d_18(pretrained: bool, num_classes: int) -> nn.Module:
    import torchvision

    try:
        weights = (
            torchvision.models.video.R2Plus1D_18_Weights.DEFAULT if pretrained else None
        )
        net = torchvision.models.video.r2plus1d_18(weights=weights)
    except (AttributeError, TypeError):
        net = torchvision.models.video.r2plus1d_18(pretrained=pretrained)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


def _mvit_v2_s(pretrained: bool, num_classes: int) -> Tuple[nn.Module, nn.Module]:
    """MViTv2-S (Kinetics-400). Head is ``Sequential(Dropout, Linear)``; swap the
    final Linear. Returns ``(net, new_head_linear)`` so the caller can freeze the
    backbone and train only the head."""
    import torchvision

    try:
        weights = (
            torchvision.models.video.MViT_V2_S_Weights.DEFAULT if pretrained else None
        )
        net = torchvision.models.video.mvit_v2_s(weights=weights)
    except (AttributeError, TypeError):
        net = torchvision.models.video.mvit_v2_s(pretrained=pretrained)
    net.head[-1] = nn.Linear(net.head[-1].in_features, num_classes)
    return net, net.head[-1]


def _swin3d_t(pretrained: bool, num_classes: int) -> Tuple[nn.Module, nn.Module]:
    """Video Swin-T (Kinetics-400). Head is a single ``Linear``; swap it.
    Returns ``(net, new_head_linear)``."""
    import torchvision

    try:
        weights = (
            torchvision.models.video.Swin3D_T_Weights.DEFAULT if pretrained else None
        )
        net = torchvision.models.video.swin3d_t(weights=weights)
    except (AttributeError, TypeError):
        net = torchvision.models.video.swin3d_t(pretrained=pretrained)
    net.head = nn.Linear(net.head.in_features, num_classes)
    return net, net.head


def _replace_last_linear(module: nn.Module, num_classes: int) -> nn.Linear:
    """Swap the deepest ``nn.Linear`` in *module* for a fresh one sized to
    *num_classes*; return the new layer. Used to retarget arbitrary classifier
    heads (e.g. Hiera's) without hard-coding their attribute layout."""
    last_name = last_lin = None
    for name, m in module.named_modules():
        if isinstance(m, nn.Linear):
            last_name, last_lin = name, m
    if last_lin is None:
        raise RuntimeError("no nn.Linear head found to replace")
    new = nn.Linear(last_lin.in_features, num_classes)
    parent = module
    *parents, leaf = last_name.split(".")
    for p in parents:
        parent = getattr(parent, p)
    setattr(parent, leaf, new)
    return new


def _hiera_base(pretrained: bool, num_classes: int) -> Tuple[nn.Module, nn.Module]:
    """Hiera-B fine-tuned on Kinetics-400 (``hiera-transformer`` package, imports
    as ``hiera``). Expects 16x224x224 clips. Returns ``(net, new_head_linear)``."""
    try:
        import hiera as _hiera
    except ImportError as e:  # pragma: no cover - optional heavy dep
        raise ImportError(
            "kind='hiera' needs the 'hiera-transformer' package "
            "(pip install hiera-transformer)."
        ) from e

    kwargs = dict(pretrained=pretrained)
    if pretrained:
        kwargs["checkpoint"] = "mae_k400_ft_k400"
    try:
        # Fast path: let Hiera build a head sized to our classes directly.
        net = _hiera.hiera_base_16x224(num_classes=num_classes, **kwargs)
        head = _replace_last_linear(net, num_classes)  # ensure a fresh head + handle
    except (RuntimeError, TypeError):
        # Older signature, or head/checkpoint size mismatch: load the backbone,
        # then retarget its classifier to our class count.
        net = _hiera.hiera_base_16x224(**kwargs)
        head = _replace_last_linear(net, num_classes)
    return net, head


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class EmbeddingTemporalModel(nn.Module):
    """2D frame encoder + GRU with attention pooling over the time axis."""

    def __init__(
        self,
        num_classes: int,
        hidden: int = 256,
        pretrained: bool = True,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.encoder, feat_dim = _resnet18(pretrained)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self._frozen = freeze_encoder
        self.gru = nn.GRU(feat_dim, hidden, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(2 * hidden, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(2 * hidden, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, K, C, H, W)
        b, k, c, h, w = x.shape
        feats = self._encode(x.reshape(b * k, c, h, w)).reshape(b, k, -1)
        seq, _ = self.gru(feats)                  # (B, K, 2*hidden)
        w_attn = torch.softmax(self.attn(seq), dim=1)  # (B, K, 1)
        pooled = (seq * w_attn).sum(dim=1)        # (B, 2*hidden)
        return self.classifier(pooled)

    def _encode(self, frames: torch.Tensor) -> torch.Tensor:
        if self._frozen:
            with torch.no_grad():
                return self.encoder(frames)
        return self.encoder(frames)


class VideoModel(nn.Module):
    """R(2+1)D-18 video classifier. Expects (B, K, C, H, W); permutes inside."""

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.net = _r2plus1d_18(pretrained, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, K, C, H, W)
        return self.net(x.permute(0, 2, 1, 3, 4))        # -> (B, C, K, H, W)


class PermuteVideoModel(nn.Module):
    """Generic wrapper for a pretrained video backbone in the linear-probe regime.

    Takes the temporal clip ``(B, K, C, H, W)`` and permutes it to the
    channels-first-time ``(B, C, K, H, W)`` layout every torchvision video net
    (and Hiera) expects. When ``freeze_backbone`` is set, every parameter outside
    the freshly attached ``head`` has ``requires_grad=False``, so only the new
    classifier trains. PyTorch then skips storing backbone activations for the
    backward pass, keeping peak memory near a single forward — what lets these
    K400 transformers fine-tune inside 8 GB.
    """

    def __init__(
        self,
        net: nn.Module,
        head: nn.Module,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.net = net
        if freeze_backbone:
            trainable = {id(p) for p in head.parameters()}
            for p in self.net.parameters():
                if id(p) not in trainable:
                    p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, K, C, H, W)
        return self.net(x.permute(0, 2, 1, 3, 4))        # -> (B, C, K, H, W)


_VIDEO_LOADERS = {"mvit": _mvit_v2_s, "swin3d": _swin3d_t, "hiera": _hiera_base}


def build_model(kind: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if kind == "head":
        return EmbeddingTemporalModel(num_classes, pretrained=pretrained)
    if kind == "video":
        return VideoModel(num_classes, pretrained=pretrained)
    if kind in _VIDEO_LOADERS:
        net, head = _VIDEO_LOADERS[kind](pretrained, num_classes)
        return PermuteVideoModel(net, head, freeze_backbone=True)
    raise ValueError(
        f"unknown model kind: {kind!r} (expected one of {MODEL_KINDS})"
    )


# ---------------------------------------------------------------------------
# Class-name -> unsafe index mapping + probability helpers
# ---------------------------------------------------------------------------
def unsafe_indices(class_names: List[str]) -> List[int]:
    """Indices of classes whose name maps to the *unsafe* category."""
    return [i for i, name in enumerate(class_names) if classify_label(name) == "unsafe"]


def unsafe_prob(probs: torch.Tensor, unsafe_idx: List[int]) -> torch.Tensor:
    """Sum softmax mass over the unsafe classes -> (B,) probability in [0, 1]."""
    if not unsafe_idx:
        return torch.zeros(probs.shape[0], device=probs.device)
    return probs[:, unsafe_idx].sum(dim=1)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: str,
    model: nn.Module,
    kind: str,
    class_names: List[str],
    window: int,
    img_size: int,
) -> None:
    torch.save(
        {
            "kind": kind,
            "class_names": list(class_names),
            "window": int(window),
            "img_size": int(img_size),
            "unsafe_idx": unsafe_indices(class_names),
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_checkpoint(
    path: str, device: Optional[str] = None
) -> Tuple[nn.Module, Dict]:
    """Load a temporal checkpoint -> (model in eval mode, config dict)."""
    ckpt = torch.load(path, map_location=device or "cpu")
    model = build_model(ckpt["kind"], len(ckpt["class_names"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    if device:
        model.to(device)
    config = {
        "kind": ckpt["kind"],
        "class_names": ckpt["class_names"],
        "window": ckpt["window"],
        "img_size": ckpt["img_size"],
        "unsafe_idx": ckpt.get("unsafe_idx", unsafe_indices(ckpt["class_names"])),
    }
    return model, config
