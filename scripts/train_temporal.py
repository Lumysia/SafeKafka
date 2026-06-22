"""Train a temporal safety-behaviour model on the clip manifest.

Two model kinds (see safestream/temporal/model.py):
    --model head   : ResNet-18 frame encoder + GRU/attention head (light, default)
    --model video  : R(2+1)D-18 3D-conv video net (heavier, stronger ablation)

Saves the best-val checkpoint to <project>/<name>/best.pt (a dict carrying the
class names, window length and unsafe-class indices) plus a sidecar
temporal_config.json for quick inspection.

Usage:
    python -m scripts.train_temporal --data clips_manifest.csv --model head --epochs 30
    python -m scripts.train_temporal --data clips_manifest.csv --model video --imgsz 112
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from safestream.common.encoding import detect_device
from safestream.temporal.dataset import ClipWindowDataset, read_manifest
from safestream.temporal.model import (
    DEFAULT_IMG_SIZE,
    MODEL_KINDS,
    build_model,
    save_checkpoint,
    unsafe_indices,
)

# Backbones built around fixed-length positional embeddings: they only accept a
# 16-frame window (their pretrained Kinetics-400 setting).
_FIXED_WINDOW_16 = {"mvit", "hiera"}


def _run_epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    for clips, targets in loader:
        clips, targets = clips.to(device), targets.to(device)
        with torch.set_grad_enabled(train):
            logits = model(clips)
            loss = criterion(logits, targets)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        loss_sum += float(loss.item()) * targets.size(0)
        correct += int((logits.argmax(1) == targets).sum().item())
        total += targets.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="clips_manifest.csv")
    p.add_argument("--model", default="head", choices=list(MODEL_KINDS))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=None,
                   help="Frame size (default: 224 for head, 112 for video)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--project", default="runs")
    p.add_argument("--name", default="temporal")
    args = p.parse_args()

    if args.model in _FIXED_WINDOW_16 and args.window != 16:
        print(
            f"--model {args.model} requires --window 16 (fixed positional "
            f"embeddings); got {args.window}.",
            file=sys.stderr,
        )
        return 2

    img_size = args.imgsz or DEFAULT_IMG_SIZE[args.model]
    device = detect_device(args.device)
    print(f"Training kind={args.model} device={device} imgsz={img_size} window={args.window}")

    # Fix class ordering from the train split so every split shares indices.
    _, class_names = read_manifest(args.data, "train")
    if not class_names:
        print("No train clips in manifest.", file=sys.stderr)
        return 2
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Unsafe class indices: {unsafe_indices(class_names)}")

    pin = device.startswith("cuda")

    def make_loader(split, train):
        ds = ClipWindowDataset(args.data, split, window=args.window,
                               img_size=img_size, class_names=class_names, train=train)
        kw = {}
        if args.workers > 0:
            kw.update(persistent_workers=True, prefetch_factor=4)
        return DataLoader(ds, batch_size=args.batch, shuffle=train,
                          num_workers=args.workers, drop_last=train,
                          pin_memory=pin, **kw)

    train_loader = make_loader("train", True)
    val_loader = make_loader("val", False)

    model = build_model(args.model, len(class_names), pretrained=True).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    params = [q for q in model.parameters() if q.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    out_dir = Path(args.project) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    best_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, device, criterion, optimizer)
        va_loss, va_acc = _run_epoch(model, val_loader, device, criterion)
        flag = ""
        if va_acc >= best_acc:
            best_acc = va_acc
            save_checkpoint(str(best_path), model, args.model, class_names,
                            args.window, img_size)
            flag = "  <- best"
        print(f"epoch {epoch:3d}  train_loss={tr_loss:.3f} acc={tr_acc:.3f}  "
              f"val_loss={va_loss:.3f} acc={va_acc:.3f}{flag}")

    (out_dir / "temporal_config.json").write_text(
        json.dumps({"kind": args.model, "class_names": class_names,
                    "window": args.window, "img_size": img_size,
                    "unsafe_idx": unsafe_indices(class_names),
                    "best_val_acc": best_acc}, indent=2),
        encoding="utf-8",
    )
    print(f"\nDone. Best val acc={best_acc:.3f}. Weights: {best_path}")
    print(f"Evaluate: python -m scripts.evaluate_temporal --weights {best_path} --data {args.data}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
