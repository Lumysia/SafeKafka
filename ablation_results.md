# SOTA ablation — temporal safety-behaviour models

Window=16, epochs=30, data=`clips_manifest.csv`.

**Controlled:** same manifest splits & `sample_frames` selection, window=16, epochs, seed, AdamW, and identical `evaluate_temporal`/`evaluate_streaming` metric code.  Smart thresholds are calibrated per model on the val split to alert-recall >= 0.95, so each smart column is that model's val operating point applied to test (apples-to-apples across models).  
**Inherent differences (see Regime):** input resolution (112 for R(2+1)D-18, 224 for the rest) and fine-tune regime.

| Model | Params (train/total) | Offline unsafe AP | F1 | 8-cls top-1 | Smart false-alert rate | Smart enter-thr (cal) | Alert P/R | Latency ms/frame | Regime |
|---|---|---|---|---|---|---|---|---|---|
| YOLOv8 (per-frame) | — | 0.887 | 0.803 | 0.641 | 0.311 | 0.60 | 0.732 / 0.812 | 6.40 | per-frame classifier |
| head (ResNet18+GRU) | 1.19M / 12.4M | 0.862 | 0.704 | 0.440 | 0.557 | 0.65 | 0.634 / 0.922 | 13.63 | frozen encoder + GRU head |
| video (R(2+1)D-18) | 31.30M / 31.3M | 0.755 | 0.695 | 0.472 | 0.705 | 0.65 | 0.578 / 0.922 | 26.39 | full fine-tune |
| MViTv2-S | 0.01M / 34.2M | 0.795 | 0.707 | 0.488 | 0.852 | 0.65 | 0.544 / 0.969 | 48.38 | frozen backbone (linear probe) |
| Video Swin-T | 0.01M / 27.9M | 0.805 | 0.702 | 0.480 | 0.623 | 0.70 | 0.596 / 0.875 | 39.76 | frozen backbone (linear probe) |
| Hiera-B | 0.01M / 50.8M | 0.862 | 0.727 | 0.536 | 0.787 | 0.45 | 0.571 / 1.000 | 46.96 | frozen backbone (linear probe) |
