# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

SafeStream-Kafka is a real-time safe/unsafe behaviour analytics pipeline for IoT workplace
video. Four Python services communicate exclusively through three Kafka topics (Confluent
Cloud or a local broker). A fine-tuned YOLOv8 model labels frames; an aggregator computes
rolling/cumulative safety stats and fires alerts; a FastAPI dashboard renders everything live.

```
video/RTSP/webcam → producer → [cctv-frames] → detector → [safety-detections] → aggregator → [safety-alerts]
                                     │                                                │
                                     └────────────────── dashboard ──────────────────┘  → http://localhost:8000
```

## Commands

Dependencies are managed with **uv** (`uv.lock`, `pyproject.toml`); `requirements.txt` exists
for plain-pip installs. On this Windows machine a `.venv` is already present.

```powershell
# Tests (no Kafka or YOLO needed — pure aggregator/label logic)
python -m pytest                         # all
python -m pytest tests/test_aggregator.py::test_alert_fires_when_ratio_crosses_threshold  # one

# Run the four services (each is a python -m module; open separate terminals)
python -m safestream.dashboard           # start first; serves :8000
python -m safestream.aggregator          # OPTIONAL standalone — see note below
python -m safestream.detector --weights previous_weights/best.pt
python -m safestream.producer --video-dir demo_videos/cam_01 --camera-id cam-01 --loop --realtime

# One-time setup against a fresh broker
python -m scripts.create_topics          # creates the 3 topics

# Full demo with bundled local Kafka + all services
docker compose up --build                # then open http://localhost:8000

# Temporal models + SOTA ablation (no Kafka; uses clips_manifest.csv)
python -m scripts.train_temporal --data clips_manifest.csv --model hiera --window 16 --epochs 30
python -m scripts.evaluate_temporal  --weights runs/ablation_hiera/best.pt --data clips_manifest.csv
python -m scripts.evaluate_streaming --weights runs/ablation_hiera/best.pt --split test --calibrate-split val
python -m scripts.ablation --data clips_manifest.csv               # full run, all 5 kinds
python -m scripts.ablation --data clips_manifest.csv --reuse-weights  # regen table, no retrain
```

There is no linter/formatter configured. `.vscode/launch.json` has debug configs for each
service plus a compound "SafeStream: all services".

⚠️ **This Windows console is cp1252.** Scripts print to a cp1252 terminal, so non-ASCII in
`print()` output (`≥`, `→`, `✓`, `—`) raises `UnicodeEncodeError` mid-run — keep stdout ASCII
(`>=`, `->`). Markdown/JSON files are written UTF-8 and are unaffected, so Unicode is fine in
`*.md`/`*.json` content (just not in things printed to the console).

## Configuration

Everything is driven by environment variables read once at import in
[safestream/settings.py](safestream/settings.py) (loads `.env` if present; copy from
`.env.example`). `get_settings()` is `lru_cache`d — settings are effectively immutable for a
process's lifetime. Set `USE_LOCAL_BROKER=true` to use the docker-compose plaintext broker
instead of Confluent Cloud SASL credentials.

## Architecture notes (the non-obvious parts)

**Message contracts are implicit JSON** — there are no shared schema classes. Each service
hand-builds a dict and `json.dumps` it. The keys flow: producer emits
`{camera_id, frame_id, source, timestamp, image_b64}` → detector emits
`{camera_id, frame_id, source, timestamp, detections[], safe_count, unsafe_count, total_detections}`
→ aggregator emits alert dicts. Messages are **keyed by `camera_id`** so each camera maps to
one ordered partition. Frames are JPEG→base64→JSON (see
[common/encoding.py](safestream/common/encoding.py)).

**The detector handles both YOLO model types.** [detector/__main__.py](safestream/detector/__main__.py)
inspects `preds[0].probs`: if present it's a **classification** model (one top-1 label per
frame, no bbox); otherwise it iterates `preds[0].boxes` as a **detection** model. This repo's
shipped `previous_weights/best.pt` is a classification model. `ultralytics`/`torch` are
imported lazily inside the detector so the other services don't pay the import cost.

**Safe/unsafe classification is keyword-based data, not code.**
[common/labels.py](safestream/common/labels.py) maps raw YOLO class names to
`safe`/`unsafe`/`other` via keyword tables; an *unsafe* keyword always wins over a *safe* one
(so `no_safety_vest` → unsafe). To support a new model's class names, edit the keyword tuples,
not the consumers.

**`SafetyAggregator` is a shared, thread-safe stateful class**
([aggregator/aggregator.py](safestream/aggregator/aggregator.py)) used two ways:
1. The dashboard instantiates it **in-process** and a background Kafka thread feeds it
   (`_kafka_loop` in [dashboard/__main__.py](safestream/dashboard/__main__.py)). The dashboard
   *also* re-publishes alerts to `safety-alerts` itself.
2. [aggregator/__main__.py](safestream/aggregator/__main__.py) is an **optional standalone**
   service doing the same thing for a separate-container deployment.

   ⚠️ Because both the dashboard and the standalone aggregator consume `safety-detections` and
   publish to `safety-alerts`, running both at once duplicates alerts. The standard local/Docker
   setup runs the dashboard only; the standalone aggregator is for split deployments.

**The dashboard does a lot beyond the aggregator.** It runs two Kafka consumer threads
(`_kafka_loop` for detections, `_frames_loop` for raw frames), serves MJPEG camera streams
(`/api/stream/{camera_id}`), draws bbox overlays with OpenCV, tracks live latency/throughput,
and pushes a combined snapshot over a WebSocket (`/ws`) at ~1 Hz. The single-file frontend is
[dashboard/static/index.html](safestream/dashboard/static/index.html).

**Live F1/confusion metrics are demo-only.** `_truth_from_source` in the dashboard derives
ground truth from the clip *filename prefix* (`0–3` → unsafe, `4–7` → safe). Arbitrary camera
streams have no labels, so only throughput/latency/ratios are meaningful for them. Offline
metrics in `/api/metrics` are loaded from `eval_results.json` at the repo root when present
(falling back to hard-coded constants otherwise); regenerate the file with
`python -m scripts.evaluate`, which mirrors the detector's exact inference path. Note these are
classification metrics (binary unsafe AP / 8-class macro AP), **not** bbox IoU mAP. `evaluate.py`
**canonicalizes class names** (`_canon`: strip a leading `N_` index prefix, lowercase) so the
model's `Safe_Walkway_Violation` lines up with the dataset dir `0_safe_walkway_violation`; comparing
the raw strings silently scores 8-class top-1 / macro-AP as 0 (the binary metrics survive via
`classify_label`).

**Device selection** (`detect_device` in common/encoding.py): `auto` resolves to MPS on Apple
Silicon, then CUDA, then CPU. Override per-run with `--device` or `DETECTOR_DEVICE`.

**Temporal models are a second model family with one factory.**
[temporal/model.py](safestream/temporal/model.py) `build_model(kind, num_classes)` builds any of
`MODEL_KINDS = ("head", "video", "mvit", "swin3d", "hiera")`: the in-repo ResNet18+GRU `head`, a
torchvision R(2+1)D-18 `video` net, and three Kinetics-400-pretrained video transformers (MViTv2-S,
Video Swin-T, Hiera-B). The three transformers are **linear-probed** — `PermuteVideoModel` freezes
the backbone and trains only a fresh ~6K-param head, which is what keeps them inside 8 GB.
Checkpoints carry their `kind`, so `load_checkpoint` → `evaluate_temporal` → `evaluate_streaming` →
the live `TemporalInfer` detector all work for any kind with **no other change** (this is why adding
a kind is just registering it in the factory). `mvit`/`hiera` have fixed positional embeddings and
**require `--window 16`**. Run one live instead of YOLO with `DETECTOR_MODE=temporal` +
`TEMPORAL_WEIGHTS` (the detector imports torch/ultralytics lazily either way). `hiera` needs the
`hiera-transformer` dependency; the first run of any transformer downloads its K400 weights (needs
internet once).

**The SOTA ablation is one runner.** [scripts/ablation.py](scripts/ablation.py) trains + evaluates
every kind under identical conditions by shelling out to `train_temporal` / `evaluate_temporal` /
`evaluate_streaming` (so those stay the single source of truth), measures params + ms/frame latency,
and emits `ablation_results.json` + a paste-ready `ablation_results.md`. The YOLO baseline row's
offline numbers come from `eval_results.json`; its streaming row comes from running the per-frame
YOLO through `evaluate_streaming --model-type yolo` (`--baseline-weights`, on by default).
`--reuse-weights` skips training when `runs/ablation_<kind>/best.pt` already exists (regenerates the
table in minutes); each kind is wrapped in try/except so one failure records an "unavailable" row
and the rest still complete.

**Streaming alert thresholds are calibrated per model — offline quality ≠ streaming quality.**
Every model emits `unsafe_prob` on a different scale, so the single global `AGG_ENTER_THRESHOLD`
(0.5) is wrong for all of them (at 0.5 most models alarm on ~every clip). `evaluate_streaming
--calibrate-split val` sweeps the smart aggregator's enter/exit thresholds on **val** to a recall
target (default ≥ 0.95, lowest false-alert rate among qualifying thresholds), then reports on
**test** — recorded under a `calibration` block in the JSON and the "Smart enter-thr (cal)" column.
Key empirical finding from this repo's runs: a model's offline AP/F1 does **not** predict its
streaming false-alert rate, so pick models on the streaming metric, not offline AP.

## Training

`scripts/prepare_dataset.py` samples evenly spaced frames from clip-level-labelled videos into
a YOLO **classification** directory tree (no `dataset.yaml`). `scripts/train_yolo.py` trains a
`*-cls` model. Pass the resulting `best.pt` to the detector via `--weights`. See the README's
"Training your own YOLOv8 classifier" section for the exact `uv run` invocations.

**Temporal models train from `clips_manifest.csv`, not the YOLO tree.** `scripts/train_temporal.py`
reads the clip manifest (`video_path,label,split`) and trains a windowed model via `ClipWindowDataset`
([temporal/dataset.py](safestream/temporal/dataset.py)), which decodes K frames per clip with the
same `sample_frames` selection as the YOLO pipeline and **caches** decoded/resized frames to
`.clipcache/s<img>_n<n>/` (keyed by image size + frame count, so all 224px models share one cache —
delete the dir to rebuild). The AdamW optimizer filters to `requires_grad` params, so a frozen
backbone trains only its head automatically. `scripts/ablation.py` orchestrates training + both
evaluators across all kinds; prefer it over running the per-kind scripts by hand.
