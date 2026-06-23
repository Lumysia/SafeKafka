# SafeStream-Kafka

Real-time safe / unsafe behaviour analytics for IoT workplace video streams, powered by a
fine-tuned YOLOv8 detector and Apache Kafka (Confluent Cloud).

Three independent services communicate through three Kafka topics:

```
CCTV / RTSP / video file
        │
        ▼
[ safestream.producer ] ── publishes frames ──▶  cctv-frames
        │
        ▼
[ safestream.detector ] ── YOLOv8 + safe/unsafe labels ──▶  safety-detections
        │
        ▼
[ safestream.aggregator ] ── rolling + cumulative totals ──▶  safety-alerts
        │
        ▼
[ safestream.dashboard ] ── subscribes, renders live HTML ──▶  http://localhost:8000
```

## Requirements

- macOS on Apple Silicon (M1 / M2 / M3 / M4) or Linux
- Python 3.10 – 3.12
- Homebrew (only needed for the optional local Kafka via Docker)
- A YOLOv8 weights file (`yolov8m.pt` for the COCO baseline, or your own `best.pt`)

## Docker Compose demo

For the fastest live demonstration, use the bundled Docker Compose stack. It starts Kafka,
creates topics, launches the dashboard, detector, and producer, then replays the included
demo videos through the pipeline.

```bash
docker compose up --build
```

Then open <http://localhost:8000>. The producer replays one bundled clip per behaviour
class from `demo_videos/`, the detector uses the 8-class `previous_weights/best.pt`, and
the dashboard shows continuous MJPEG camera feeds plus live `cam-01` / `cam-02` safe and
unsafe totals.

To run in the background:

```bash
docker compose up --build -d
docker compose logs -f dashboard detector producer
```

Stop everything with:

```bash
docker compose down
```

## Mac M1 setup

```bash
# 1. Clone or unzip the project, then cd into it
cd safestream-kafka

# 2. Create and activate a virtualenv (arm64 Python from python.org or pyenv)
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Copy the env template and fill in your Confluent Cloud creds
cp .env.example .env
# edit .env: KAFKA_BOOTSTRAP_SERVERS, KAFKA_API_KEY, KAFKA_API_SECRET

# 5. (Optional) start a local Kafka broker if you don't have Confluent Cloud
docker compose up -d
# then set USE_LOCAL_BROKER=true in .env

# 6. Create the three Kafka topics
python -m scripts.create_topics
```

YOLOv8 inference uses Apple's Metal Performance Shaders (MPS) on M-series Macs automatically
when available. The detector logs which device it picked at startup.

## Windows (PowerShell) setup

```powershell
# 1. Clone or unzip the project, then cd into it
cd SafeKafka

# 2. Create and activate a virtualenv
py -m venv .venv

.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Copy the env template and fill in your Confluent Cloud creds
copy .env.example .env
# edit .env: KAFKA_BOOTSTRAP_SERVERS, KAFKA_API_KEY, KAFKA_API_SECRET

# 5. (Optional) start a local Kafka broker
docker compose up -d
# then set USE_LOCAL_BROKER=true in .env

# 6. Create the three Kafka topics
py -m scripts.create_topics
```

### GPU support on Windows

If you have an **NVIDIA GPU**, install the latest Game Ready or Studio driver on Windows
(≥ 527.41). This automatically enables GPU passthrough for both native Windows and WSL2 —
no separate CUDA install needed. Verify with `nvidia-smi` in PowerShell; if it shows your
GPU, `--device auto` in the training script will pick it up.

## Running the four services

Open four terminals (or use the VS Code launch configurations under `.vscode/launch.json`):

```bash
# Terminal 1 — dashboard (start this first so you can watch the totals)
python -m safestream.dashboard

# Terminal 2 — aggregator
python -m safestream.aggregator

# Terminal 3 — detector
python -m safestream.detector --weights previous_weights/best.pt

# Terminal 4 — producer (drives the pipeline)
python -m safestream.producer --video-dir demo_videos/cam_01 --camera-id cam-01 --loop --realtime

python -m safestream.producer --video-dir demo_videos/cam_02 --camera-id cam-02 --loop --realtime
```

Then open <http://localhost:8000> and watch per-camera `total_safe` / `total_unsafe`
and the rolling unsafe ratio update in real time.

### Demo: seeing the aggregator's impact

The aggregator's job is **alert noise-reduction** — it collapses raw per-frame unsafe
detections into a few windowed, rate-limited alerts (`AGG_UNSAFE_RATIO_ALERT`,
`AGG_MIN_WINDOW_OBS`, `AGG_ALERT_COOLDOWN_SECONDS`). To see that impact, run the dashboard
with `AGG_ENABLED=false`:

```bash
# PowerShell:  $env:AGG_ENABLED="false"; python -m safestream.dashboard
AGG_ENABLED=false python -m safestream.dashboard
```

With it **off**, the dashboard bypasses the window/threshold/cooldown logic and fires one
alert per frame that contains any unsafe detection — the "dozens of alerts" firehose. A
header badge shows the active mode. Cumulative/rolling stats still populate, so you can
replay the same clips on vs off and compare the alert volume directly. (This toggle only
affects the in-process dashboard; the standalone `safestream.aggregator` is unaffected.)

### Running from VS Code

The `.vscode/launch.json` file ships four debug configurations — Producer, Detector,
Aggregator, Dashboard — plus a compound configuration **"SafeStream: all services"**
that starts everything at once.

## Configuration cheat sheet

All knobs are read from `.env` (see `.env.example` for the full list):

| Variable                             | Default             | Notes                                                    |
| ------------------------------------ | ------------------- | -------------------------------------------------------- |
| `KAFKA_BOOTSTRAP_SERVERS`            | —                   | Confluent Cloud bootstrap endpoint                       |
| `KAFKA_API_KEY` / `KAFKA_API_SECRET` | —                   | SASL credentials                                         |
| `USE_LOCAL_BROKER`                   | `false`             | If true, ignore SASL and use plaintext localhost broker  |
| `TOPIC_FRAMES`                       | `cctv-frames`       | Producer → Detector                                      |
| `TOPIC_DETECTIONS`                   | `safety-detections` | Detector → Aggregator                                    |
| `TOPIC_ALERTS`                       | `safety-alerts`     | Aggregator → downstream sinks                            |
| `AGG_WINDOW_SECONDS`                 | `60`                | Rolling-window length                                    |
| `AGG_UNSAFE_RATIO_ALERT`             | `0.30`              | Ratio that triggers a WARN alert                         |
| `AGG_MIN_WINDOW_OBS`                 | `5`                 | Minimum rolling-window obs before alerting               |
| `AGG_ENABLED`                        | `true`              | `false` = naive per-frame alerting (see demo note below) |
| `DETECTOR_DEVICE`                    | `auto`              | `auto`, `mps`, `cuda`, or `cpu`                          |
| `DETECTOR_CONF`                      | `0.25`              | YOLOv8 confidence threshold                              |
| `DASHBOARD_HOST`                     | `127.0.0.1`         | FastAPI bind host                                        |
| `DASHBOARD_PORT`                     | `8000`              | FastAPI bind port                                        |

## Dashboard metrics

The dashboard shows live streaming metrics. Offline detector/classifier metrics below were
computed with `previous_weights/best.pt` on `yolo_dataset/test` (`1500` sampled test frames).
This project uses YOLOv8 **classification** weights.

This is a **classification** model, so the headline quality numbers are classification
metrics — not bbox/IoU `mAP`. (Earlier versions of this table labelled the two AP figures
below as `mAP @ 0.5` / `mAP @ 0.5:0.95`; that naming was misleading and has been corrected.)

| Metric                    | Value                                | Notes                                                                                              |
| ------------------------- | ------------------------------------ | -------------------------------------------------------------------------------------------------- |
| Binary unsafe AP (PR-AUC) | `0.887`                              | Area under the precision–recall curve for the `unsafe` score. Classification AP, **not** bbox mAP. |
| 8-class macro AP          | `0.729`                              | Macro-averaged one-vs-rest AP across the 8 classes. Classification AP, **not** bbox mAP.           |
| Detector precision        | `0.769`                              | Binary unsafe precision on the held-out test split.                                                |
| Detector recall           | `0.841`                              | Binary unsafe recall on the held-out test split.                                                   |
| Binary F1 score           | `0.803`                              | Binary unsafe-vs-safe F1 on the held-out test split.                                               |
| Accuracy                  | `0.789`                              | Binary safe-vs-unsafe accuracy. See the imbalance caveat below.                                    |
| 8-class top-1 accuracy    | `0.641`                              | Exact 8-class classification accuracy (see the class-name canonicalization note below).            |
| Confusion matrix          | `TP=646`, `FP=194`, `FN=122`, `TN=538` | Positive class is `unsafe`; rows are true unsafe/safe.                                           |
| Alerts per minute         | live dashboard                       | Count of generated alerts in the last 60 seconds.                                                  |
| End-to-end latency        | live dashboard                       | Frame timestamp to dashboard detection consumption, rolling average/max over the last 60 seconds.  |
| Throughput                | live dashboard                       | `safety-detections` messages per second over the last 60 seconds.                                  |

**How to read these.** The test set is roughly balanced — `768/1500` (~51%) frames are
`unsafe` — so the trivial "always unsafe" baseline scores only ~`0.512` accuracy; the honest
headline numbers are **F1 (`0.803`) and recall (`0.841`)**, not accuracy. The model errs toward
false alarms over misses (`FP=194` vs `FN=122`), which is the desirable bias for a safety monitor.
Metrics are also **frame-level** (`125` clips × `12` sampled frames), so frames from the same clip
are correlated and the effective sample size is smaller than `1500`.

**Class-name canonicalization.** `scripts/evaluate.py` normalizes class names before the 8-class
comparison (`_canon`: strip a leading `N_` index prefix, lowercase), so the model's
`Safe_Walkway_Violation` matches the dataset directory `0_safe_walkway_violation`. Without it the
8-class top-1 / macro-AP score 0 even though the binary metrics (which route through
`classify_label`) look fine.

**Reproduce these numbers.** They are produced by `scripts/evaluate.py`, which runs the exact
production inference path on the classification test split and writes `eval_results.json`. The
dashboard loads that file when present (served at `/api/metrics`) and falls back to the same
baked-in values otherwise:

```bash
uv run python -m scripts.evaluate --weights previous_weights/best.pt --data yolo_dataset
```

Live dashboard F1/confusion matrix are demo-only because the bundled demo clip filenames
encode ground-truth class IDs. Arbitrary camera streams still show throughput, latency,
alerts, sliding-window ratios, and class distributions, but evaluation metrics need labels.

## Training your own YOLOv8 classifier

The dataset has clip-level labels (each clip is one behaviour, no bounding boxes),
so we train YOLOv8 in **classification** mode. `scripts/prepare_dataset.py` samples
evenly spaced frames from every clip and writes a classification directory tree
(`train/<class>/*.jpg`, `val/...`, `test/...`) — no `dataset.yaml` needed.

```bash
# install runtime dependencies with uv, then add dataset tooling for training
uv sync
uv pip install fiftyone fiftyone-db huggingface_hub

# default: pull clips from the Voxel51 hub dataset and prepare all 8 classes
uv run python -m scripts.prepare_dataset --out yolo_dataset --frames-per-clip 12

# OR use download videos directly from source link https://data.mendeley.com/datasets/xjmtb22pff/1) and use local folders instead
uv run python -m scripts.prepare_dataset --src path/to/videos --out yolo_dataset --frames-per-clip 12

# train your own model if necessary
# imgsz defaults to 224
uv run python -m scripts.train_yolo --data yolo_dataset --epochs 50 --device auto
```

This produces a `*-cls` model at `runs/safestream_yolov8m/weights/best.pt`,
which you pass to the detector via `--weights`. The detector auto-detects
classification weights and emits one labelled result per frame (no bounding box):

```bash
uv run python -m safestream.detector --weights runs/safestream_yolov8m/weights/best.pt
```

## Temporal models & SOTA ablation

Beyond the per-frame YOLO classifier, the repo includes a **temporal** model family
([safestream/temporal/](safestream/temporal/)) that classifies a *window of frames* per clip, plus
a one-command ablation benchmarking them against the per-frame baseline and against modern
video-recognition SOTA. A single factory `build_model(kind, num_classes)` builds every kind:

| `kind`   | backbone                    | regime                          | input |
| -------- | --------------------------- | ------------------------------- | ----- |
| `head`   | ResNet-18 + GRU/attention   | frozen encoder, head trained    | 224   |
| `video`  | R(2+1)D-18 (torchvision)    | full fine-tune                  | 112   |
| `mvit`   | MViTv2-S (Kinetics-400)     | backbone frozen (linear probe)  | 224   |
| `swin3d` | Video Swin-T (Kinetics-400) | backbone frozen (linear probe)  | 224   |
| `hiera`  | Hiera-B (Kinetics-400)      | backbone frozen (linear probe)  | 224   |

All train at `--window 16`; `mvit`/`hiera` *require* window 16 (fixed positional embeddings). The
transformers train only a ~6K-param head, so they fit in 8 GB; the first run of each downloads its
Kinetics-400 weights (needs internet once). `hiera` needs the `hiera-transformer` package (already
in `requirements.txt`). Set `DETECTOR_MODE=temporal` + `TEMPORAL_WEIGHTS` to run one live in the
detector instead of YOLO.

```bash
# train one temporal model on the clip manifest
python -m scripts.train_temporal --data clips_manifest.csv --model hiera --window 16 --epochs 30

# offline clip-level metrics (same schema as scripts/evaluate.py)
python -m scripts.evaluate_temporal --weights runs/ablation_hiera/best.pt --data clips_manifest.csv

# streaming alert metrics, calibrated per model (see below)
python -m scripts.evaluate_streaming --weights runs/ablation_hiera/best.pt \
    --split test --calibrate-split val

# run the whole ablation (all 5 kinds + the YOLO baseline) -> ablation_results.{json,md}
python -m scripts.ablation --data clips_manifest.csv --epochs 30
# regenerate the table from existing checkpoints, without retraining:
python -m scripts.ablation --data clips_manifest.csv --reuse-weights
```

### Per-model threshold calibration

The "smart" aggregator fires when an EWMA of a model's `unsafe_prob` crosses `AGG_ENTER_THRESHOLD`.
But each model emits `unsafe_prob` on a different scale, so one global threshold is unfair: at the
default `0.5` most models alarm on nearly every clip. `evaluate_streaming --calibrate-split val`
fixes this — it sweeps the enter/exit thresholds on the **val** split, picks each model's threshold
at a matched operating point (default: lowest false-alert rate among thresholds with
alert-recall ≥ 0.95), and reports on the **test** split. The chosen thresholds land in a
`calibration` block in the output JSON and the "Smart enter-thr (cal)" column. The per-frame YOLO
baseline goes through the *same* path (`--model-type yolo`), so its streaming row is directly
comparable.

### Results & findings

Full run (window 16, 30 epochs; streaming calibrated to recall ≥ 0.95 on val, reported on test):

| Model               | Offline unsafe AP | F1        | 8-cls top-1 | Smart false-alert | enter-thr (cal) | Alert P / R (test) | Latency ms/frame |
| ------------------- | ----------------- | --------- | ----------- | ----------------- | --------------- | ------------------ | ---------------- |
| YOLOv8 (per-frame)  | **0.887**         | **0.803** | **0.641**   | **0.311**         | 0.60            | **0.732** / 0.812  | **6.4**          |
| head (ResNet18+GRU) | 0.862             | 0.704     | 0.440       | 0.557             | 0.65            | 0.634 / 0.922      | **13.4**         |
| video (R(2+1)D-18)  | 0.755             | 0.695     | 0.472       | 0.705             | 0.65            | 0.578 / 0.922      | 26.5             |
| MViTv2-S            | 0.795             | 0.707     | 0.488       | 0.852             | 0.65            | 0.544 / 0.969      | 48.8             |
| Video Swin-T        | 0.805             | 0.702     | 0.480       | 0.623             | 0.70            | 0.596 / 0.875      | 39.8             |
| Hiera-B             | 0.862             | **0.727** | 0.536       | 0.787             | 0.45            | 0.571 / **1.000**  | 47.5             |

_Latency = wall-clock ms per incoming frame through each model's `.infer()` wrapper, measured
identically for every row (warm-up, then a timed loop, `torch.cuda.synchronize()` on CUDA). The
temporal models re-run the full 16-frame window every frame; YOLO classifies a single frame — so the
column is directly comparable across rows._

**Headline: temporal modeling does not beat the per-frame baseline on this dataset.**

1. **Offline, YOLO wins outright** — best binary AP (0.887), F1 (0.803), and 8-class top-1 (0.641);
   no temporal model beats it. These violations are mostly *appearance/state* (panel open vs closed,
   person in walkway), visible in a single frame, so aggregating 16 frames adds little. *(Caveat:
   YOLO AP is over 1,500 frames, temporal AP over 125 clips — related but not identical units.)*
2. **Offline quality ≠ streaming quality.** `head` and `Hiera` have identical offline AP (0.862) but
   opposite streaming false-alert rates (0.557 vs 0.787). Choosing a model by offline F1 would pick
   one of the *worst* alert streams — the threshold + aggregator layer matters as much as the model.
3. **Calibration is the biggest lever.** No model is well-calibrated at the default 0.5 (chosen
   thresholds span 0.45–0.70). Before calibration Hiera *looked* like the best streamer (FA 0.377) —
   but only because it was being scored at a recall it never actually reached; at matched recall it's
   near the bottom.
4. **The per-frame baseline also streams best — closing the gap.** Run through the *same* aggregator,
   YOLO has the **lowest false-alert rate (0.311)** and **highest alert precision (0.732)** of any
   model. It does so at a lower test recall (0.812 vs 0.88–1.00), because its val→test recall gap is
   larger — so it's a precision/recall trade, not a strict domination. But the upshot stands: the
   shipped *per-frame YOLO → EWMA/hysteresis aggregator* pipeline produces a competitive-or-better
   alert stream, and the heavyweight 3D video transformers don't justify their cost.
5. **Cost.** The shipped per-frame YOLO is also the cheapest to run — **6.4 ms/frame**, ~2× faster
   than the cheap GRU `head` (13.4) and 6–8× faster than the transformer linear-probes (40–48
   ms/frame), which push a full 3D backbone over the whole 16-frame window every frame. So the
   baseline wins on streaming alert quality *and* latency; the heavyweight temporal nets cost more for
   no accuracy gain.

**How to read the streaming columns.** Operating points are matched on *val* recall (≥ 0.95), so the
*test* recall varies per row (0.81–1.00) — read false-alert rate alongside recall, not in isolation.
Val has few safe clips, so the reported false-alert rate is the *test* number (over 125 clips); the
tie-break on the highest qualifying threshold keeps the choice stable. The takeaway for deployment:
**invest in per-model (and per-camera) threshold calibration and the aggregator, not a heavier
backbone.**

## Project layout

```
safestream-kafka/
├── README.md
├── requirements.txt
├── .env.example          # template — copy to .env and fill in
├── .gitignore
├── docker-compose.yml    # local Confluent Platform (optional)
├── .vscode/              # launch configs and editor settings
├── safestream/           # importable Python package
│   ├── settings.py       # central config loaded from .env
│   ├── common/           # shared helpers (Kafka clients, encoding, labels)
│   ├── producer/         # video → cctv-frames
│   ├── detector/         # cctv-frames → YOLOv8 (or temporal model) → safety-detections
│   ├── temporal/         # temporal model family: build_model factory + clip-window dataset
│   ├── aggregator/       # safety-detections → safety-alerts + dashboard state
│   └── dashboard/        # FastAPI + WebSocket front-end
├── scripts/
│   ├── create_topics.py
│   ├── prepare_dataset.py     # videos → YOLO classification dataset tree
│   ├── train_yolo.py          # train a *-cls model on that tree
│   ├── evaluate.py            # recompute offline YOLO metrics → eval_results.json
│   ├── train_temporal.py      # train a temporal model (head/video/mvit/swin3d/hiera)
│   ├── evaluate_temporal.py   # offline clip-level metrics for a temporal model
│   ├── evaluate_streaming.py  # streaming alert metrics + per-model threshold calibration
│   └── ablation.py            # train+eval every kind → ablation_results.{json,md}
└── tests/
    └── test_aggregator.py
```
