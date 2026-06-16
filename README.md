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

## Dashboard metrics

The dashboard separates live streaming metrics from offline detector evaluation metrics.

| Metric | Where it appears | How it is computed |
| ------ | ---------------- | ------------------ |
| `mAP @ 0.5` | README / report only | Offline YOLO evaluation on an annotated held-out test split. Not computed live because the stream has no per-frame bounding-box ground truth. |
| `mAP @ 0.5:0.95` | README / report only | Offline YOLO evaluation averaged over IoU thresholds. Not a live dashboard metric. |
| Detector precision | README / report only | Offline detector/classifier evaluation against held-out labels. |
| Detector recall | README / report only | Offline detector/classifier evaluation against held-out labels. |
| Binary F1 score | Dashboard `Demo Evaluation` | Live demo-only binary safe-vs-unsafe F1. Ground truth is inferred from bundled demo clip filename class IDs. |
| Confusion matrix | Dashboard `Demo Evaluation` | Live demo-only `TP unsafe`, `FP unsafe`, `FN unsafe`, `TN safe` counts from the same filename-derived ground truth. |
| Alerts per minute | Dashboard `Live Operations` | Count of generated alerts in the last 60 seconds. |
| End-to-end latency | Dashboard `Live Operations` | Time from frame timestamp to dashboard detection consumption, shown as rolling average and max over the last 60 seconds. |
| Throughput | Dashboard `Live Operations` | `safety-detections` messages per second over the last 60 seconds. |

The live F1/confusion matrix are for the bundled Docker demo only. For arbitrary camera
streams, the dashboard still shows live throughput, latency, alerts, sliding-window ratios,
and class distributions, but binary evaluation requires ground-truth labels.

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
python -m safestream.producer --video-dir path/to/video_directory.mp4 --camera-id cam-01

python -m safestream.producer --video-dir data/cameras/cam_01 --camera-id cam-01 --loop --realtime

python -m safestream.producer --video-dir data/cameras/cam_02 --camera-id cam-02 --loop --realtime
```

Then open <http://localhost:8000> and watch per-camera `total_safe` / `total_unsafe`
and the rolling unsafe ratio update in real time.

### Running from VS Code

The `.vscode/launch.json` file ships four debug configurations — Producer, Detector,
Aggregator, Dashboard — plus a compound configuration **"SafeStream: all services"**
that starts everything at once.

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

This produces a `*-cls` model at `runs/classify/runs/safestream_yolov8m/weights/best.pt`,
which you pass to the detector via `--weights`. The detector auto-detects
classification weights and emits one labelled result per frame (no bounding box):

```bash
uv run python -m safestream.detector --weights runs/classify/runs/safestream_yolov8m/weights/best.pt
```

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
│   ├── detector/         # cctv-frames → YOLOv8 → safety-detections
│   ├── aggregator/       # safety-detections → safety-alerts + dashboard state
│   └── dashboard/        # FastAPI + WebSocket front-end
├── scripts/
│   ├── create_topics.py
│   └── train_yolo.py
└── tests/
    └── test_aggregator.py
```

## Configuration cheat sheet

All knobs are read from `.env` (see `.env.example` for the full list):

| Variable                             | Default             | Notes                                                   |
| ------------------------------------ | ------------------- | ------------------------------------------------------- |
| `KAFKA_BOOTSTRAP_SERVERS`            | —                   | Confluent Cloud bootstrap endpoint                      |
| `KAFKA_API_KEY` / `KAFKA_API_SECRET` | —                   | SASL credentials                                        |
| `USE_LOCAL_BROKER`                   | `false`             | If true, ignore SASL and use plaintext localhost broker |
| `TOPIC_FRAMES`                       | `cctv-frames`       | Producer → Detector                                     |
| `TOPIC_DETECTIONS`                   | `safety-detections` | Detector → Aggregator                                   |
| `TOPIC_ALERTS`                       | `safety-alerts`     | Aggregator → downstream sinks                           |
| `AGG_WINDOW_SECONDS`                 | `60`                | Rolling-window length                                   |
| `AGG_UNSAFE_RATIO_ALERT`             | `0.30`              | Ratio that triggers a WARN alert                        |
| `AGG_MIN_WINDOW_OBS`                 | `5`                 | Minimum rolling-window obs before alerting              |
| `DETECTOR_DEVICE`                    | `auto`              | `auto`, `mps`, `cuda`, or `cpu`                         |
| `DETECTOR_CONF`                      | `0.25`              | YOLOv8 confidence threshold                             |
| `DASHBOARD_HOST`                     | `127.0.0.1`         | FastAPI bind host                                       |
| `DASHBOARD_PORT`                     | `8000`              | FastAPI bind port                                       |

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
