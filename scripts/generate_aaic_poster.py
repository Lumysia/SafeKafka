"""Generate an AAIC-style poster image for the SafeStream-Kafka project."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from textwrap import wrap

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - import guard for CLI use
    raise SystemExit(
        "Pillow is required. Install project dependencies with `pip install -r requirements.txt`."
    ) from exc


CANVAS_SIZE = 4200
MARGIN = 170
GUTTER = 80
HEADER_HEIGHT = 520
SECTION_BAR_HEIGHT = 74

CREAM = "#F6EFE7"
PANEL = "#FFF9F1"
INK = "#341002"
MUTED = "#6F5A4E"
TERRACOTTA = "#C28056"
LIGHT_TERRACOTTA = "#E6B595"
DARK_TERRACOTTA = "#8A594E"
SAGE = "#7E9B83"
RED = "#B64A42"
GOLD = "#D7A74F"
LINE = "#D8C8B8"


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf",
                "/Library/Fonts/Arial Bold.ttf",
            ]
        )
    candidates.extend(
        [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttf",
            "/Library/Fonts/Arial.ttf",
        ]
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


FONT_TITLE = load_font(104, bold=True)
FONT_SUBTITLE = load_font(45)
FONT_SECTION = load_font(38, bold=True)
FONT_HEADING = load_font(32, bold=True)
FONT_BODY = load_font(27)
FONT_BODY_BOLD = load_font(27, bold=True)
FONT_SMALL = load_font(23)
FONT_SMALL_BOLD = load_font(23, bold=True)
FONT_STAT = load_font(66, bold=True)
FONT_TINY = load_font(19)


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def wrapped_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            lines.append("")
            continue
        estimate = max(12, int(max_width / max(text_size(draw, "M", font)[0], 1)))
        current = ""
        for word in paragraph.split():
            trial = word if not current else f"{current} {word}"
            if text_size(draw, trial, font)[0] <= max_width:
                current = trial
            else:
                if current:
                    lines.append(current)
                if text_size(draw, word, font)[0] <= max_width:
                    current = word
                else:
                    lines.extend(wrap(word, width=estimate))
                    current = ""
        if current:
            lines.append(current)
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    font: ImageFont.ImageFont,
    fill: str,
    max_width: int,
    line_gap: int = 9,
) -> int:
    x, y = xy
    for line in wrapped_lines(draw, text, font, max_width):
        if line:
            draw.text((x, y), line, font=font, fill=fill)
        y += text_size(draw, "Ag", font)[1] + line_gap
    return y


def draw_panel(draw: ImageDraw.ImageDraw, box: Box, title: str) -> Box:
    draw.rounded_rectangle(
        (box.x, box.y, box.x + box.w, box.y + box.h),
        radius=34,
        fill=PANEL,
        outline=LINE,
        width=3,
    )
    draw.rounded_rectangle(
        (box.x, box.y, box.x + box.w, box.y + SECTION_BAR_HEIGHT),
        radius=34,
        fill=TERRACOTTA,
    )
    draw.rectangle(
        (box.x, box.y + SECTION_BAR_HEIGHT - 34, box.x + box.w, box.y + SECTION_BAR_HEIGHT),
        fill=TERRACOTTA,
    )
    draw.text((box.x + 32, box.y + 16), title, font=FONT_SECTION, fill="white")
    return Box(box.x + 34, box.y + SECTION_BAR_HEIGHT + 30, box.w - 68, box.h - SECTION_BAR_HEIGHT - 60)


def bullet_list(
    draw: ImageDraw.ImageDraw,
    items: list[str],
    box: Box,
    font: ImageFont.ImageFont = FONT_SMALL,
    gap: int = 14,
) -> int:
    y = box.y
    bullet_width = 34
    for item in items:
        draw.ellipse((box.x + 4, y + 13, box.x + 18, y + 27), fill=TERRACOTTA)
        y = draw_wrapped(draw, item, (box.x + bullet_width, y), font, INK, box.w - bullet_width, 8)
        y += gap
    return y


def metric_card(draw: ImageDraw.ImageDraw, box: Box, value: str, label: str, color: str) -> None:
    draw.rounded_rectangle(
        (box.x, box.y, box.x + box.w, box.y + box.h),
        radius=26,
        fill="#FFFFFF",
        outline=LINE,
        width=2,
    )
    draw.text((box.x + 22, box.y + 22), value, font=FONT_STAT, fill=color)
    draw_wrapped(draw, label, (box.x + 24, box.y + 108), FONT_SMALL_BOLD, INK, box.w - 48, 5)


def compact_metric(draw: ImageDraw.ImageDraw, box: Box, value: str, label: str, color: str) -> None:
    draw.rounded_rectangle(
        (box.x, box.y, box.x + box.w, box.y + box.h),
        radius=22,
        fill="#FFFFFF",
        outline=LINE,
        width=2,
    )
    draw.text((box.x + 18, box.y + 16), value, font=FONT_HEADING, fill=color)
    draw_wrapped(draw, label, (box.x + 20, box.y + 62), FONT_SMALL, INK, box.w - 40, 4)


def draw_pipeline(draw: ImageDraw.ImageDraw, box: Box) -> None:
    labels = ["Video\nsources", "Producer", "Kafka\nframes", "YOLOv8\ndetector", "Aggregator", "Dashboard\n& alerts"]
    gap = 30
    w = (box.w - gap * (len(labels) - 1)) // len(labels)
    x = box.x + 6
    y = box.y + 78
    for index, label in enumerate(labels):
        fill = LIGHT_TERRACOTTA if index in {0, 2} else "#FFFFFF"
        outline = DARK_TERRACOTTA if index in {2, 4} else TERRACOTTA
        draw.rounded_rectangle((x, y, x + w, y + 150), radius=24, fill=fill, outline=outline, width=4)
        lines = label.split("\n")
        text_height = len(lines) * 34
        line_y = y + (150 - text_height) // 2
        for line in lines:
            text_width, _ = text_size(draw, line, FONT_SMALL_BOLD)
            draw.text((x + (w - text_width) // 2, line_y), line, font=FONT_SMALL_BOLD, fill=INK)
            line_y += 36
        if index < len(labels) - 1:
            arrow_x = x + w + 5
            draw.line((x + w, y + 75, x + w + gap - 5, y + 75), fill=INK, width=5)
            draw.polygon(
                [(arrow_x + gap - 7, y + 75), (arrow_x + 9, y + 60), (arrow_x + 9, y + 90)],
                fill=INK,
            )
        x += w + gap


def draw_model_chart(draw: ImageDraw.ImageDraw, box: Box) -> None:
    models = [
        ("YOLOv8", 156, RED),
        ("R18+GRU", 73, TERRACOTTA),
        ("R(2+1)D", 38, DARK_TERRACOTTA),
        ("Swin-T", 25, SAGE),
        ("Hiera-B", 21, SAGE),
        ("MViTv2", 21, SAGE),
    ]
    max_fps = 160
    x0 = box.x + 80
    y0 = box.y + box.h - 80
    chart_w = box.w - 160
    chart_h = box.h - 150
    draw.line((x0, y0, x0 + chart_w, y0), fill=MUTED, width=3)
    draw.line((x0, y0, x0, y0 - chart_h), fill=MUTED, width=3)
    bar_w = chart_w // len(models) - 28
    for i, (name, fps, color) in enumerate(models):
        left = x0 + i * (chart_w // len(models)) + 18
        height = int(chart_h * fps / max_fps)
        top = y0 - height
        draw.rounded_rectangle((left, top, left + bar_w, y0), radius=16, fill=color)
        value = f"{fps}"
        value_w, _ = text_size(draw, value, FONT_SMALL_BOLD)
        draw.text((left + (bar_w - value_w) // 2, top - 34), value, font=FONT_SMALL_BOLD, fill=INK)
        name_w, _ = text_size(draw, name, FONT_TINY)
        draw.text((left + (bar_w - name_w) // 2, y0 + 16), name, font=FONT_TINY, fill=INK)
    draw.text((x0, box.y + 4), "Single-stream ceiling (frames per second)", font=FONT_SMALL_BOLD, fill=INK)


def draw_detection_system(draw: ImageDraw.ImageDraw, box: Box) -> None:
    stages = [
        ("Frame", "JPEG decode"),
        ("YOLOv8", "class + confidence"),
        ("Mapper", "safe / unsafe / other"),
        ("Window", "60 s ratio + EWMA"),
        ("Alert", "WARN / HIGH"),
    ]
    gap = 18
    card_w = (box.w - gap * (len(stages) - 1)) // len(stages)
    y = box.y + 22
    x = box.x
    for index, (title, detail) in enumerate(stages):
        fill = LIGHT_TERRACOTTA if index in {1, 3} else "#FFFFFF"
        draw.rounded_rectangle((x, y, x + card_w, y + 132), radius=22, fill=fill, outline=TERRACOTTA, width=3)
        title_w, _ = text_size(draw, title, FONT_SMALL_BOLD)
        draw.text((x + (card_w - title_w) // 2, y + 30), title, font=FONT_SMALL_BOLD, fill=INK)
        detail_w, _ = text_size(draw, detail, FONT_TINY)
        draw.text((x + (card_w - detail_w) // 2, y + 76), detail, font=FONT_TINY, fill=MUTED)
        if index < len(stages) - 1:
            draw.line((x + card_w, y + 66, x + card_w + gap, y + 66), fill=INK, width=4)
            draw.polygon(
                [(x + card_w + gap - 2, y + 66), (x + card_w + 6, y + 54), (x + card_w + 6, y + 78)],
                fill=INK,
            )
        x += card_w + gap
    draw_wrapped(
        draw,
        "The aggregator is the alert owner: it smooths frame-level predictions, tracks each camera independently, and emits one event for sustained unsafe behavior instead of one alert per unsafe frame.",
        (box.x, y + 172),
        FONT_SMALL,
        INK,
        box.w,
        7,
    )


def draw_header(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle((0, 0, CANVAS_SIZE, HEADER_HEIGHT), fill=INK)
    draw.rectangle((0, HEADER_HEIGHT - 22, CANVAS_SIZE, HEADER_HEIGHT), fill=TERRACOTTA)
    title = "SafeStream-Kafka"
    subtitle = "Real-Time Safety Analytics for IoT Workplace Video Streams"
    authors = "Vitor Brandao Raposo, Yujie Lin, Livia Zhang, Quang Thong Phung, Ahmed Seyam"
    affiliation = "Faculty of Engineering and Applied Science, Ontario Tech University"
    draw.text((MARGIN, 92), title, font=FONT_TITLE, fill="white")
    draw.text((MARGIN, 232), subtitle, font=FONT_SUBTITLE, fill=LIGHT_TERRACOTTA)
    draw.text((MARGIN, 330), authors, font=FONT_HEADING, fill="white")
    draw.text((MARGIN, 386), affiliation, font=FONT_SMALL, fill="#EBDCCE")

    tag_x = CANVAS_SIZE - MARGIN - 870
    draw.rounded_rectangle((tag_x, 120, CANVAS_SIZE - MARGIN, 372), radius=36, fill="#4B2C1F")
    draw.text((tag_x + 42, 158), "Kafka + YOLOv8 + FastAPI", font=FONT_HEADING, fill=LIGHT_TERRACOTTA)
    draw.text((tag_x + 42, 220), "Live safety state from camera streams", font=FONT_BODY_BOLD, fill="white")
    draw.text((tag_x + 42, 278), "Frame events, detections, alerts, dashboard", font=FONT_SMALL, fill="#EBDCCE")


def generate(output: Path) -> None:
    image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), CREAM)
    draw = ImageDraw.Draw(image)
    draw_header(draw)

    col_w = (CANVAS_SIZE - 2 * MARGIN - 2 * GUTTER) // 3
    top = HEADER_HEIGHT + 80
    col_x = [MARGIN, MARGIN + col_w + GUTTER, MARGIN + 2 * (col_w + GUTTER)]

    left_x = col_x[0]
    mid_x = col_x[1]
    right_x = col_x[2]
    row1_y = top
    row2_y = top + 650
    row3_y = top + 1260
    bottom_y = top + 1900

    problem = draw_panel(draw, Box(left_x, row1_y, col_w, 590), "Background and Problem Definition")
    draw_wrapped(
        draw,
        "Workplace cameras already capture safety-relevant activity, but continuous manual monitoring does not scale. SafeStream-Kafka converts camera frames into per-camera safety counts, rolling unsafe ratios, and alert events.",
        (problem.x, problem.y),
        FONT_BODY,
        INK,
        problem.w,
    )
    bullet_list(
        draw,
        [
            "Accepts video files, RTSP streams, webcams, and replayed CCTV clips.",
            "Classifies each sampled frame as safe, unsafe, or other.",
            "Aggregates frame labels before raising an alert.",
        ],
        Box(problem.x, problem.y + 260, problem.w, 250),
        FONT_SMALL,
    )

    gaps = draw_panel(draw, Box(left_x, row2_y, col_w, 540), "Research Gaps")
    bullet_list(
        draw,
        [
            "Offline test accuracy does not show how a model behaves in a live alert stream.",
            "Frame-level predictions can flicker and create repeated false alerts.",
            "Alert thresholds need model-specific calibration.",
            "Few safety-vision prototypes define explicit streaming topic contracts.",
        ],
        gaps,
    )

    contributions = draw_panel(draw, Box(left_x, row3_y, col_w, 570), "Key Contributions")
    bullet_list(
        draw,
        [
            "Kafka topics separate frame ingestion, inference, aggregation, and visualization.",
            "Per-camera state keeps counts, rolling windows, and alert levels independent.",
            "The dashboard reports current frames, recent alerts, latency, throughput, and unsafe-ratio trends.",
            "The model comparison links accuracy, false alerts, and per-frame cost.",
        ],
        contributions,
        FONT_SMALL,
        9,
    )

    data = draw_panel(draw, Box(mid_x, row1_y, col_w, 590), "Data Collection")
    draw.text((data.x, data.y), "Inclusion Criteria", font=FONT_HEADING, fill=INK)
    bullet_list(
        draw,
        [
            "Readable CCTV files, clip directories, RTSP streams, webcams, or replayed video sources.",
            "Labels must map to safe, unsafe, or other categories.",
            "Voxel51 workplace-safety clips: 691 clips, 8 classes, typical 1080p footage.",
        ],
        Box(data.x, data.y + 56, data.w, 260),
        FONT_SMALL,
        8,
    )
    draw.text((data.x, data.y + 350), "Collection Process", font=FONT_HEADING, fill=INK)
    draw_wrapped(
        draw,
        "The producer samples frames at a configured analytics FPS, JPEG-encodes each frame, and publishes camera_id, frame_id, source, timestamp, and image payload.",
        (data.x, data.y + 405),
        FONT_SMALL,
        INK,
        data.w,
    )

    detection = draw_panel(draw, Box(mid_x, row2_y, col_w, 540), "Detection System")
    draw.text((detection.x, detection.y), "Safety Alert Detection", font=FONT_HEADING, fill=INK)
    detection = Box(detection.x, detection.y + 36, detection.w, detection.h - 36)
    draw_detection_system(draw, detection)

    results = draw_panel(draw, Box(right_x, row1_y, col_w, 590), "Results")
    card_w = (results.w - 36) // 2
    metric_card(draw, Box(results.x, results.y, card_w, 170), "0.887", "unsafe average precision", RED)
    metric_card(draw, Box(results.x + card_w + 36, results.y, card_w, 170), "0.641", "8-class accuracy", TERRACOTTA)
    metric_card(draw, Box(results.x, results.y + 200, card_w, 170), "0.198s", "mean end-to-end latency", SAGE)
    metric_card(draw, Box(results.x + card_w + 36, results.y + 200, card_w, 170), "0 lag", "at 4 fps controlled run", GOLD)
    bullet_list(
        draw,
        [
            "YOLOv8 had the lowest safe-clip alert rate: 0.311.",
            "Base64 JSON frame payloads averaged about 352 KB.",
        ],
        Box(results.x, results.y + 420, results.w, 90),
        FONT_SMALL,
        6,
    )

    findings = draw_panel(draw, Box(right_x, row2_y, col_w, 540), "Feature Analysis")
    bullet_list(
        draw,
        [
            "Single-frame YOLOv8 performed best on behaviors visible in one frame.",
            "Temporal models increased inference cost without improving accuracy in this setup.",
            "Offline metrics and streaming false-alert behavior must be measured separately.",
            "Per-model thresholds reduced unsafe-probability scale mismatch.",
        ],
        findings,
        FONT_SMALL,
        10,
    )

    chart = draw_panel(draw, Box(right_x, row3_y, col_w, 570), "Model Throughput")
    draw_model_chart(draw, chart)

    arch = draw_panel(draw, Box(MARGIN, bottom_y, col_w * 2 + GUTTER, 810), "System Architecture")
    draw.text((arch.x, arch.y), "Deployment Setting", font=FONT_HEADING, fill=INK)
    draw.text(
        (arch.x + 360, arch.y + 4),
        "CCTV, RTSP, webcam, and replayed video streams routed through Kafka services",
        font=FONT_SMALL,
        fill=MUTED,
    )
    arch = Box(arch.x, arch.y + 48, arch.w, arch.h - 48)
    draw_pipeline(draw, Box(arch.x, arch.y + 15, arch.w, 220))
    draw.text((arch.x, arch.y + 270), "Kafka Topics", font=FONT_HEADING, fill=INK)
    topic_w = (arch.w - 60) // 3
    for index, (topic, detail, color) in enumerate(
        [
            ("cctv-frames", "encoded frame events", LIGHT_TERRACOTTA),
            ("safety-detections", "model output summaries", "#FFFFFF"),
            ("safety-alerts", "aggregated alert events", LIGHT_TERRACOTTA),
        ]
    ):
        x = arch.x + index * (topic_w + 30)
        y = arch.y + 325
        draw.rounded_rectangle((x, y, x + topic_w, y + 135), radius=24, fill=color, outline=TERRACOTTA, width=3)
        draw.text((x + 24, y + 26), topic, font=FONT_BODY_BOLD, fill=INK)
        draw.text((x + 24, y + 78), detail, font=FONT_SMALL, fill=MUTED)
    draw_wrapped(
        draw,
        "Solid arrows show the online path: producer -> detector -> aggregator -> dashboard. The standalone aggregator owns safety-alerts; the dashboard reads alerts and frames without re-running aggregation.",
        (arch.x, arch.y + 500),
        FONT_SMALL,
        INK,
        arch.w,
        10,
    )

    dataset = draw_panel(draw, Box(right_x, bottom_y, col_w, 810), "Dataset Overview")
    metric_w = (dataset.w - 48) // 3
    metric_h = 112
    dataset_metrics = [
        ("691", "CCTV clips", TERRACOTTA),
        ("8", "behavior classes", DARK_TERRACOTTA),
        ("12", "frames per clip", SAGE),
        ("1080p", "typical resolution", TERRACOTTA),
        ("5 wks", "capture period", DARK_TERRACOTTA),
        ("1,500", "test frames", SAGE),
    ]
    for index, (value, label, color) in enumerate(dataset_metrics):
        row = index // 3
        col = index % 3
        x = dataset.x + col * (metric_w + 24)
        y = dataset.y + row * (metric_h + 20)
        compact_metric(draw, Box(x, y, metric_w, metric_h), value, label, color)

    draw.text((dataset.x, dataset.y + 285), "Prepared Split", font=FONT_HEADING, fill=INK)
    split_x = dataset.x
    split_y = dataset.y + 338
    split_w = dataset.w
    bar_h = 42
    train_w = int(split_w * 0.70)
    val_w = int(split_w * 0.20)
    draw.rounded_rectangle((split_x, split_y, split_x + split_w, split_y + bar_h), radius=18, fill="#FFFFFF", outline=LINE, width=2)
    draw.rounded_rectangle((split_x, split_y, split_x + train_w, split_y + bar_h), radius=18, fill=RED)
    draw.rectangle((split_x + train_w - 18, split_y, split_x + train_w, split_y + bar_h), fill=RED)
    draw.rectangle((split_x + train_w, split_y, split_x + train_w + val_w, split_y + bar_h), fill=TERRACOTTA)
    draw.rounded_rectangle((split_x + train_w + val_w, split_y, split_x + split_w, split_y + bar_h), radius=18, fill=SAGE)
    draw.rectangle((split_x + train_w + val_w, split_y, split_x + train_w + val_w + 18, split_y + bar_h), fill=SAGE)
    draw.text((split_x + 18, split_y + 8), "train 70%", font=FONT_TINY, fill="white")
    draw.text((split_x + train_w + 18, split_y + 8), "val 20%", font=FONT_TINY, fill="white")
    draw.text((split_x + train_w + val_w + 18, split_y + 8), "test 10%", font=FONT_TINY, fill="white")

    draw.text((dataset.x, dataset.y + 425), "Training Data", font=FONT_HEADING, fill=INK)
    bullet_list(
        draw,
        [
            "Source: Voxel51 Safe and Unsafe Behaviours.",
            "Annotations: clip-level behavior labels; no bounding boxes.",
            "YOLO input: 224 x 224 images, batch size 16, 50 epochs.",
            "Test split: 768 of 1,500 sampled frames are unsafe.",
            "Location: Eskisehir, TR production facility.",
        ],
        Box(dataset.x, dataset.y + 480, dataset.w, 260),
        FONT_SMALL,
        0,
    )

    conclusion = draw_panel(draw, Box(MARGIN, bottom_y + 870, CANVAS_SIZE - 2 * MARGIN, 330), "Project Summary")
    draw_wrapped(
        draw,
        "SafeStream-Kafka separates ingestion, inference, aggregation, and visualization across Kafka-backed services. YOLOv8 labels sampled frames, and the aggregator converts frame labels into per-camera counts, rolling ratios, and alert state.",
        (conclusion.x, conclusion.y),
        FONT_SMALL,
        INK,
        conclusion.w,
        10,
    )
    bullet_list(
        draw,
        [
            "Working prototype: producer, detector, aggregator, dashboard, local Kafka setup, and training scripts.",
            "Next work: finalize single-owner alerting, profile multi-camera load, and add a public smoke-test stream.",
        ],
        Box(conclusion.x, conclusion.y + 155, conclusion.w, 150),
        FONT_SMALL,
        0,
    )

    footer_y = CANVAS_SIZE - 180
    draw.rounded_rectangle((MARGIN, footer_y, CANVAS_SIZE - MARGIN, footer_y + 96), radius=28, fill=INK)
    footer = "Code: github.com/Lumysia/SafeKafka   |   Course: ENGR 5785G Real-Time Data Analytics for IoT   |   Sensor Syndicate"
    draw.text((MARGIN + 46, footer_y + 32), footer, font=FONT_SMALL_BOLD, fill="white")

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/safestream_aaic_poster.png"),
        help="Path for the generated poster image.",
    )
    args = parser.parse_args()
    generate(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
