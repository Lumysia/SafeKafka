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
MARGIN = 50
HEADER_HEIGHT = 430
CENTER_GAP = 70
SECTION_BAR_HEIGHT = 92

NAVY = "#003C71"
NAVY_DARK = "#00283C"
ORANGE = "#C84D12"
ORANGE_LIGHT = "#E75D2A"
TEAL = "#155F7B"
TEAL_LIGHT = "#2C7CA3"
BLUE_GRAY = "#97A8B6"
PANEL = "#FFFFFF"
INK = "#111111"
MUTED = "#334A5B"
TABLE_A = "#F8D9D2"
TABLE_B = "#FCEBE8"
CALLOUT = "#BBD8EF"
LINE = "#1E5D83"
LIGHT_LINE = "#8DB9D4"
SCRIPT_DIR = Path(__file__).resolve().parent
ONTARIO_TECH_LOGO = SCRIPT_DIR / "assets" / "ontario_tech_logo.png"
ARCHITECTURE_DIAGRAM = SCRIPT_DIR / "assets" / "architecture.png"
STREAMING_EVAL_DIAGRAM = SCRIPT_DIR / "assets" / "streaming_evaluation.png"


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if bold:
        candidates = [
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/segoeuib.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
        ]
    else:
        candidates = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
        ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


FONT_TITLE = load_font(84)
FONT_TITLE_BOLD = load_font(84, bold=True)
FONT_SUBTITLE = load_font(38)
FONT_AUTHORS = load_font(30)
FONT_SECTION = load_font(36, bold=True)
FONT_HEADING = load_font(30, bold=True)
FONT_BODY = load_font(27)
FONT_BODY_BOLD = load_font(27, bold=True)
FONT_SMALL = load_font(24)
FONT_SMALL_BOLD = load_font(24, bold=True)
FONT_TINY = load_font(18)
FONT_TINY_BOLD = load_font(18, bold=True)


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def draw_centered(
    draw: ImageDraw.ImageDraw,
    box: Box,
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    text_w, text_h = text_size(draw, text, font)
    draw.text((box.x + (box.w - text_w) // 2, box.y + (box.h - text_h) // 2 - 3), text, font=font, fill=fill)


def paste_image_fit(canvas: Image.Image, path: Path, box: Box, pad: int = 0) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing image asset: {path}")
    image = Image.open(path).convert("RGBA")
    max_w = box.w - pad * 2
    max_h = box.h - pad * 2
    scale = min(max_w / image.width, max_h / image.height)
    resized = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
    x = box.x + (box.w - resized.width) // 2
    y = box.y + (box.h - resized.height) // 2
    canvas.alpha_composite(resized, (x, y))


def wrapped_lines(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
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
                continue
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
    line_gap: int = 7,
) -> int:
    x, y = xy
    line_height = text_size(draw, "Ag", font)[1] + line_gap
    for line in wrapped_lines(draw, text, font, max_width):
        if line:
            draw.text((x, y), line, font=font, fill=fill)
        y += line_height
    return y


def draw_wrapped_centered(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: Box,
    font: ImageFont.ImageFont,
    fill: str,
    line_gap: int = 6,
) -> None:
    lines = wrapped_lines(draw, text, font, box.w)
    line_height = text_size(draw, "Ag", font)[1] + line_gap
    y = box.y + (box.h - len(lines) * line_height) // 2
    for line in lines:
        if line:
            line_w, _ = text_size(draw, line, font)
            draw.text((box.x + (box.w - line_w) // 2, y), line, font=font, fill=fill)
        y += line_height


def draw_bar(draw: ImageDraw.ImageDraw, box: Box, title: str, radius: int = 15) -> None:
    draw.rounded_rectangle((box.x, box.y, box.x + box.w, box.y + box.h), radius=radius, fill=ORANGE, outline=INK, width=2)
    draw_centered(draw, box, title, FONT_SECTION, "white")


def draw_panel(draw: ImageDraw.ImageDraw, box: Box, title: str) -> Box:
    draw.rectangle((box.x, box.y, box.x + box.w, box.y + box.h), fill=PANEL, outline=ORANGE_LIGHT, width=3)
    draw_bar(draw, Box(box.x, box.y, box.w, SECTION_BAR_HEIGHT), title, radius=0)
    return Box(box.x + 42, box.y + SECTION_BAR_HEIGHT + 34, box.w - 84, box.h - SECTION_BAR_HEIGHT - 52)


def bullet_list(
    draw: ImageDraw.ImageDraw,
    items: list[str],
    box: Box,
    font: ImageFont.ImageFont = FONT_SMALL,
    gap: int = 11,
    bullet_fill: str = INK,
) -> int:
    y = box.y
    bullet_width = 34
    for item in items:
        draw.ellipse((box.x + 2, y + 11, box.x + 14, y + 23), fill=bullet_fill)
        y = draw_wrapped(draw, item, (box.x + bullet_width, y), font, INK, box.w - bullet_width, 5)
        y += gap
    return y


def official_logo_image(target_width: int) -> Image.Image:
    if not ONTARIO_TECH_LOGO.exists():
        raise FileNotFoundError(f"Missing logo asset: {ONTARIO_TECH_LOGO}")
    logo = Image.open(ONTARIO_TECH_LOGO).convert("RGBA")
    scale = target_width / logo.width
    target_height = int(logo.height * scale)
    return logo.resize((target_width, target_height), Image.Resampling.LANCZOS)


def draw_header(image: Image.Image, draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle((0, 0, CANVAS_SIZE, HEADER_HEIGHT), fill=NAVY)
    logo = official_logo_image(680)
    image.alpha_composite(logo, (62, 84))

    title_x = 970
    title_y = 70
    draw.text((title_x, title_y), "SafeStream-Kafka: Real-Time Safety Analytics", font=FONT_TITLE, fill="white")
    draw.text((title_x, title_y + 122), "for IoT Workplace Video Streams", font=FONT_TITLE, fill="white")
    draw.text(
        (title_x + 8, title_y + 228),
        "Vitor Brandao Raposo, Yujie Lin, Livia Zhang, Quang Thong Phung, Ahmed Seyam",
        font=FONT_AUTHORS,
        fill="white",
    )
    draw.text(
        (title_x + 8, title_y + 274),
        "Faculty of Engineering and Applied Science, Ontario Tech University",
        font=FONT_TINY_BOLD,
        fill="white",
    )

    badge_x = CANVAS_SIZE - 545
    draw.rounded_rectangle((badge_x, 70, CANVAS_SIZE - 80, 315), radius=42, fill=NAVY_DARK)
    draw.text((badge_x + 54, 112), "Kafka", font=load_font(54, bold=True), fill=ORANGE_LIGHT)
    draw.text((badge_x + 54, 176), "YOLOv8", font=load_font(54, bold=True), fill="white")
    draw.text((badge_x + 54, 240), "FastAPI", font=load_font(54, bold=True), fill=CALLOUT)


def draw_gap_cards(draw: ImageDraw.ImageDraw, box: Box) -> None:
    titles = [
        "Hospital Setting\nApplication",
        "Privacy-Preserving\nMonitoring",
        "Integrated\nMultimodal Fusion",
        "Limited Real-\nWorld Validation",
    ]
    bodies = [
        "Limited real-time system that detects unsafe behavior in hospitals or workplaces using streaming fusion",
        "Video methods raise ethical and privacy concerns without careful aggregation and display controls",
        "Existing systems usually evaluate video or wearable data separately rather than one live pipeline",
        "Public benchmarks rarely report streaming false-alert behavior under deployment constraints",
    ]
    impacts = ["deployment gap", "privacy risk", "fusion gap", "validation gap"]
    gap = 56
    card_w = (box.w - gap * 3) // 4
    for index, (title, body) in enumerate(zip(titles, bodies)):
        x = box.x + index * (card_w + gap)
        draw.rectangle((x, box.y, x + card_w, box.y + 155), fill=TEAL)
        ty = box.y + 34
        for line in title.split("\n"):
            line_w, _ = text_size(draw, line, FONT_SMALL_BOLD)
            draw.text((x + (card_w - line_w) // 2, ty), line, font=FONT_SMALL_BOLD, fill="white")
            ty += 35
        draw.rectangle((x, box.y + 155, x + card_w, box.y + box.h), fill="#D4DAE0")
        draw_wrapped(draw, body, (x + 28, box.y + 178), FONT_SMALL, INK, card_w - 56, 4)
        badge_y = box.y + box.h - 82
        draw.rounded_rectangle((x + 28, badge_y, x + card_w - 28, badge_y + 48), radius=8, fill="#EEF4F7", outline=LIGHT_LINE, width=2)
        draw_centered(draw, Box(x + 28, badge_y, card_w - 56, 48), impacts[index].upper(), FONT_TINY_BOLD, TEAL)


def draw_background_flow(draw: ImageDraw.ImageDraw, box: Box) -> None:
    cards = [
        ("Capture", "RTSP / webcam / replayed CCTV"),
        ("Classify", "YOLOv8 safe vs. unsafe labels"),
        ("Alert", "Rolling camera state + EWMA"),
    ]
    gap = 26
    card_w = (box.w - gap * 2) // 3
    for index, (title, detail) in enumerate(cards):
        x = box.x + index * (card_w + gap)
        draw.rounded_rectangle((x, box.y, x + card_w, box.y + box.h), radius=18, fill="#F4F7F9", outline=LIGHT_LINE, width=2)
        draw.ellipse((x + 24, box.y + 22, x + 78, box.y + 76), fill=TEAL)
        draw_centered(draw, Box(x + 24, box.y + 22, 54, 54), str(index + 1), FONT_SMALL_BOLD, "white")
        draw.text((x + 96, box.y + 22), title, font=FONT_SMALL_BOLD, fill=TEAL)
        draw_wrapped(draw, detail, (x + 96, box.y + 62), FONT_TINY_BOLD, INK, card_w - 122, 3)
        if index < 2:
            arrow_x = x + card_w
            arrow_y = box.y + box.h // 2
            draw.line((arrow_x, arrow_y, arrow_x + gap, arrow_y), fill=LINE, width=3)
            draw.polygon([(arrow_x + gap, arrow_y), (arrow_x + gap - 16, arrow_y - 10), (arrow_x + gap - 16, arrow_y + 10)], fill=LINE)


def draw_stream_contract_card(draw: ImageDraw.ImageDraw, box: Box) -> None:
    draw.rounded_rectangle((box.x, box.y, box.x + box.w, box.y + box.h), radius=24, fill="#F4F7F9", outline=LIGHT_LINE, width=2)
    draw.text((box.x + 28, box.y + 24), "Streaming Contract", font=FONT_SMALL_BOLD, fill=TEAL)
    rows = [
        ("cctv-frames", "camera_id, frame_id, image_b64"),
        ("safety-detections", "safe_count, unsafe_count"),
        ("safety-alerts", "level, unsafe_ratio, timestamp"),
    ]
    y = box.y + 78
    for topic, payload in rows:
        draw.rounded_rectangle((box.x + 28, y, box.x + box.w - 28, y + 52), radius=8, fill="white", outline="#C7DDEB", width=1)
        draw.text((box.x + 46, y + 13), topic, font=FONT_TINY_BOLD, fill=TEAL)
        draw.text((box.x + 240, y + 13), payload, font=FONT_TINY, fill=INK)
        y += 64


def draw_contributions(draw: ImageDraw.ImageDraw, box: Box) -> None:
    items = [
        (
            "Multimodal\nDetection\nFramework",
            ["Kafka separates camera ingestion, inference, aggregation, and dashboard rendering.", "Each camera keeps ordered state through camera_id keyed topics.", "Frame events become rolling safety counts and alert levels."],
        ),
        (
            "Feature\nEngineering and\nSelection",
            ["Raw YOLO labels are mapped into safe, unsafe, or other classes.", "Unsafe evidence is smoothed before alert decisions are published.", "Model thresholds are calibrated instead of assumed globally."],
        ),
        (
            "Real-World\nClinical\nValidation",
            ["Replayable clips simulate continuous camera streams and realistic latency.", "Dashboard tracks frames, alerts, throughput, and unsafe-ratio trends.", "Streaming metrics are reported separately from offline accuracy."],
        ),
        (
            "High Accuracy\nwith Attention\nMechanism",
            ["YOLOv8 remains strongest for visible single-frame safety behavior.", "Temporal models are compared against alert quality and per-frame cost.", "The selected deployment path minimizes repeated false alerts."],
        ),
    ]
    gap = 110
    card_w = (box.w - gap * 3) // 4
    for index, (title, bullets) in enumerate(items):
        x = box.x + index * (card_w + gap)
        top_fill = TEAL if index == 0 else TEAL_LIGHT if index in {1, 2} else BLUE_GRAY
        header_h = 210
        body_y = box.y + header_h + 68
        body_h = box.h - header_h - 95
        draw.rounded_rectangle((x, box.y, x + card_w, box.y + header_h), radius=15, fill=top_fill)
        y = box.y + 28
        for line in title.split("\n"):
            line_w, _ = text_size(draw, line, FONT_SMALL_BOLD)
            draw.text((x + (card_w - line_w) // 2, y), line, font=FONT_SMALL_BOLD, fill="white")
            y += 34
        draw.line((x + card_w // 2, box.y + header_h, x + card_w // 2, body_y), fill=LINE, width=2)
        body_box = (x + 22, body_y, x + card_w - 22, body_y + body_h)
        draw.rounded_rectangle(body_box, radius=22, fill="white", outline=LINE, width=2)
        number = f"0{index + 1}"
        draw.rounded_rectangle((x + 58, body_y + 36, x + 138, body_y + 100), radius=8, fill="#EAF4FA", outline=LIGHT_LINE, width=2)
        draw_centered(draw, Box(x + 58, body_y + 36, 80, 64), number, FONT_SMALL_BOLD, TEAL)
        y_text = body_y + 135
        for bullet in bullets:
            draw.ellipse((x + 60, y_text + 12, x + 72, y_text + 24), fill=TEAL)
            y_text = draw_wrapped(draw, bullet, (x + 88, y_text), FONT_SMALL, INK, card_w - 146, 5)
            y_text += 18
        draw.line((x + 58, body_y + body_h - 70, x + card_w - 58, body_y + body_h - 70), fill=LIGHT_LINE, width=2)
        draw_centered(draw, Box(x + 58, body_y + body_h - 58, card_w - 116, 38), "STREAMING-FIRST", FONT_TINY_BOLD, TEAL)


def draw_architecture(canvas: Image.Image, draw: ImageDraw.ImageDraw, box: Box) -> None:
    draw.rounded_rectangle((box.x + 20, box.y + 70, box.x + box.w - 20, box.y + box.h - 70), radius=42, fill="white", outline=ORANGE_LIGHT, width=3)
    draw.rectangle((box.x + box.w // 2 - 220, box.y + 42, box.x + box.w // 2 + 220, box.y + 105), fill=ORANGE)
    draw_centered(draw, Box(box.x + box.w // 2 - 220, box.y + 42, 440, 63), "Mermaid System Flow", FONT_SMALL_BOLD, "white")
    diagram = Box(box.x + 70, box.y + 130, box.w - 140, 210)
    draw.rounded_rectangle((diagram.x, diagram.y, diagram.x + diagram.w, diagram.y + diagram.h), radius=18, fill="#FAFAFA", outline="#D6D6D6", width=2)
    paste_image_fit(canvas, ARCHITECTURE_DIAGRAM, diagram, 18)
    contract = Box(box.x + 130, box.y + 358, box.w - 260, 210)
    draw.text((contract.x, contract.y), "Kafka topic contract", font=FONT_HEADING, fill=TEAL)
    topic_rows = [
        ("cctv-frames", "raw camera frames"),
        ("safety-detections", "model counts + labels"),
        ("safety-alerts", "aggregated alert state"),
    ]
    row_y = contract.y + 58
    for topic, detail in topic_rows:
        draw.rounded_rectangle((contract.x, row_y, contract.x + contract.w, row_y + 48), radius=12, fill="#EFF8FC", outline=LIGHT_LINE, width=2)
        draw.text((contract.x + 28, row_y + 11), topic, font=FONT_SMALL_BOLD, fill=TEAL)
        draw.text((contract.x + 470, row_y + 13), detail, font=FONT_SMALL, fill=INK)
        row_y += 60


def draw_data_collection(draw: ImageDraw.ImageDraw, box: Box) -> None:
    left = Box(box.x, box.y + 20, 650, box.h - 40)
    right = Box(box.x + 700, box.y + 20, box.w - 700, box.h - 40)
    for sub in [left, right]:
        draw.rounded_rectangle((sub.x, sub.y, sub.x + sub.w, sub.y + sub.h), radius=80, fill="white", outline=INK, width=2)

    draw.text((left.x + 42, left.y + 45), "Inclusion Criteria:", font=FONT_HEADING, fill=ORANGE_LIGHT)
    bullet_list(
        draw,
        [
            "Video files, RTSP streams, webcams, and replayed CCTV clips.",
            "Labels map to safe, unsafe, or other categories.",
            "Voxel51 workplace-safety clips: 691 clips across 8 classes.",
            "Each message is keyed by camera_id for ordered per-camera processing.",
        ],
        Box(left.x + 45, left.y + 95, 510, 300),
        FONT_SMALL,
        6,
        NAVY,
    )
    table_x = left.x + 45
    table_y = left.y + 395
    headers = ["Source", "FPS", "Payload", "Topic"]
    rows = [
        ["CCTV", "4", "JPEG+b64", "cctv-frames"],
        ["RTSP", "cfg", "JSON", "cctv-frames"],
        ["YOLO", "live", "counts", "detections"],
        ["Agg", "1Hz", "alerts", "safety-alerts"],
    ]
    col_w = [130, 85, 155, 205]
    draw.rectangle((table_x, table_y, table_x + sum(col_w), table_y + 45), fill=TEAL)
    x = table_x
    for header, width in zip(headers, col_w):
        draw_centered(draw, Box(x, table_y, width, 45), header, FONT_TINY_BOLD, "white")
        x += width
    for ridx, row in enumerate(rows):
        y = table_y + 45 + ridx * 46
        draw.rectangle((table_x, y, table_x + sum(col_w), y + 46), fill="#DCE2E7" if ridx % 2 == 0 else "#C9D1D8")
        x = table_x
        for cell, width in zip(row, col_w):
            draw.text((x + 10, y + 12), cell, font=FONT_TINY, fill=INK)
            x += width

    draw.text((right.x + 50, right.y + 45), "Collection Process:", font=FONT_HEADING, fill=ORANGE_LIGHT)
    steps = [
        ("Video source", "sample frames"),
        ("Producer", "encode + publish"),
        ("Detector", "YOLO labels"),
        ("Aggregator", "smooth alerts"),
        ("Dashboard", "live view"),
    ]
    start_x = right.x + 70
    start_y = right.y + 160
    for index, (name, detail) in enumerate(steps):
        x = start_x + index * 250
        draw.rounded_rectangle((x, start_y, x + 175, start_y + 120), radius=18, fill="#F7FBFD", outline=LINE, width=3)
        draw_centered(draw, Box(x + 15, start_y + 24, 145, 32), name, FONT_TINY_BOLD, INK)
        draw_centered(draw, Box(x + 15, start_y + 68, 145, 32), detail, FONT_TINY, MUTED)
        if index < len(steps) - 1:
            draw.line((x + 175, start_y + 60, x + 250, start_y + 60), fill=LINE, width=3)
            draw.polygon([(x + 250, start_y + 60), (x + 232, start_y + 49), (x + 232, start_y + 71)], fill=LINE)
    notes = [
        "Study data enter the same service contract as webcam or RTSP frames.",
        "The detector publishes compact counts and class details for every frame.",
        "The dashboard displays frames, latency, throughput, trends, and alerts.",
    ]
    y = right.y + 345
    for note in notes:
        draw.rounded_rectangle((right.x + 95, y, right.x + right.w - 95, y + 98), radius=11, fill=TEAL)
        draw_wrapped(draw, note, (right.x + 125, y + 18), FONT_TINY_BOLD, "white", right.w - 250, 3)
        y += 122


def draw_results_table(draw: ImageDraw.ImageDraw, box: Box) -> None:
    x = box.x
    y = box.y
    columns = ["Input", "Accuracy", "AP/AUC", "Latency", "F1/Recall", "Alert Rate"]
    widths = [430, 325, 320, 320, 330, 330]
    groups = [
        ("Evaluation", [
            ["YOLOv8 classifier", "0.641", "0.887", "0.198 s", "0.76 / 0.91", "0.311"],
            ["R18+GRU temporal", "0.613", "0.842", "73 fps", "stream eval", "calibrated"],
            ["Hiera-B linear probe", "0.620", "0.861", "21 fps", "stream eval", "calibrated"],
        ]),
        ("Streaming", [
            ["Kafka payload", "JSON", "base64", "352 KB", "per frame", "camera-keyed"],
            ["Controlled run", "4 fps", "0 lag", "1 Hz ws", "live", "stable"],
            ["Alert owner", "Aggregator", "EWMA", "60 s", "WARN/HIGH", "deduped"],
        ]),
    ]
    total_w = sum(widths)
    for group, rows in groups:
        draw.rectangle((x, y, x + total_w, y + 64), fill=ORANGE_LIGHT)
        draw_centered(draw, Box(x, y, total_w, 64), group, FONT_SMALL_BOLD, "white")
        y += 64
        draw.rectangle((x, y, x + total_w, y + 62), fill=TABLE_A)
        col_x = x
        for col, width in zip(columns, widths):
            draw.text((col_x + 12, y + 17), col, font=FONT_SMALL_BOLD, fill=INK)
            col_x += width
        y += 62
        for ridx, row in enumerate(rows):
            draw.rectangle((x, y, x + total_w, y + 62), fill=TABLE_B if ridx % 2 == 0 else TABLE_A)
            col_x = x
            for cell, width in zip(row, widths):
                draw.text((col_x + 12, y + 18), cell, font=FONT_SMALL, fill=INK)
                col_x += width
            y += 62


def draw_feature_chart(canvas: Image.Image, draw: ImageDraw.ImageDraw, box: Box) -> None:
    draw.rounded_rectangle((box.x + 10, box.y + 5, box.x + box.w - 10, box.y + box.h - 15), radius=28, fill="#FFFFFF", outline="#E5E5E5", width=2)
    draw.text((box.x + 55, box.y + 32), "Streaming threshold calibration", font=FONT_SMALL_BOLD, fill=TEAL)
    draw.text((box.x + 1210, box.y + 32), "Model speed comparison", font=FONT_SMALL_BOLD, fill=TEAL)
    callout = Box(box.x + 40, box.y + 80, 1040, 335)
    draw.rounded_rectangle((callout.x, callout.y, callout.x + callout.w, callout.y + callout.h), radius=70, fill=CALLOUT, outline=INK, width=2)
    bullet_list(
        draw,
        [
            "Every model emits unsafe_prob on a different scale, so thresholds are calibrated on validation clips.",
            "YOLOv8 performs best when unsafe behavior is visible in one frame.",
            "Temporal models increase inference cost without improving this deployment metric.",
            "Offline AP and streaming false-alert rate must be reported separately.",
        ],
        Box(callout.x + 55, callout.y + 45, callout.w - 110, 230),
        FONT_SMALL,
        8,
        INK,
    )

    bar = Box(box.x + 1190, box.y + 100, 650, 315)
    draw.text((bar.x + 170, bar.y - 18), "Throughput by Model", font=FONT_TINY_BOLD, fill=MUTED)
    x0 = bar.x + 85
    y0 = bar.y + bar.h - 42
    draw.line((x0, y0, x0 + 500, y0), fill="#A6A6A6", width=2)
    draw.line((x0, y0, x0, y0 - 265), fill="#A6A6A6", width=2)
    models = [("YOLO", 156, "#78C7E7"), ("R18", 73, "#3B83B7"), ("R2D", 38, "#78C7E7"), ("Swin", 25, "#3B83B7"), ("Hiera", 21, "#78C7E7")]
    for idx, (name, value, color) in enumerate(models):
        bx = x0 + 28 + idx * 94
        bh = int(265 * value / 160)
        draw.rectangle((bx, y0 - bh, bx + 62, y0), fill=color)
        draw_centered(draw, Box(bx - 15, y0 + 8, 85, 24), name, FONT_TINY, INK)
        draw_centered(draw, Box(bx - 8, y0 - bh - 32, 70, 24), str(value), FONT_TINY, INK)

    diagram_box = Box(box.x + 65, box.y + 500, box.w - 130, 300)
    draw.rounded_rectangle((diagram_box.x, diagram_box.y, diagram_box.x + diagram_box.w, diagram_box.y + diagram_box.h), radius=18, fill="white", outline="#D6D6D6", width=2)
    draw_centered(draw, Box(diagram_box.x, diagram_box.y + 16, diagram_box.w, 35), "Mermaid Streaming Evaluation Flow", FONT_SMALL_BOLD, TEAL)
    paste_image_fit(canvas, STREAMING_EVAL_DIAGRAM, Box(diagram_box.x + 45, diagram_box.y + 72, diagram_box.w - 90, diagram_box.h - 110), 0)

    note = Box(box.x + 85, box.y + box.h - 165, box.w - 170, 135)
    draw.rounded_rectangle((note.x, note.y, note.x + note.w, note.y + note.h), radius=28, fill="#FCEADF", outline=INK, width=2)
    draw_wrapped(
        draw,
        "Best deployment choice is selected on streaming behavior, not offline accuracy alone. Calibration reduces probability-scale mismatch and prevents repeated alerts from flickering frame predictions.",
        (note.x + 26, note.y + 24),
        FONT_SMALL,
        INK,
        note.w - 52,
        5,
    )


def generate(output: Path) -> None:
    image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "white")
    draw = ImageDraw.Draw(image)
    image = image.convert("RGBA")
    draw = ImageDraw.Draw(image)
    draw_header(image, draw)

    divider_x = CANVAS_SIZE // 2
    draw.line((divider_x, HEADER_HEIGHT + 35, divider_x, CANVAS_SIZE - 140), fill="#E22117", width=5)

    left_x = MARGIN
    right_x = divider_x + CENTER_GAP // 2
    col_w = divider_x - MARGIN - CENTER_GAP // 2
    right_w = CANVAS_SIZE - right_x - MARGIN

    problem = draw_panel(draw, Box(left_x, 500, col_w, 565), "Background and Problem Definition")
    problem_text_w = problem.w - 720
    bullet_list(
        draw,
        [
            "Workplace cameras already capture safety-relevant activity, but continuous manual monitoring does not scale.",
            "SafeStream-Kafka converts camera frames into per-camera safety counts, rolling unsafe ratios, and alert events.",
            "The system accepts video files, RTSP streams, webcams, and replayed CCTV clips.",
            "Aggregating frame labels before alerting reduces flicker and repeated false alarms.",
        ],
        Box(problem.x, problem.y + 12, problem_text_w, 350),
        FONT_SMALL,
        8,
    )
    draw_stream_contract_card(draw, Box(problem.x + problem_text_w + 64, problem.y + 4, 610, 270))
    draw_background_flow(draw, Box(problem.x + 54, problem.y + 326, problem.w - 108, 95))

    gaps = draw_panel(draw, Box(left_x, 1105, col_w, 650), "Research Gaps")
    draw_gap_cards(draw, Box(gaps.x, gaps.y + 5, gaps.w, 465))

    draw_bar(draw, Box(left_x + 210, 1785, col_w - 420, 140), "Key Contributions", radius=14)
    draw_contributions(draw, Box(left_x - 20, 1960, col_w + 40, 1045))

    draw_bar(draw, Box(left_x + 260, 3050, col_w - 520, 130), "System Architecture", radius=14)
    draw_architecture(image, draw, Box(left_x, 3235, col_w, 700))

    data_bar = Box(right_x + 350, 455, right_w - 700, 140)
    draw_bar(draw, data_bar, "Data Collection", radius=18)
    draw_data_collection(draw, Box(right_x, 630, right_w, 820))

    draw_bar(draw, Box(right_x + 350, 1480, right_w - 700, 135), "Results", radius=18)
    draw_results_table(draw, Box(right_x + 45, 1655, right_w - 90, 680))

    draw_bar(draw, Box(right_x + 350, 2385, right_w - 700, 135), "Feature Analysis", radius=18)
    draw_feature_chart(image, draw, Box(right_x + 45, 2555, right_w - 90, 1320))

    footer_y = CANVAS_SIZE - 100
    footer = "Code: github.com/Lumysia/SafeKafka   |   Course: ENGR 5785G Real-Time Data Analytics for IoT   |   Sensor Syndicate"
    draw.text((MARGIN + 20, footer_y), footer, font=FONT_SMALL_BOLD, fill=NAVY_DARK)

    output.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output, quality=95)


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
