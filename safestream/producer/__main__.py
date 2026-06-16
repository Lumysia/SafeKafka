"""SafeStream producer: video / RTSP / webcam / dataset directory -> cctv-frames.

Examples:
    # Single file
    python -m safestream.producer --video clip.mp4

    # Whole dataset directory (replayed in order, can loop forever)
    python -m safestream.producer --video-dir ~/fiftyone/Voxel51_Safe_and_Unsafe --loop

    # Same dataset split across two synthetic cameras
    python -m safestream.producer --video-dir clips/ --cameras cam-01,cam-02

    # Mac webcam
    python -m safestream.producer --webcam 0

    # Real RTSP
    python -m safestream.producer --rtsp rtsp://192.168.1.20:554/stream

Frames are JPEG-encoded, base64-wrapped, and published as JSON keyed by
camera_id so that downstream consumers see one ordered partition per camera.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import random
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2

from safestream.common.encoding import encode_frame_jpeg
from safestream.common.kafka_clients import make_producer
from safestream.settings import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s :: %(message)s",
)
logger = logging.getLogger("safestream.producer")

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm")


def parse_args() -> argparse.Namespace:
    s = get_settings()
    p = argparse.ArgumentParser(description="SafeStream Kafka producer")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=str, action="append",
                     help="Path to one video file. Repeat the flag to queue several.")
    src.add_argument("--video-dir", type=str,
                     help="Directory of video clips to stream in order")
    src.add_argument("--rtsp", type=str, help="RTSP URL of a live camera")
    src.add_argument(
        "--webcam",
        type=int,
        nargs="?",
        const=0,
        help="Use a local webcam device index (default 0)",
    )

    p.add_argument("--camera-id", default=s.producer_camera_id,
                   help="Single camera_id when only one synthetic camera is used")
    p.add_argument("--cameras", default=None,
                   help="Comma-separated camera_ids, e.g. cam-01,cam-02. "
                        "Clips are dealt round-robin across them, so the dashboard "
                        "shows N cards.")

    p.add_argument("--fps", type=float, default=s.producer_analytics_fps,
                   help="Analytics frame rate")
    p.add_argument("--jpeg-quality", type=int, default=s.producer_jpeg_quality)
    p.add_argument("--max-frames", type=int, default=None,
                   help="Stop after N frames in total")
    p.add_argument("--max-clips", type=int, default=None,
                   help="Limit how many clips to play (handy with --video-dir)")
    p.add_argument("--shuffle", action="store_true",
                   help="Shuffle the order of clips (only with --video-dir / multi --video)")
    p.add_argument("--loop", action="store_true",
                   help="When the playlist ends, start again from the top")
    p.add_argument("--realtime", action="store_true",
                   help="Sleep between frames so playback matches --fps")
    return p.parse_args()


_STOP = False


def _install_signal_handlers() -> None:
    def _handler(signum, _frame):
        global _STOP
        logger.info("Signal %s received, shutting down...", signum)
        _STOP = True

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def _collect_videos(args) -> List[str]:
    """Return an ordered list of video sources, or [] for live/webcam modes."""
    paths: List[str] = []
    if args.video:
        for v in args.video:
            p = Path(v).expanduser().resolve()
            if not p.exists():
                logger.error("Video file does not exist: %s", p)
                sys.exit(2)
            paths.append(str(p))
    elif args.video_dir:
        root = Path(args.video_dir).expanduser().resolve()
        if not root.exists():
            logger.error("Video directory does not exist: %s", root)
            sys.exit(2)
        for ext in VIDEO_EXTS:
            paths.extend(str(p) for p in root.rglob(f"*{ext}"))
        if not paths:
            logger.error("No video files found under %s "
                         "(looked for extensions: %s)", root, ", ".join(VIDEO_EXTS))
            sys.exit(2)
        paths.sort()
    if args.shuffle and paths:
        random.seed(42)
        random.shuffle(paths)
    if args.max_clips and paths:
        paths = paths[: args.max_clips]
    return paths


def _camera_ids(args) -> List[str]:
    if args.cameras:
        ids = [c.strip() for c in args.cameras.split(",") if c.strip()]
        if not ids:
            logger.error("--cameras was empty after parsing")
            sys.exit(2)
        return ids
    return [args.camera_id]


def _open_capture(source) -> cv2.VideoCapture:
    return cv2.VideoCapture(source) if not isinstance(source, int) else cv2.VideoCapture(source)


def _delivery_report(err, _msg):
    if err is not None:
        logger.warning("Delivery failed: %s", err)


def _stream_capture(
    cap: cv2.VideoCapture,
    source_name: str,
    camera_id: str,
    args,
    producer,
    s,
    sent_start: int,
) -> int:
    """Stream a single open capture. Returns number of frames sent."""
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    stride = max(1, int(round(src_fps / max(args.fps, 0.1))))
    period = 1.0 / max(args.fps, 0.1)
    logger.info(
        "[%s] %s opened (native_fps=%.1f stride=%d target_fps=%.1f)",
        camera_id, source_name, src_fps, stride, args.fps,
    )

    sent = sent_start
    idx = 0
    next_emit = time.time()
    while not _STOP:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if idx % stride == 0:
            try:
                b64 = encode_frame_jpeg(frame, args.jpeg_quality)
            except RuntimeError as e:
                logger.warning("Encode failed: %s", e)
                idx += 1
                continue
            payload = {
                "camera_id": camera_id,
                "frame_id": sent,
                "source": source_name,
                "timestamp": time.time(),
                "image_b64": b64,
            }
            producer.produce(
                s.topic_frames,
                key=camera_id,
                value=json.dumps(payload),
                on_delivery=_delivery_report,
            )
            producer.poll(0)
            sent += 1
            if sent % 25 == 0:
                logger.info("Sent %d frames", sent)
            if args.max_frames and sent >= args.max_frames:
                logger.info("Reached --max-frames=%d, stopping", args.max_frames)
                _set_stop(True)
                break
            if args.realtime:
                next_emit += period
                delay = next_emit - time.time()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_emit = time.time()
        idx += 1
    return sent


def run() -> int:
    args = parse_args()
    s = get_settings()
    _install_signal_handlers()

    logger.info("Settings:\n%s", s.summary())

    videos = _collect_videos(args)
    cam_ids = _camera_ids(args)
    is_live = bool(args.rtsp) or args.webcam is not None

    if videos:
        logger.info("Playlist: %d clip(s) across %d camera(s): %s",
                    len(videos), len(cam_ids), cam_ids)
    else:
        logger.info("Live source mode, camera_id=%s", cam_ids[0])

    producer = make_producer()
    sent = 0
    try:
        if is_live:
            # Webcam / RTSP — single endless source
            source = args.rtsp if args.rtsp else args.webcam
            source_name = args.rtsp if args.rtsp else f"webcam:{args.webcam}"
            while not _STOP:
                cap = _open_capture(source)
                if not cap.isOpened():
                    logger.error("Could not open source: %s", source_name)
                    return 3
                try:
                    sent = _stream_capture(
                        cap, source_name, cam_ids[0], args, producer, s, sent
                    )
                finally:
                    cap.release()
                if not args.loop:
                    break
        else:
            # Playlist of video files, optionally round-robined across cameras
            if len(cam_ids) > 1:
                threads = []
                for index, cam_id in enumerate(cam_ids):
                    cam_videos = videos[index::len(cam_ids)] or videos
                    thread = threading.Thread(
                        target=_stream_video_list,
                        args=(cam_videos, cam_id, args, s),
                        daemon=True,
                        name=f"producer-{cam_id}",
                    )
                    thread.start()
                    threads.append(thread)
                while not _STOP and any(thread.is_alive() for thread in threads):
                    time.sleep(0.5)
            else:
                playlist: Iterable[Tuple[str, str]] = _build_playlist(videos, cam_ids,
                                                                     loop=args.loop)
                for vp, cam_id in playlist:
                    if _STOP:
                        break
                    cap = cv2.VideoCapture(vp)
                    if not cap.isOpened():
                        logger.warning("Could not open %s, skipping", vp)
                        continue
                    try:
                        sent = _stream_capture(cap, vp, cam_id, args, producer, s, sent)
                    finally:
                        cap.release()
    finally:
        logger.info("Flushing producer (sent=%d)...", sent)
        producer.flush(10)
    return 0


def _stream_video_list(videos: List[str], camera_id: str, args, s) -> None:
    producer = make_producer()
    sent = 0
    try:
        while not _STOP:
            for vp in videos:
                if _STOP:
                    break
                cap = cv2.VideoCapture(vp)
                if not cap.isOpened():
                    logger.warning("Could not open %s, skipping", vp)
                    continue
                try:
                    sent = _stream_capture(cap, vp, camera_id, args, producer, s, sent)
                finally:
                    cap.release()
            if not args.loop:
                break
    finally:
        logger.info("[%s] flushing producer (sent=%d)...", camera_id, sent)
        producer.flush(10)


def _build_playlist(videos: List[str], cam_ids: List[str], loop: bool) -> Iterable[Tuple[str, str]]:
    """Yield (video_path, camera_id) pairs. With loop=True the playlist is endless."""
    cam_cycle = itertools.cycle(cam_ids)

    def _one_pass() -> Iterable[Tuple[str, str]]:
        for vp in videos:
            yield vp, next(cam_cycle)

    if loop:
        while True:
            yield from _one_pass()
    else:
        yield from _one_pass()


def _set_stop(value: bool) -> None:
    global _STOP
    _STOP = value


if __name__ == "__main__":
    sys.exit(run())
