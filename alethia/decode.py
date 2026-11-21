"""Video decoding: timestamps, rotation, BGR conversion and the Ultralytics letterbox.

Frames are bucketed into seconds by presentation timestamp measured from the first decoded
frame (section 4 of the brief), using exact fractions so that 12.0 never drifts to 11.999.
The letterbox reproduces Ultralytics' LetterBox with the settings model.predict() uses for a single
image (imgsz 640, auto=True, stride 32), so the model input is byte-identical to the reference.
"""

from __future__ import annotations

import io
from fractions import Fraction
from typing import Iterator

import av
import cv2
import numpy as np

IMGSZ = 640
STRIDE = 32
PAD_VALUE = 114


def letterbox(img: np.ndarray) -> np.ndarray:
    """Ultralytics LetterBox(new_shape=(640, 640), auto=True, stride=32), copied line for line.

    Re-implemented here (OpenCV only) so decode workers do not import torch, which costs several
    hundred MB per process. tests/test_decode.py checks it is pixel-identical to Ultralytics' class.
    """
    shape = img.shape[:2]
    r = min(IMGSZ / shape[0], IMGSZ / shape[1])
    new_unpad = max(round(shape[1] * r), 1), max(round(shape[0] * r), 1)
    dw, dh = IMGSZ - new_unpad[0], IMGSZ - new_unpad[1]
    dw, dh = np.mod(dw, STRIDE), np.mod(dh, STRIDE)
    dw /= 2
    dh /= 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3)


def rotation_k(frame: av.VideoFrame) -> int:
    """Number of 90-degree counterclockwise turns (np.rot90 k) that make the frame upright.

    PyAV reports the display-matrix rotation in degrees, counterclockwise. FFmpeg's own
    autorotate (used by OpenCV and the ffmpeg CLI) turns the picture by that same angle.
    The sign is checked against OpenCV in tests/test_decode.py.
    """
    angle = getattr(frame, "rotation", 0) or 0
    return int(round(angle / 90.0)) % 4


def to_model_input(frame: av.VideoFrame) -> np.ndarray:
    """Decoded frame -> upright, letterboxed BGR uint8 array (H, W, 3), as model.predict() would see it."""
    img = frame.to_ndarray(format="bgr24")
    k = rotation_k(frame)
    if k:
        img = np.rot90(img, k)
    return np.ascontiguousarray(letterbox(np.ascontiguousarray(img)))


def letterbox_shape(height: int, width: int) -> tuple[int, int]:
    """Shape the letterbox produces for an upright frame of this size (for buffer sizing)."""
    r = min(IMGSZ / height, IMGSZ / width)
    new_w, new_h = int(round(width * r)), int(round(height * r))
    dw, dh = (IMGSZ - new_w) % STRIDE, (IMGSZ - new_h) % STRIDE
    return new_h + dh, new_w + dw


def iter_frames(data: bytes, threads: int = 1) -> Iterator[tuple[int, av.VideoFrame]]:
    """Yield (second, frame) in decode order. Second = floor((pts - first_pts) * time_base)."""
    with av.open(io.BytesIO(data)) as container:
        if not container.streams.video:
            raise ValueError("no video stream")
        stream = container.streams.video[0]
        if threads > 1:
            stream.thread_type = "AUTO"
            stream.codec_context.thread_count = threads
        else:
            stream.codec_context.thread_count = 1
        time_base = stream.time_base or Fraction(1, 90000)
        rate = stream.average_rate or stream.base_rate or Fraction(30)
        first = None
        last_pts = None
        step = Fraction(1, 1) / Fraction(rate) / time_base  # one frame, in time_base units
        for frame in container.decode(stream):
            pts = frame.pts
            if pts is None:  # no timestamp: one frame after the previous one
                pts = 0 if last_pts is None else last_pts + step
            if first is None:
                first = pts
            last_pts = pts
            rel = Fraction(pts - first) * time_base
            yield max(0, int(rel // 1)), frame
