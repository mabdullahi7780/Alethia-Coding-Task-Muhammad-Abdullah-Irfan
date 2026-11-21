"""Decoder tests on synthetic clips: rotation, late start, variable frame rate, short last second.

The rotated clip needs an ffmpeg binary; tests use the one from the imageio-ffmpeg package
(a test-only dependency, not needed by run.py).
"""

from __future__ import annotations

import collections
import io
import subprocess
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

from alethia.decode import iter_frames, letterbox_shape, to_model_input


def encode(pts_ms: list[int], width=96, height=64, codec="libx264") -> bytes:
    """Encode an H.264 mp4 whose frames have the given timestamps (ms). Each frame is distinct."""
    buf = io.BytesIO()
    with av.open(buf, "w", format="mp4") as out:
        stream = out.add_stream(codec, rate=30)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.time_base = Fraction(1, 1000)
        stream.codec_context.time_base = Fraction(1, 1000)
        for i, t in enumerate(pts_ms):
            img = np.zeros((height, width, 3), np.uint8)
            img[: height // 2, : width // 3] = (0, 0, 255)  # red block top-left: orientation marker
            img[-8:, :] = (i * 7) % 255
            frame = av.VideoFrame.from_ndarray(img, format="bgr24")
            frame.pts = t
            frame.time_base = Fraction(1, 1000)
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return buf.getvalue()


def seconds_histogram(data: bytes) -> dict[int, int]:
    return dict(collections.Counter(sec for sec, _ in iter_frames(data)))


def test_constant_rate_and_short_last_second():
    data = encode([round(i * 1000 / 10) for i in range(22)])  # 10 fps, 2.2 s
    assert seconds_histogram(data) == {0: 10, 1: 10, 2: 2}


def test_boundary_frame_lands_in_next_second():
    data = encode([0, 500, 999, 1000, 1999, 2000])
    assert seconds_histogram(data) == {0: 3, 1: 2, 2: 1}


def test_timestamps_counted_from_first_decoded_frame():
    data = encode([5000 + i * 100 for i in range(25)])  # stream starts at 5 s
    assert seconds_histogram(data) == {0: 10, 1: 10, 2: 5}


def test_variable_frame_rate():
    pts = [0, 40, 80, 600, 990, 1010, 1500, 3100, 3200]  # bursts, and no frames at all in second 2
    assert seconds_histogram(encode(pts)) == {0: 5, 1: 2, 3: 2}


def test_letterbox_shape_matches_ultralytics():
    data = encode([0], width=640, height=360)
    _, frame = next(iter_frames(data))
    assert to_model_input(frame).shape == (384, 640, 3)
    assert letterbox_shape(360, 640) == (384, 640)
    assert letterbox_shape(480, 270) == (640, 384)
    assert letterbox_shape(236, 416) == (384, 640)


@pytest.mark.parametrize("angle", [90, -90, 180])
def test_rotation_matches_opencv_autorotate(tmp_path: Path, angle: int):
    ffmpeg = pytest.importorskip("imageio_ffmpeg").get_ffmpeg_exe()
    src, rotated = tmp_path / "src.mp4", tmp_path / f"rot{angle}.mp4"
    src.write_bytes(encode([0, 33, 66], width=96, height=64))
    subprocess.run([ffmpeg, "-loglevel", "error", "-display_rotation", str(angle), "-i", str(src),
                    "-c", "copy", str(rotated)], check=True)

    _, frame = next(iter_frames(rotated.read_bytes()))
    ours = to_model_input(frame)

    cap = cv2.VideoCapture(str(rotated))  # OpenCV applies the display matrix by default
    ok, ref = cap.read()
    cap.release()
    assert ok
    from ultralytics.data.augment import LetterBox
    ref = LetterBox((640, 640), auto=True, stride=32)(image=ref)
    assert ours.shape == ref.shape, (angle, ours.shape, ref.shape)
    assert np.abs(ours.astype(int) - ref.astype(int)).mean() < 3.0  # same picture, same orientation


def test_letterbox_identical_to_ultralytics():
    from ultralytics.data.augment import LetterBox

    from alethia.decode import letterbox

    ref = LetterBox((640, 640), auto=True, stride=32)
    rng = np.random.default_rng(0)
    sizes = [(360, 640), (480, 270), (272, 480), (640, 360), (236, 416), (360, 202), (360, 626),
             (1080, 1920), (1920, 1080), (720, 1280), (1, 1), (17, 999), (999, 17), (640, 640), (100, 100)]
    sizes += [tuple(int(v) for v in rng.integers(1, 2000, 2)) for _ in range(40)]
    for h, w in sizes:
        img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        a, b = letterbox(img), ref(image=img)
        assert a.shape == b.shape and np.array_equal(a, b), (h, w, a.shape, b.shape)
