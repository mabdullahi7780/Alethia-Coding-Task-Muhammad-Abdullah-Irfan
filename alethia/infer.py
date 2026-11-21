"""Batched YOLO11l inference that reproduces Ultralytics' model.predict() exactly.

model.predict(frame, imgsz=640, conf=0.4, iou=0.7) does: letterbox on CPU (done by our decode workers),
upload uint8, BHWC -> BCHW, BGR -> RGB, float / 255, forward through the fused AutoBackend model,
then nms.non_max_suppression(conf, iou=0.7, classes=None, agnostic=False, max_det=300).
We run the same steps on a batch of same-shape frames.

NMS runs at a lower confidence floor (default 0.3) so that the skip mode can see detections near
the 0.4 cutoff. A lower-confidence box can never suppress a higher-confidence one, so every box at
0.4 or above is identical to an NMS run at 0.4; tests/test_infer.py checks this against predict().
"""

from __future__ import annotations

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils import nms

from .rules import CLASS_TO_BIT, CONF, FrameDet

IOU = 0.7  # the brief's value; note Ultralytics 8.4 predict() would default to 0.5, so it is always passed
MAX_DET = 300
NMS_NO_TIME_LIMIT = 1e6


def pick_device(requested: str | None = None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Detector:
    def __init__(self, weights: str = "yolo11l.pt", device: str | None = None, half: bool = False,
                 conf_floor: float = 0.3, margin: tuple[float, float] = (0.3, 0.5)):
        self.device = pick_device(device)
        self.half = half
        self.conf_floor = min(conf_floor, CONF, margin[0])
        self.margin = margin
        yolo = YOLO(weights)
        # Build the predictor exactly as model.predict() does (fused AutoBackend, eval mode, device).
        dummy = np.zeros((IMGSZ_DUMMY, IMGSZ_DUMMY, 3), np.uint8)
        kwargs = dict(imgsz=640, conf=CONF, iou=IOU, device=self.device, verbose=False)
        if half:
            kwargs["quantize"] = 16  # Ultralytics 8.4 name for FP16 (replaces the deprecated half=True)
        yolo.predict(dummy, **kwargs)
        self.yolo = yolo  # kept so tests can call the plain predict() on the very same model
        self.predictor = yolo.predictor
        self.model = yolo.predictor.model
        self.fp16 = bool(getattr(self.model, "fp16", False))

    @torch.inference_mode()
    def raw(self, batch: np.ndarray) -> list[torch.Tensor]:
        """batch: uint8 (B, H, W, 3) letterboxed BGR. Returns per-image (n, 6) [x1, y1, x2, y2, conf, cls] on CPU."""
        im = torch.from_numpy(batch).to(self.device, non_blocking=True)
        im = im.permute(0, 3, 1, 2).contiguous().flip(1)
        im = (im.half() if self.fp16 else im.float()).div_(255)
        preds = self.model(im)
        # max_time_img: Ultralytics stops NMS after 0.05 s per image + 2 s and silently drops the
        # detections of every image it has not reached yet. A slow batch must never lose detections.
        out = nms.non_max_suppression(preds, self.conf_floor, IOU, None, False, max_det=MAX_DET, nc=0,
                                      max_time_img=NMS_NO_TIME_LIMIT)
        return [o[:, :6].float().cpu() for o in out]

    def frame_dets(self, raw: list[torch.Tensor]) -> list[FrameDet]:
        lo, hi = self.margin
        result = []
        for det in raw:
            people, mask, near = 0, 0, False
            for conf, cls in det[:, 4:6].tolist():
                bit = CLASS_TO_BIT.get(int(cls))
                if bit is None:
                    continue
                if conf >= CONF:
                    mask |= 1 << bit
                    if bit == 0:
                        people += 1
                if lo <= conf < hi:
                    near = True
            result.append(FrameDet(people, mask, near))
        return result

    def __call__(self, batch: np.ndarray) -> tuple[list[FrameDet], list[torch.Tensor]]:
        raw = self.raw(batch)
        return self.frame_dets(raw), raw


IMGSZ_DUMMY = 64
