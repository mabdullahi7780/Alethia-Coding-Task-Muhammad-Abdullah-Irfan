"""Our batched inference must give the same detections as plain model.predict() on real frames.

Needs real data and weights, so it is skipped unless these are set:
    ALETHIA_TEST_PARQUET  path to a FineVideo Parquet file (e.g. train-00000-of-01357.parquet)
    ALETHIA_WEIGHTS       path to yolo11l.pt (default: yolo11l.pt)
Optional: ALETHIA_DEVICE (cuda:0, mps, cpu), ALETHIA_FRAMES_PER_VIDEO (default 100), ALETHIA_TEST_BATCH (16).
Locally use a small file (devtools/make_mini_parquet.py); on Kaggle use a full Parquet file.
"""

from __future__ import annotations

import os

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
from ultralytics.utils import ops

PARQUET = os.environ.get("ALETHIA_TEST_PARQUET")
pytestmark = pytest.mark.skipif(not PARQUET, reason="set ALETHIA_TEST_PARQUET to run")

def load_frames(per_video: int):
    """One video at a time (low memory): every video in the file, every 7th frame."""
    from alethia.decode import iter_frames, to_model_input

    pf = pq.ParquetFile(PARQUET)
    for row, batch in enumerate(pf.iter_batches(batch_size=1, columns=["mp4"])):
        raw, model_in = [], []
        for i, (_, frame) in enumerate(iter_frames(batch.column(0)[0].as_py())):
            if i % 7:  # spread the frames out over the video
                continue
            raw.append(np.ascontiguousarray(frame.to_ndarray(format="bgr24")))
            model_in.append(to_model_input(frame))
            if len(raw) == per_video:
                break
        yield row, raw, model_in


def test_batched_matches_predict():
    from alethia.infer import IOU, Detector
    from alethia.rules import CONF

    weights = os.environ.get("ALETHIA_WEIGHTS", "yolo11l.pt")
    device = os.environ.get("ALETHIA_DEVICE") or None
    det = Detector(weights, device=device)
    ref_model = det.yolo  # same weights, same fused model: plain predict() vs our batched path

    n_frames = n_boxes = 0
    mismatches = []
    max_box_diff = max_conf_diff = 0.0
    for row, raw, model_in in load_frames(int(os.environ.get("ALETHIA_FRAMES_PER_VIDEO", 100))):
        bs = int(os.environ.get("ALETHIA_TEST_BATCH", 16))
        for start in range(0, len(model_in), bs):
            batch = np.stack(model_in[start:start + bs])
            ours_dets, ours_raw = det(batch)
            for j, frame in enumerate(raw[start:start + bs]):
                r = ref_model.predict(frame, imgsz=640, conf=CONF, iou=IOU, device=det.device, verbose=False)[0]
                ref = r.boxes.data.cpu().numpy()  # boxes in original-frame coords
                ref_lb = ref_model.predictor.preprocess([frame]).shape  # same letterbox shape check
                assert ref_lb[2:] == batch.shape[1:3], (row, ref_lb, batch.shape)
                ours = ours_raw[j].numpy()
                ours = ours[ours[:, 4] >= CONF]
                n_frames += 1
                n_boxes += len(ref)
                ref_key = sorted((int(c), round(float(s), 2)) for s, c in ref[:, 4:6])
                our_key = sorted((int(c), round(float(s), 2)) for s, c in ours[:, 4:6])
                if [c for c, _ in ref_key] != [c for c, _ in our_key]:
                    mismatches.append((row, start + j, ref_key, our_key))
                    continue
                if len(ref):
                    o = ours[np.lexsort((-ours[:, 4], ours[:, 5]))].copy()
                    f = ref[np.lexsort((-ref[:, 4], ref[:, 5]))]
                    max_conf_diff = max(max_conf_diff, float(np.abs(o[:, 4] - f[:, 4]).max()))
                    # ours are in letterbox coordinates: map back to the frame exactly as predict() does
                    o[:, :4] = ops.scale_boxes(batch.shape[1:3], torch.from_numpy(o[:, :4]), frame.shape).numpy()
                    max_box_diff = max(max_box_diff, float(np.abs(o[:, :4] - f[:, :4]).max()))
    print(f"\nframes {n_frames}, reference boxes {n_boxes}, class-set mismatches {len(mismatches)}, "
          f"max conf diff {max_conf_diff:.2e}, max box diff {max_box_diff:.4f}px")
    for m in mismatches[:10]:
        print("MISMATCH", m)
    assert not mismatches
    assert max_conf_diff < 1e-3 and max_box_diff < 0.5
