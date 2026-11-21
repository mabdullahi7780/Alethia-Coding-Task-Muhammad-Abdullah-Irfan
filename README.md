# Alethia Coding Task- Muhammad Abdullah Irfan

Video indexing with YOLO11l. For each second of each FineVideo video, `run.py` records which of
**person, car, dog, laptop, cell phone** appear and how many people are visible, and writes one JSON
line per second (section 4 of the brief).

```
python run.py --input <folder of .parquet files> --output results.jsonl
```

## Install and run (Ubuntu 22.04, Python 3.11, NVIDIA driver with CUDA 12)

1. System packages (OpenCV, installed by Ultralytics, needs these two libraries on a minimal Ubuntu):

   ```
   sudo apt-get update
   sudo apt-get install -y libgl1 libglib2.0-0
   ```

2. Python packages, in a fresh virtual environment:

   ```
   pip install -r requirements.txt
   ```

   `requirements.txt` pins exact versions and takes PyTorch from its CUDA 12.6 index
   (`torch==2.14.1+cu126`), which runs on any NVIDIA driver that supports CUDA 12.

3. **Preparation command** (run once, needs internet, not timed):

   ```
   python prepare.py
   ```

   It downloads the official `yolo11l.pt` next to `run.py`, checks its SHA-256, checks that PyTorch can
   see the GPU (it exits with an error if not), and runs the model once on the GPU.

4. **Timed run** (no internet needed):

   ```
   python run.py --input <folder of .parquet files> --output results.jsonl
   ```

   Progress is printed to stderr every 60 s. Next to the output it also writes `results.jsonl.stats.json`
   (timings and counts). All settings have defaults.

## What it does

- **Model settings** as in section 3: `yolo11l.pt`, input size 640, confidence 0.4, Ultralytics'
  aspect-preserving letterbox, class-aware NMS at IoU 0.7, BGR frames, rotation tags applied. The
  model input and NMS reproduce `model.predict(frame, imgsz=640, conf=0.4, iou=0.7)` exactly
  (`tests/test_infer.py`: 0 mismatches on 1,320 real frames on a T4). Ultralytics 8.4 would default to
  IoU 0.5 for `predict`, so 0.7 is always passed explicitly. Inference runs in FP16 on CUDA.
- **Parallel pipeline** (`alethia/pipeline.py`): one decode process per CPU core (PyAV) turns every
  frame into a letterboxed BGR image in shared memory; one GPU process per GPU batches frames from
  all videos and runs the model. A video's lines are written once the video is finished.
- **Seconds**: a frame belongs to second `floor(timestamp - first frame's timestamp)`, using exact
  timestamps; a line is written for every second up to the one holding the last frame.
- **Adaptive sampling** (`alethia/rules.py`): for each second the model first sees 5 evenly spaced
  frames. If every object is clearly present (in at least 4 of the 5) or clearly absent (in at most 1),
  and at least 4 of the 5 agree on the people count, the second is decided from them. Otherwise the
  model runs on every remaining frame of that second and the brief's section 5 rule is applied to all
  of them (present in at least half of the frames; people = median, rounded down at .5).
- **Damaged videos**: missing bytes, undecodable data and decoder failures give one `error` line per
  video (lines already produced are kept); a crashed or hung decode process is restarted and the run
  continues.

## Accuracy and speed (measured on Kaggle, 2x T4)

Against our every-frame FP32 reference on 96 videos (12 files, 60 s each, 5,543 seconds), see
`experiments/phase3_compare/`:

| | object F1 | people exact | people within 1 | model frames per video-second | video-seconds/s |
|---|---|---|---|---|---|
| every frame, FP32 (reference) | 1 | 100% | 100% | 27.8 | 3.34 |
| every frame, FP16 | 0.9999 | 99.82% | 100% | 27.8 | 8.21 |
| adaptive sampling (first settings), FP16 | 1.0000 | 99.86% | 100% | 12.5 | 16.05 |
| **`run.py`** (adaptive sampling, final settings), FP16 | 0.9993 | 99.78% | 100% | 8.6 | |

`run.py` uses cheaper settings chosen with `tools/sweep.py` (no confidence-margin check, people
majority rule); the last row is the full-run notebook re-validating them on the same 96 videos
(`experiments/phase4_full_run/`). The 50-file run (1,650 videos, 128 h of video) took 5 h 58 min on
the 2x T4 machine at 21.5 video-seconds per second, with no error lines; the results are in
`results.jsonl.gz`, and the machine and timings are in `HARDWARE.md`.

## Repository

| Path | What it is |
|---|---|
| `run.py` | the submitted script (adaptive sampling, FP16) |
| `run_full.py`, `run_skip.py` | every-frame reference and the first adaptive settings, same engine |
| `prepare.py` | preparation step: weights, GPU check |
| `alethia/` | decoding, inference, per-second rules, parallel pipeline, command line |
| `tests/` | unit tests (rules, decoding, letterbox) and the `predict()` equivalence test |
| `tools/` | `check_results.py` (format), `eval.py` (scores vs a reference), `sweep.py` (offline tuning), `inspect_videos.py`, `build_notebooks.py` |
| `kaggle/` | the Kaggle notebooks used (smoke test, comparison, full run) |
| `experiments/` | outputs of the smoke test, the comparison run and the 50-file run |
| `results.jsonl.gz`, `HARDWARE.md`, `WRITEUP.md`, `ai_chats/` | submission files |

Tests: `pytest tests/` (the equivalence test needs `ALETHIA_TEST_PARQUET` pointing at a FineVideo
Parquet file and a GPU).
