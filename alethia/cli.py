"""Command line shared by run.py, run_full.py and run_skip.py."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .pipeline import Options, run

REPO = Path(__file__).resolve().parents[1]


def default_weights() -> str:
    for cand in (REPO / "yolo11l.pt", Path.cwd() / "yolo11l.pt"):
        if cand.exists():
            return str(cand)
    return "yolo11l.pt"  # Ultralytics downloads it (the README preparation step does this ahead of time)


def main(mode: str) -> int:
    ap = argparse.ArgumentParser(description=f"Index videos with YOLO11l ({mode} mode).")
    ap.add_argument("--input", required=True, help="folder of .parquet files (or one file)")
    ap.add_argument("--output", required=True, help="results .jsonl file")
    ap.add_argument("--weights", default=os.environ.get("ALETHIA_WEIGHTS") or default_weights())
    ap.add_argument("--device", default=None, help="e.g. cuda:0, mps, cpu (default: every CUDA GPU)")
    ap.add_argument("--half", action="store_true", help="FP16 inference")
    ap.add_argument("--workers", type=int, default=0, help="decode processes (default: CPU cores minus GPUs)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--slots", type=int, default=160, help="frame buffers per decode process")
    ap.add_argument("--decode-threads", type=int, default=1)
    ap.add_argument("--dump-frames", default=None, help="write per-frame detections (offline tuning)")
    ap.add_argument("--max-seconds", type=float, default=0, help="evaluation only: first N seconds of each video")
    ap.add_argument("--videos-per-file", type=int, default=0, help="evaluation only: first N videos per file")
    ap.add_argument("--stats-every", type=float, default=30)
    if mode == "skip":
        ap.add_argument("--samples", type=int, default=5, help="frames sampled per second (K)")
        ap.add_argument("--hi", type=float, default=0.8, help="present if seen in at least this share of samples")
        ap.add_argument("--lo", type=float, default=0.2, help="absent if seen in at most this share of samples")
        ap.add_argument("--margin", type=float, nargs=2, default=(0.3, 0.5),
                        help="a sampled detection with confidence in [lo, hi) makes the second borderline")
        ap.add_argument("--no-margin", action="store_true", help="turn off the confidence-margin check")
    a = ap.parse_args()

    opts = Options(
        input=a.input, output=a.output, mode=mode, weights=a.weights, device=a.device, half=a.half,
        workers=a.workers, batch=a.batch, slots=a.slots, decode_threads=a.decode_threads,
        dump_frames=a.dump_frames, max_seconds=a.max_seconds, videos_per_file=a.videos_per_file,
        stats_every=a.stats_every,
    )
    if mode == "skip":
        opts.samples, opts.hi, opts.lo = a.samples, a.hi, a.lo
        opts.margin, opts.use_margin = tuple(a.margin), not a.no_margin
    return run(opts)


if __name__ == "__main__":
    sys.exit(main("full"))
