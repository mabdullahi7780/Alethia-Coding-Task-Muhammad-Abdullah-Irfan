"""Command line shared by run.py, run_full.py and run_skip.py.

    run.py       the submission: adaptive sampling with the settings chosen in experiments/ (option 3),
                 FP16 on CUDA. Same as: run_skip.py --half --no-margin --people-rule majority
    run_full.py  every frame through the model (the brief's section 5 rule); our reference
    run_skip.py  adaptive sampling with the first, more conservative defaults (margin on, all-equal)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .pipeline import Options, run

REPO = Path(__file__).resolve().parents[1]

PRESETS = {
    # option 3 from experiments/phase3: K=5, 80/20, no confidence margin, people majority, FP16
    "run": dict(mode="skip", half=True, samples=5, hi=0.8, lo=0.2, use_margin=False, people_rule="majority"),
    "skip": dict(mode="skip", half=False, samples=5, hi=0.8, lo=0.2, use_margin=True, people_rule="all-equal"),
    "full": dict(mode="full", half=False),
}


def default_weights() -> str:
    for cand in (REPO / "yolo11l.pt", Path.cwd() / "yolo11l.pt"):
        if cand.exists():
            return str(cand)
    return "yolo11l.pt"  # Ultralytics would download it; prepare.py does that before the timed run


def main(preset: str) -> int:
    p = PRESETS[preset]
    mode = p["mode"]
    ap = argparse.ArgumentParser(description=f"Index videos with YOLO11l ({preset}: {mode} mode).")
    ap.add_argument("--input", required=True, help="folder of .parquet files (or one file)")
    ap.add_argument("--output", required=True, help="results .jsonl file")
    ap.add_argument("--weights", default=os.environ.get("ALETHIA_WEIGHTS") or default_weights())
    ap.add_argument("--device", default=None, help="e.g. cuda:0, mps, cpu (default: every CUDA GPU)")
    prec = ap.add_mutually_exclusive_group()
    prec.add_argument("--half", dest="half", action="store_true", default=p["half"], help="FP16 inference (CUDA only)")
    prec.add_argument("--fp32", dest="half", action="store_false", help="FP32 inference")
    ap.add_argument("--workers", type=int, default=0, help="decode processes (default: CPU cores minus GPUs)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--slots", type=int, default=160, help="frame buffers per decode process")
    ap.add_argument("--decode-threads", type=int, default=1)
    ap.add_argument("--dump-frames", default=None, help="write per-frame detections (offline tuning)")
    ap.add_argument("--max-seconds", type=float, default=0, help="evaluation only: first N seconds of each video")
    ap.add_argument("--videos-per-file", type=int, default=0, help="evaluation only: first N videos per file")
    ap.add_argument("--stats-every", type=float, default=60)
    if mode == "skip":
        ap.add_argument("--samples", type=int, default=p["samples"], help="frames sampled per second (K)")
        ap.add_argument("--hi", type=float, default=p["hi"], help="present if seen in at least this share of samples")
        ap.add_argument("--lo", type=float, default=p["lo"], help="absent if seen in at most this share of samples")
        ap.add_argument("--margin", type=float, nargs=2, default=(0.3, 0.5),
                        help="with the margin check on: a sampled detection with confidence in [lo, hi) "
                             "makes the second borderline")
        m = ap.add_mutually_exclusive_group()
        m.add_argument("--use-margin", dest="use_margin", action="store_true", default=p["use_margin"])
        m.add_argument("--no-margin", dest="use_margin", action="store_false")
        ap.add_argument("--people-rule", choices=["all-equal", "majority"], default=p["people_rule"])
    a = ap.parse_args()

    opts = Options(
        input=a.input, output=a.output, mode=mode, weights=a.weights, device=a.device, half=a.half,
        workers=a.workers, batch=a.batch, slots=a.slots, decode_threads=a.decode_threads,
        dump_frames=a.dump_frames, max_seconds=a.max_seconds, videos_per_file=a.videos_per_file,
        stats_every=a.stats_every,
    )
    if mode == "skip":
        opts.samples, opts.hi, opts.lo = a.samples, a.hi, a.lo
        opts.margin, opts.use_margin, opts.people_rule = tuple(a.margin), a.use_margin, a.people_rule
    return run(opts)
