"""Inspect every video in one or more FineVideo Parquet files (Phase 1, step 1.3).

Decodes each video fully (no pixel conversion) and records the properties that
matter for matching the reference: resolution, codec, frame rate, timestamps,
variable frame rate, rotation, and frame count.

Usage:
    python tools/inspect_videos.py --input <parquet file or folder> --out inspect.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import statistics
import sys
import time
from fractions import Fraction
from pathlib import Path

import av
import pyarrow.parquet as pq


def rotation_of(stream, first_frame) -> str:
    """Report every rotation signal we can see, so the decode step can pick the right one."""
    signals = []
    tag = (stream.metadata or {}).get("rotate")
    if tag is not None:
        signals.append(f"tag:{tag}")
    frame_rot = getattr(first_frame, "rotation", None) if first_frame is not None else None
    if frame_rot:
        signals.append(f"frame:{frame_rot}")
    return ";".join(signals)


def inspect_video(data: bytes) -> dict:
    row: dict = {}
    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        cc = stream.codec_context
        row.update(
            codec=cc.name,
            width=cc.width,
            height=cc.height,
            pix_fmt=cc.pix_fmt,
            avg_fps=float(stream.average_rate) if stream.average_rate else None,
            time_base=str(stream.time_base),
            n_video_streams=len(container.streams.video),
        )
        pts = []
        first = None
        for frame in container.decode(stream):
            if first is None:
                first = frame
            pts.append(frame.pts)
        row["rotation"] = rotation_of(stream, first)
        row["frames"] = len(pts)
        if pts:
            tb = stream.time_base
            row["first_pts_s"] = float(pts[0] * tb)
            rel = [Fraction(p - pts[0]) * tb for p in pts]
            row["last_ts_s"] = float(rel[-1])
            row["seconds_lines"] = int(rel[-1]) + 1  # floor(last) + 1 lines per section 4
            deltas = [b - a for a, b in zip(pts, pts[1:])]
            row["non_monotonic_pts"] = sum(d <= 0 for d in deltas)
            if deltas:
                med = statistics.median(deltas)
                row["vfr_frac"] = round(sum(abs(d - med) > 0.5 * med for d in deltas) / len(deltas), 4)
            # min and max frames that land in a single second (matters for "half of the frames")
            counts: dict[int, int] = {}
            for t in rel:
                counts[int(t)] = counts.get(int(t), 0) + 1
            row["min_frames_per_sec"] = min(counts.values())
            row["max_frames_per_sec"] = max(counts.values())
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many videos (0 = all)")
    args = ap.parse_args()

    src = Path(args.input)
    files = sorted(src.glob("*.parquet")) if src.is_dir() else [src]
    fields = [
        "video", "bytes", "error", "codec", "width", "height", "pix_fmt", "avg_fps", "time_base",
        "n_video_streams", "rotation", "frames", "first_pts_s", "last_ts_s", "seconds_lines",
        "non_monotonic_pts", "vfr_frac", "min_frames_per_sec", "max_frames_per_sec", "decode_s",
    ]
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for path in files:
            pf = pq.ParquetFile(path)
            row_idx = 0
            for rg in range(pf.num_row_groups):
                if args.limit and row_idx >= args.limit:
                    break
                table = pf.read_row_group(rg, columns=["mp4"])
                for value in table.column("mp4").to_pylist():
                    if args.limit and row_idx >= args.limit:
                        break
                    rec = {"video": f"{path.name}:{row_idx}", "bytes": len(value) if value else 0}
                    t0 = time.perf_counter()
                    try:
                        if not value:
                            raise ValueError("empty mp4 bytes")
                        rec.update(inspect_video(value))
                    except Exception as exc:  # record and keep going, as run.py must
                        rec["error"] = f"{type(exc).__name__}: {exc}"[:200]
                    rec["decode_s"] = round(time.perf_counter() - t0, 2)
                    writer.writerow(rec)
                    handle.flush()
                    print(rec["video"], rec.get("width"), rec.get("height"), rec.get("avg_fps"),
                          rec.get("frames"), rec.get("rotation"), rec.get("error", ""), file=sys.stderr)
                    row_idx += 1


if __name__ == "__main__":
    main()
