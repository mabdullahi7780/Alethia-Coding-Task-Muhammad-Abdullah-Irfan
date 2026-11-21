"""Offline sweep of the skip-mode settings, from per-frame detections saved by run_full.py --dump-frames.

Usage: python tools/sweep.py --dump full32.frames [--ref full32.jsonl] [--out sweep.csv]
       (reads full32.frames.gpu0, full32.frames.gpu1, ...)

For every combination of K (samples per second), presence thresholds (hi, lo), confidence margin and
people rule, it replays the skip decision for every second exactly as alethia.rules does, and scores the
result against the exact every-frame rule. No GPU needed: every frame's detections are already known.

Reported per setting:
    cost            frames through the model per video-second (full mode = average frames per second)
    escalated       share of sampled seconds that had to run every frame
    f1 / exact / w1 object F1, people exact-match, people within one (all seconds pooled)
    exact_p5        5th percentile over videos of each video's people exact-match (errors cluster by video)
    triggers        why seconds escalated (margin / people / presence); a second can have several reasons
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alethia.rules import CONF, FrameDet, exact_second, sample_indices  # noqa: E402

MARGINS = {"off": None, "0.30-0.50": (0.30, 0.50), "0.35-0.45": (0.35, 0.45)}
THRESHOLDS = [(1.0, 0.0), (0.8, 0.2), (0.6, 0.4)]
KS = [3, 4, 5, 6, 8]
PEOPLE_RULES = ["all-equal", "majority"]


def load(dump: str):
    """-> list of (video, second, frames) where frames is a list of [(bit, conf), ...] per frame."""
    seconds = []
    for path in sorted(glob.glob(dump + ".gpu*")):
        with open(path) as fh:
            for line in fh:
                rec = json.loads(line)
                seconds.append((rec["video"], rec["second"], [[tuple(d) for d in f] for f in rec["frames"]]))
    return seconds


def frame_det(dets, margin) -> FrameDet:
    people = mask = 0
    near = False
    for bit, conf in dets:
        if conf >= CONF:
            mask |= 1 << bit
            people += bit == 0
        if margin and margin[0] <= conf < margin[1]:
            near = True
    return FrameDet(people, mask, near)


def decide(samples, hi, lo, use_margin, people_rule):
    """Same logic as alethia.rules.adaptive_decision, plus a 'majority' people rule. Returns (result, reasons)."""
    k = len(samples)
    reasons = set()
    if use_margin and any(f.near for f in samples):
        reasons.add("margin")
    counts = collections.Counter(f.people for f in samples)
    people, top = counts.most_common(1)[0]
    if people_rule == "all-equal":
        if top != k:
            reasons.add("people")
    elif top / k < hi:  # majority: the most common count must reach the presence threshold
        reasons.add("people")
    mask = 0
    for i in range(5):
        frac = sum(f.mask >> i & 1 for f in samples) / k
        if frac >= hi:
            mask |= 1 << i
        elif frac > lo:
            reasons.add("presence")
    if reasons:
        return None, reasons
    from alethia.rules import objects_from_mask
    return (objects_from_mask(mask), people), reasons


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--ref", default=None, help="full-mode .jsonl, to check the dump reproduces it")
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-f1", type=float, default=0.97)
    ap.add_argument("--min-exact", type=float, default=0.90)
    ap.add_argument("--min-w1", type=float, default=0.97)
    a = ap.parse_args()

    seconds = load(a.dump)
    n_frames = sum(len(f) for _, _, f in seconds)
    print(f"{len(seconds)} seconds, {n_frames} frames, {len({v for v, _, _ in seconds})} videos")

    dets = {name: [[frame_det(d, m) for d in frames] for _, _, frames in seconds] for name, m in MARGINS.items()}
    reference = [exact_second(fr) for fr in dets["off"]]

    if a.ref:
        ref_lines = {}
        for line in open(a.ref):
            rec = json.loads(line)
            if "second" in rec:
                ref_lines[(rec["video"], rec["second"])] = (rec["objects"], rec["people"])
        same = sum(ref_lines.get((v, s)) == (r[0], r[1]) for (v, s, _), r in zip(seconds, reference))
        print(f"dump reproduces the full-mode output on {same}/{len(seconds)} seconds")

    videos = [v for v, _, _ in seconds]
    rows = []
    for k, (hi, lo), mname, prule in itertools.product(KS, THRESHOLDS, MARGINS, PEOPLE_RULES):
        use_margin = MARGINS[mname] is not None
        cost = esc = sampled = tp = fp = fn = exact = w1 = 0
        triggers = collections.Counter()
        per_video = collections.defaultdict(lambda: [0, 0])
        for i, frames in enumerate(dets[mname]):
            n = len(frames)
            ref_obj, ref_people = reference[i]
            if n <= k:
                result, c = reference[i], n
            else:
                sampled += 1
                result, reasons = decide([frames[j] for j in sample_indices(n, k)], hi, lo, use_margin, prule)
                if result is None:
                    esc += 1
                    triggers.update(reasons)
                    result, c = reference[i], n
                else:
                    c = k
            cost += c
            r, p = set(ref_obj), set(result[0])
            tp += len(r & p)
            fp += len(p - r)
            fn += len(r - p)
            ok = result[1] == ref_people
            exact += ok
            w1 += abs(result[1] - ref_people) <= 1
            per_video[videos[i]][0] += ok
            per_video[videos[i]][1] += 1
        ns = len(seconds)
        vid_exact = sorted(e / t for e, t in per_video.values())
        rows.append({
            "K": k, "hi": hi, "lo": lo, "margin": mname, "people_rule": prule,
            "cost": round(cost / ns, 2), "escalated": round(esc / max(sampled, 1), 3),
            "f1": round(2 * tp / max(2 * tp + fp + fn, 1), 4) if (tp + fp + fn) else 1.0,
            "exact": round(exact / ns, 4), "w1": round(w1 / ns, 4),
            "exact_p5": round(float(np.percentile(vid_exact, 5)), 4),
            "trig_margin": triggers["margin"], "trig_people": triggers["people"], "trig_presence": triggers["presence"],
        })
    full_cost = round(n_frames / len(seconds), 2)
    ok = [r for r in rows if r["f1"] >= a.min_f1 and r["exact"] >= a.min_exact and r["w1"] >= a.min_w1]
    ok.sort(key=lambda r: (r["cost"], -r["exact"]))
    cols = list(rows[0])
    print(f"\nfull mode cost: {full_cost} frames per video-second")
    print(f"settings meeting f1>={a.min_f1}, exact>={a.min_exact}, w1>={a.min_w1}: {len(ok)} of {len(rows)}; cheapest 15:")
    print("  ".join(cols))
    for r in ok[:15]:
        print("  ".join(str(r[c]) for c in cols))
    default = next(r for r in rows if (r["K"], r["hi"], r["margin"], r["people_rule"]) == (5, 0.8, "0.30-0.50", "all-equal"))
    print("\ncurrent default (K=5, 0.8/0.2, margin 0.30-0.50, all-equal):")
    print(json.dumps(default))
    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: r["cost"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
