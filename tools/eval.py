"""Score a results file against a reference results file (our every-frame run).

Usage: python tools/eval.py --ref full.jsonl --pred skip.jsonl [--json out.json]

Metrics, over every (video, second) line in the reference:
    object F1     micro F1 over (video, second, object) items
    people exact  share of seconds where the people count is identical
    people +-1    share of seconds where it differs by at most one
A second missing from the prediction counts as no objects and a people mismatch.
Also prints the five worst videos by people exact-match, since errors cluster by video.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys


def load(path: str) -> tuple[dict, set]:
    seconds, errors = {}, set()
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if "error" in rec:
                errors.add(rec["video"])
            else:
                seconds[(rec["video"], rec["second"])] = (frozenset(rec["objects"]), rec["people"])
    return seconds, errors


def score(ref: dict, pred: dict) -> dict:
    tp = fp = fn = exact = within1 = 0
    per_video = collections.defaultdict(lambda: [0, 0])  # exact, total
    for key, (r_obj, r_people) in ref.items():
        p_obj, p_people = pred.get(key, (frozenset(), None))
        tp += len(r_obj & p_obj)
        fp += len(p_obj - r_obj)
        fn += len(r_obj - p_obj)
        ok = p_people == r_people
        exact += ok
        within1 += p_people is not None and abs(p_people - r_people) <= 1
        per_video[key[0]][0] += ok
        per_video[key[0]][1] += 1
    n = len(ref)
    f1 = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 1.0
    worst = sorted(((e / t, v, t) for v, (e, t) in per_video.items()))[:5]
    return {
        "seconds": n,
        "object_f1": round(f1, 5),
        "object_tp_fp_fn": [tp, fp, fn],
        "people_exact": round(exact / n, 5) if n else 1.0,
        "people_within1": round(within1 / n, 5) if n else 1.0,
        "missing_seconds": sum(k not in pred for k in ref),
        "extra_seconds": sum(k not in ref for k in pred),
        "worst_videos_people_exact": [{"video": v, "exact": round(r, 4), "seconds": t} for r, v, t in worst],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    ref, ref_err = load(a.ref)
    pred, pred_err = load(a.pred)
    result = score(ref, pred)
    result["error_videos_ref"], result["error_videos_pred"] = len(ref_err), len(pred_err)
    print(json.dumps(result, indent=2))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(result, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
