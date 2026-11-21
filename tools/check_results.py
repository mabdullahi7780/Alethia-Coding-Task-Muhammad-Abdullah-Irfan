"""Check a results .jsonl file against the output format in section 4 of the brief.

Usage: python tools/check_results.py results.jsonl [--expect-videos N]

Checks: every line is JSON; second lines have video/second/objects/people with the right types;
object names are exactly the five from section 1, without duplicates; each video's seconds run
0..last with no gaps or duplicates; error lines have video/error. Prints a summary, exits 1 on problems.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys

NAMES = {"person", "car", "dog", "laptop", "cell phone"}


def check(path: str, expect_videos: int | None = None) -> list[str]:
    problems: list[str] = []
    seconds: dict[str, list[int]] = collections.defaultdict(list)
    errors: dict[str, int] = collections.Counter()
    n = 0
    with open(path) as fh:
        for i, line in enumerate(fh, 1):
            if not line.strip():
                continue
            n += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                problems.append(f"line {i}: not JSON")
                continue
            video = rec.get("video")
            if not isinstance(video, str) or ":" not in video or not video.rsplit(":", 1)[1].isdigit():
                problems.append(f"line {i}: bad video field {video!r}")
            if "error" in rec:
                if set(rec) != {"video", "error"} or not isinstance(rec["error"], str):
                    problems.append(f"line {i}: bad error line {rec}")
                errors[video] += 1
                continue
            if set(rec) != {"video", "second", "objects", "people"}:
                problems.append(f"line {i}: keys {sorted(rec)}")
                continue
            if not isinstance(rec["second"], int) or rec["second"] < 0:
                problems.append(f"line {i}: bad second {rec['second']!r}")
            objs = rec["objects"]
            if not isinstance(objs, list) or len(set(objs)) != len(objs) or not set(objs) <= NAMES:
                problems.append(f"line {i}: bad objects {objs!r}")
            if not isinstance(rec["people"], int) or rec["people"] < 0:
                problems.append(f"line {i}: bad people {rec['people']!r}")
            seconds[video].append(rec["second"])
    for video, secs in seconds.items():
        if sorted(secs) != list(range(max(secs) + 1)):
            dup = [s for s, c in collections.Counter(secs).items() if c > 1]
            problems.append(f"{video}: seconds not 0..{max(secs)} exactly once (dups {dup[:5]})")
    for video, c in errors.items():
        if c > 1:
            problems.append(f"{video}: {c} error lines (expected at most one)")
    videos = set(seconds) | set(errors)
    if expect_videos is not None and len(videos) != expect_videos:
        problems.append(f"expected {expect_videos} videos, found {len(videos)}")
    print(f"{path}: {n} lines, {len(videos)} videos, {sum(len(s) for s in seconds.values())} second lines, "
          f"{len(errors)} videos with an error line, {len(problems)} problems")
    for p in problems[:20]:
        print("  PROBLEM", p)
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--expect-videos", type=int, default=None)
    a = ap.parse_args()
    return 1 if check(a.results, a.expect_videos) else 0


if __name__ == "__main__":
    sys.exit(main())
