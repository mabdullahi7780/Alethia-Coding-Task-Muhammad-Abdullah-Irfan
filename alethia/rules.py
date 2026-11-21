"""Per-second decision rules.

Each frame's detections are reduced to a FrameDet:
    people  number of person detections with confidence >= 0.4
    mask    bit i set if class NAMES[i] has a detection with confidence >= 0.4
    near    True if any of the five classes has a detection whose confidence sits in the
            "near the cutoff" band [margin_lo, margin_hi) (used only by the adaptive rule)

exact_second() is the brief's section 5 definition, applied to every frame of a second.
adaptive_decision() is the skip mode's early decision from K sampled frames.
"""

from __future__ import annotations

from collections import Counter
from typing import NamedTuple, Sequence

# Object names exactly as written in section 1 of the brief, with their COCO class ids.
NAMES = ("person", "car", "dog", "laptop", "cell phone")
CLASS_IDS = (0, 2, 16, 63, 67)
CLASS_TO_BIT = {cid: i for i, cid in enumerate(CLASS_IDS)}
CONF = 0.4


class FrameDet(NamedTuple):
    people: int
    mask: int
    near: bool = False


def median_floor(values: Sequence[int]) -> int:
    """Median of integer counts, rounded down when it falls halfway between two numbers."""
    if not values:
        return 0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) // 2  # non-negative ints: floor of the halfway value


def objects_from_mask(mask: int) -> list[str]:
    return [name for i, name in enumerate(NAMES) if mask >> i & 1]


def exact_second(frames: Sequence[FrameDet]) -> tuple[list[str], int]:
    """Section 5: present if found in at least half of the second's frames; people = median, floor at .5."""
    n = len(frames)
    if n == 0:
        return [], 0
    mask = 0
    for i in range(len(NAMES)):
        hits = sum(f.mask >> i & 1 for f in frames)
        if 2 * hits >= n:
            mask |= 1 << i
    return objects_from_mask(mask), median_floor([f.people for f in frames])


def sample_indices(n: int, k: int) -> list[int]:
    """K evenly spaced frame positions inside a second of n frames (centred in each stride)."""
    if n <= k:
        return list(range(n))
    return [int((i + 0.5) * n / k) for i in range(k)]


def adaptive_decision(
    samples: Sequence[FrameDet], hi: float = 0.8, lo: float = 0.2, use_margin: bool = True,
    people_rule: str = "all-equal",
) -> tuple[list[str], int] | None:
    """Decide a second from its sampled frames, or return None when it is borderline.

    Decided only if every class is clear-cut (seen in >= hi of samples, or in <= lo of them),
    the people count is clear-cut, and (with use_margin) no sampled detection is near the 0.4
    cutoff. Otherwise the caller runs the model on the rest of the second's frames.

    people_rule "all-equal": every sample must have the same people count.
    people_rule "majority":  the most common count must appear in at least `hi` of the samples
                             (K=5, hi=0.8: in 4 or 5 of them); that count is the answer.
    """
    k = len(samples)
    if k == 0:
        return [], 0
    if use_margin and any(f.near for f in samples):
        return None
    counts = Counter(f.people for f in samples)
    people, top = counts.most_common(1)[0]
    if people_rule == "all-equal":
        if top != k:
            return None
    elif people_rule == "majority":
        if top / k < hi:
            return None
    else:
        raise ValueError(f"unknown people_rule {people_rule!r}")
    mask = 0
    for i in range(len(NAMES)):
        frac = sum(f.mask >> i & 1 for f in samples) / k
        if frac >= hi:
            mask |= 1 << i
        elif frac > lo:
            return None
    return objects_from_mask(mask), people
