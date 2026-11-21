from alethia.rules import (
    FrameDet,
    adaptive_decision,
    exact_second,
    median_floor,
    sample_indices,
)

PERSON, CAR, DOG, LAPTOP, PHONE = 1, 2, 4, 8, 16


def F(people=0, mask=0, near=False):
    return FrameDet(people, mask, near)


# ---------------------------------------------------------------- median, rounded down at .5
def test_median_odd():
    assert median_floor([3, 1, 2]) == 2


def test_median_even_true_median_not_lower_element():
    assert median_floor([2, 4]) == 3  # true median 3, not the lower middle value 2


def test_median_even_halfway_rounds_down():
    assert median_floor([2, 3]) == 2
    assert median_floor([0, 1]) == 0
    assert median_floor([1, 2, 5, 9]) == 3  # (2 + 5) / 2 = 3.5 -> 3


def test_median_empty():
    assert median_floor([]) == 0


# ---------------------------------------------------------------- presence: at least half of the frames
def test_exactly_half_is_present():
    frames = [F(1, PERSON)] * 30 + [F()] * 30
    assert exact_second(frames) == (["person"], 0)  # people median of 30 ones and 30 zeros = 0.5 -> 0


def test_below_half_is_absent():
    frames = [F(0, CAR)] * 13 + [F()] * 14  # 13 of 27 < 13.5
    assert exact_second(frames)[0] == []
    frames = [F(0, CAR)] * 14 + [F()] * 13  # 14 of 27
    assert exact_second(frames)[0] == ["car"]


def test_object_order_matches_brief():
    frames = [F(2, PERSON | LAPTOP | PHONE | DOG | CAR)]
    assert exact_second(frames) == (["person", "car", "dog", "laptop", "cell phone"], 2)


def test_single_frame_and_empty_second():
    assert exact_second([F(3, PERSON)]) == (["person"], 3)
    assert exact_second([]) == ([], 0)


# ---------------------------------------------------------------- sampling positions
def test_sample_indices_spread_and_short_seconds():
    assert sample_indices(30, 5) == [3, 9, 15, 21, 27]
    assert sample_indices(24, 5) == [2, 7, 12, 16, 21]
    assert sample_indices(4, 5) == [0, 1, 2, 3]  # K or fewer frames: every frame
    assert sample_indices(5, 5) == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------- adaptive rule, K = 5, 80/20
def test_clear_cut_present_and_absent():
    s = [F(1, PERSON)] * 4 + [F(1, 0)]  # person boxes: 4/5 present by mask, counts all 1
    assert adaptive_decision(s) == (["person"], 1)
    s = [F(0, CAR)] + [F()] * 4  # car 1/5 -> absent
    assert adaptive_decision(s) == ([], 0)


def test_borderline_fraction_escalates():
    s = [F(0, DOG)] * 2 + [F()] * 3  # 2/5 = 0.4
    assert adaptive_decision(s) is None
    s = [F(0, DOG)] * 3 + [F()] * 2  # 3/5 = 0.6
    assert adaptive_decision(s) is None


def test_unequal_people_counts_escalate():
    s = [F(2, PERSON)] * 4 + [F(3, PERSON)]
    assert adaptive_decision(s) is None


def test_near_cutoff_confidence_escalates_only_with_margin():
    s = [F()] * 4 + [F(near=True)]
    assert adaptive_decision(s, use_margin=True) is None
    assert adaptive_decision(s, use_margin=False) == ([], 0)


def test_all_agree_everything():
    s = [F(2, PERSON | LAPTOP)] * 5
    assert adaptive_decision(s) == (["person", "laptop"], 2)


# ---------------------------------------------------------------- majority people rule (option 3)
def test_majority_people_rule():
    s = [F(2, PERSON)] * 4 + [F(3, PERSON)]  # 4 of 5 agree on 2
    assert adaptive_decision(s, use_margin=False, people_rule="majority") == (["person"], 2)
    assert adaptive_decision(s, use_margin=False, people_rule="all-equal") is None
    s = [F(2, PERSON)] * 3 + [F(3, PERSON)] * 2  # only 3 of 5 agree: borderline
    assert adaptive_decision(s, use_margin=False, people_rule="majority") is None


def test_sweep_simulates_the_same_rule():
    """tools/sweep.py chose option 3; its simulated decision must equal the one run.py makes."""
    import itertools
    import random

    from tools.sweep import decide

    rng = random.Random(0)
    for _ in range(3000):
        k = rng.choice([3, 4, 5, 6, 8])
        samples = [F(rng.choice([0, 0, 1, 2, 3]), rng.randrange(32), rng.random() < 0.1) for _ in range(k)]
        for (hi, lo), margin, rule in itertools.product([(1.0, 0.0), (0.8, 0.2), (0.6, 0.4)], [True, False],
                                                       ["all-equal", "majority"]):
            ours = adaptive_decision(samples, hi, lo, margin, rule)
            sim, _ = decide(samples, hi, lo, margin, rule)
            assert ours == sim, (samples, hi, lo, margin, rule)
