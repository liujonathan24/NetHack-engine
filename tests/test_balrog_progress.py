"""The real BALROG NetHack progression scorer (faithful to balrog-ai/BALROG).

Guards the vendored achievements table + the max-milestone logic against the
published anchor and the metric's defining properties.
"""
from nethack_harness.prompt.balrog import balrog_progress, _achievements


def test_published_anchor_dl10_xl6_is_12_56pct():
    # BALROG's own reported point: reaching Dlvl 10 (with low XL) = Dlvl:10 = 12.56%.
    assert round(100 * balrog_progress(10, 6), 2) == 12.56


def test_spawn_is_zero():
    assert balrog_progress(1, 1) == 0.0


def test_max_over_dlvl_and_xp_milestones():
    ach = _achievements()
    # A rollout at dlvl 2 / XL 1 scores the Dlvl:2 milestone (Xp:1 == 0).
    assert balrog_progress(2, 1) == ach["Dlvl:2"]
    # A rollout stuck at dlvl 1 but XL 3 scores the Xp:3 milestone.
    assert balrog_progress(1, 3) == ach["Xp:3"]
    # The metric is the MAX of the two axes.
    assert balrog_progress(2, 3) == max(ach["Dlvl:2"], ach["Xp:3"])


def test_monotone_nondecreasing_in_depth():
    xs = [balrog_progress(d, 1) for d in range(1, 51)]
    assert xs == sorted(xs)


def test_specials_and_clamp():
    assert balrog_progress(50, 30, ascended=True) == 1.0
    assert 0.0 <= balrog_progress(30, 20) <= 1.0
    # unknown-high levels fall back to nearest lower defined key, never crash
    assert balrog_progress(999, 999) <= 1.0
