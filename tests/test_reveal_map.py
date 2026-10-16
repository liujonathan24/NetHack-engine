"""reveal_map puts the whole level in the hero's memory: travel and farlook find
the down stairs, and the chars and tty_chars planes show the same map."""
import numpy as np
import pytest

from nethack_core.engine_env import EngineEnv


def _top(o):
    return bytes(np.asarray(o.tty_chars)[0].astype(np.uint8)).decode("latin-1").rstrip()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_chars_and_tty_agree(seed):
    env = EngineEnv()
    o, _ = env.reset(seeds=(seed, seed), character="Val-hum-neu-fem", tune={"reveal_map": 1.0})
    for key in b"sjjllkks":
        o, _, _ = env.step(key)
        assert np.array_equal(np.asarray(o.chars), np.asarray(o.tty_chars)[1:22, :79])


@pytest.mark.parametrize("seed", [303, 404])
def test_travel_and_farlook_find_the_down_stairs(seed):
    env = EngineEnv()
    o, _ = env.reset(seeds=(seed, seed), character="Val-hum-neu-fem", tune={"reveal_map": 1.0})
    for key in b";>.":
        o, _, _ = env.step(key)
    assert "staircase down" in _top(o)
    o, _, _ = env.step(27)
    start = tuple(np.asarray(o.blstats)[:2])
    for key in b"_>":
        o, _, _ = env.step(key)
    assert "Can't find" not in _top(o)
    o, _, _ = env.step(ord("."))
    assert tuple(np.asarray(o.blstats)[:2]) != start
