"""reset(modify={"goto_depth": n}) seats the hero on the downstair and the first
observation already shows him there, as a redraw would."""
import numpy as np
import pytest

from nethack_core.engine_env import EngineEnv


@pytest.mark.parametrize("character,seeds", [("Val-hum-neu-fem", (42, 7)), ("Wiz-elf-cha-mal", (23, 7)),
                                             ("Sam-hum-law-mal", (5, 9))])
@pytest.mark.parametrize("depth", [2, 3, 4])
def test_goto_depth_draws_the_hero_on_the_stair(character, seeds, depth):
    env = EngineEnv()
    o, _ = env.reset(seeds=seeds, character=character, modify={"goto_depth": depth})
    x, y = (int(v) for v in np.asarray(o.blstats)[:2])
    assert chr(np.asarray(o.tty_chars).reshape(24, 80)[y + 1, x]) == "@"
    o2, _, _ = env.step(18)
    for f in ("glyphs", "chars", "colors", "blstats"):
        assert np.array_equal(np.asarray(getattr(o, f)), np.asarray(getattr(o2, f))), f
