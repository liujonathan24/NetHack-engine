"""Cross-process checkpoint/resume on the fork engine.

A checkpoint written in one process and resumed in another must give the
same observation, field for field, for every key the straight game sent
after it. tests/_resume_worker.py is the one-process half of each test.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from nethack_core.engine_env import EngineEnv
from tests import _resume_worker as W

WORKER = pathlib.Path(W.__file__)


def worker(*args, cwd):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(WORKER.parent.parent), os.environ.get("PYTHONPATH", "")]))
    r = subprocess.run([sys.executable, str(WORKER), *map(str, args)], cwd=cwd, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]


def two_process(name, tmp_path, pick=lambda trace: list(trace["ckpts"])):
    """Play the scenario in one process, resume each picked checkpoint in its own; return (trace, {ckname: result})."""
    out = tmp_path / name
    out.mkdir()
    worker("play", name, out, cwd=tmp_path)
    trace = json.loads((out / "trace.json").read_text())
    assert trace["done_at"] is None, trace["done_at"]
    results = {}
    for ck in pick(trace):
        worker("resume", name, out, ck, cwd=tmp_path)
        results[ck] = json.loads((out / f"resume_{ck}.json").read_text())
    return trace, results


def assert_exact(res, min_steps):
    for f in ("blstats", "glyphs", "inv_strs"):
        assert f not in res["after_resume_diff"], (f, res["after_resume_status"], res["straight_status"])
    assert res["first_diff"] is None, res["first_diff"]
    assert res["steps"] >= min_steps


def row0(o):
    return bytes(np.asarray(o.tty_chars).reshape(24, 80)[0]).decode("latin-1")


@pytest.mark.parametrize("name", ["valkyrie", "tourist"])
def test_resume_is_exact(name, tmp_path):
    sc = W.SCENARIOS[name]
    trace, results = two_process(name, tmp_path)
    (res,) = results.values()
    assert res["ck_step"] >= sc["ck_from"]
    assert_exact(res, sc["after"])
    assert res["character"] == sc["character"]


@pytest.mark.xfail(strict=True, reason="NetHack side: the hero's movement points are not in the player blob yet")
def test_fast_monk_eight_checkpoints(tmp_path):
    trace, results = two_process("monk_fast", tmp_path)
    assert len(results) == 8
    for res in results.values():
        assert_exact(res, 100)


@pytest.mark.xfail(strict=True, reason="NetHack side: shopkeeper residency is not relinked on the loaded level yet")
@pytest.mark.parametrize("name", ["shop", "shop_b"])
def test_shop_level(name, tmp_path):
    trace, results = two_process(name, tmp_path)
    (res,) = results.values()
    assert res["straight_status"]["dlvl"] == 2
    if W.SCENARIOS[name].get("shop_sound"):
        sounds = [r["step"] for r in trace["recs"][res["ck_step"]:] if any(s in r["msg"] for s in W.SHOP_SOUNDS)]
        assert sounds, "the straight game must hear the shop after the checkpoint"
    assert_exact(res, 270)


def test_fifty_checkpoints_in_a_row(tmp_path):
    trace, results = two_process("fifty", tmp_path, pick=lambda t: [list(t["ckpts"])[-1]])
    assert len(trace["ckpts"]) == 50
    (res,) = results.values()
    assert_exact(res, 30)


def test_checkpoint_refused_at_prompt(tmp_path):
    from nethack_core import engine_env
    err = getattr(engine_env, "CheckpointAtPromptError", None)
    assert err is not None, "nethack_core.engine_env has no CheckpointAtPromptError"
    path = tmp_path / "ck.json"
    env = EngineEnv()
    env.reset(seeds=(42, 7))
    env.step(27)
    for k in b"#pray\r":
        o, _, _ = env.step(k)
    assert "[yn]" in row0(o) and o.misc[0] == 1
    with pytest.raises(err, match="y/n"):
        env.checkpoint(path)
    o, _, _ = env.step(ord("y"))
    assert "praying" in row0(o) and "--More--" in row0(o) and o.misc[2] == 1
    with pytest.raises(err, match="More"):
        env.checkpoint(path)
    o, _, _ = env.step(13)
    o, _, _ = env.step(13)
    assert not any(o.misc)
    env.checkpoint(path)
    assert EngineEnv().resume(path).blstats[20] == o.blstats[20]
    for k in list(b"#quit\ry") + [27] * 8:
        _, done, _ = env.step(k)
        if done:
            break
    assert env.done
    with pytest.raises(err, match="over"):
        env.checkpoint(path)


def test_old_format_checkpoint_still_resumes(tmp_path):
    env = EngineEnv()
    env.reset(seeds=(42, 7), character="Val-hum-neu-fem")
    for k in b"ssss":
        o, _, _ = env.step(k)
    env.checkpoint(tmp_path / "new.json")
    d = json.loads((tmp_path / "new.json").read_text())
    old = {k: d[k] for k in ("seed", "level", "player", "levelfiles")}
    (tmp_path / "old.json").write_text(json.dumps(old))
    env2 = EngineEnv()
    o2 = env2.resume(tmp_path / "old.json")
    assert getattr(env2, "character", None) is None
    for k in ("hp", "dlvl", "gold", "time"):  # not ac: the default-character reset is the old behaviour
        assert W.status(o2)[k] == W.status(o)[k], k
    o3, done, _ = env2.step(ord("s"))
    assert not done and o3.blstats[20] == o.blstats[20] + 1
