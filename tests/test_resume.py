"""Cross-process checkpoint/resume on the fork engine.

A checkpoint written in one process and resumed in another must give the
same observation, field for field, for every key the straight game sent
after it. tests/_resume_worker.py is the one-process half of each test.
"""
from __future__ import annotations

import base64
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from nethack_core.engine_env import CheckpointAtPromptError, EngineEnv
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


def test_fast_monk_eight_checkpoints(tmp_path):
    trace, results = two_process("monk_fast", tmp_path)
    assert len(results) == 8
    for res in results.values():
        assert_exact(res, 100)


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
    blob = base64.b64decode(d["player"])  # cut the per-game tail too
    old["player"] = base64.b64encode(blob[:blob.rindex(b"NLEt")]).decode()
    (tmp_path / "old.json").write_text(json.dumps(old))
    env2 = EngineEnv()
    o2 = env2.resume(tmp_path / "old.json")
    assert getattr(env2, "character", None) is None
    for k in ("hp", "dlvl", "gold", "time"):  # not ac: the default-character reset is the old behaviour
        assert W.status(o2)[k] == W.status(o)[k], k
    o3, done, _ = env2.step(ord("s"))
    assert not done and o3.blstats[20] == o.blstats[20] + 1


def boundary_env(character="Val-hum-neu-fem", keys=b"ssss"):
    env = EngineEnv()
    env.reset(seeds=(42, 7), character=character)
    o, _, _ = env.step(27)
    for k in keys:
        o, _, _ = env.step(k)
    return env, o


@pytest.mark.parametrize("keys", [b"2", b"20", b"F", b"m", b"_", b";", b"#", b"#pra"],
                         ids=["count", "count2", "F", "m", "travel", "farlook", "extcmd", "extcmd_typed"])
def test_checkpoint_refused_inside_a_command(keys, tmp_path):
    """A key read that sets no prompt flag is not a command boundary either."""
    env, o = boundary_env()
    env.checkpoint(tmp_path / "a.json")
    for k in keys:
        o, _, _ = env.step(k)
    assert not any(o.misc)
    with pytest.raises(CheckpointAtPromptError, match="command is being read"):
        env.checkpoint(tmp_path / "b.json")
    for _ in range(3):
        o, _, _ = env.step(27)
        if env.engine.at_command_boundary:
            break
    env.checkpoint(tmp_path / "b.json")
    assert EngineEnv().resume(tmp_path / "b.json").blstats[20] == o.blstats[20]


def test_restore_after_the_game_ends():
    """The window port outlives the game: a snapshot restored after the game
    ended replays to the same end, field for field."""
    env, o = boundary_env(keys=b"ssss")
    snap = env.snapshot()
    keys = list(b"ss") + list(b"#quit\ry") + [13] * 8
    frames, ends = [], []
    for attempt in range(2):
        for k in keys:
            o, done, _ = env.step(k)
            frames.append((attempt, k, W.digests(o)))
            if done:
                ends.append(len([f for f in frames if f[0] == attempt]))
                break
        assert done, "the game did not end"
        env.restore(snap)
    first, second = [f[2] for f in frames if f[0] == 0], [f[2] for f in frames if f[0] == 1]
    assert ends[0] == ends[1] and first == second


def test_redo_after_resume(tmp_path):
    env, o = boundary_env(keys=b"ss3s")
    env.checkpoint(tmp_path / "ck.json")
    env2 = EngineEnv()
    env2.resume(tmp_path / "ck.json")
    o1, _, _ = env.step(1)  # ^A repeats "3s"
    o2, _, _ = env2.step(1)
    assert o1.blstats[20] == o2.blstats[20] > o.blstats[20]
    assert W.digests(o1) == W.digests(o2)


def test_truncated_player_blob_is_refused(tmp_path):
    env, o = boundary_env()
    env.checkpoint(tmp_path / "ck.json")
    d = json.loads((tmp_path / "ck.json").read_text())
    blob = base64.b64decode(d["player"])
    t = blob.rindex(b"NLEt")
    for name, cut, rc in (("magic", t + 6, 9), ("struct", t + 60, 9), ("string", len(blob) - 3, 9),
                          ("body", t - 500, 7)):
        d["player"] = base64.b64encode(blob[:cut]).decode()
        (tmp_path / f"{name}.json").write_text(json.dumps(d))
        with pytest.raises(RuntimeError, match=f"rc={rc}"):
            EngineEnv().resume(tmp_path / f"{name}.json")
    o2, _, _ = env.step(ord("s"))  # the process and the straight game go on
    assert o2.blstats[20] == o.blstats[20] + 1


def test_warning_glyphs_after_resume(tmp_path):
    """see_monsters() must run in the resumed process: a Wizard wearing a ring of
    warning on Dlvl 8 sees warning glyphs move every turn."""
    WARN = range(5589, 5595)  # GLYPH_WARNING_OFF .. + WARNCOUNT

    def warned(o):
        return bool(np.isin(np.asarray(o.glyphs), WARN).any())

    env = EngineEnv()
    env.reset(seeds=(23, 7), character="Wiz-hum-neu-mal", modify={"goto_depth": 8})
    o, _, _ = env.step(27)
    for k in [ord("P"), ord("e"), ord("r")]:  # the ring of warning, right hand
        o, _, _ = env.step(k)
    keys = W.walk(200, 2)
    step = 0
    while step < 100 or any(o.misc) or not env.engine.at_command_boundary:
        o, done, _ = env.step(keys[step])
        step += 1
        assert not done
    env.checkpoint(tmp_path / "ck.json")
    straight = [(W.digests(o := env.step(k)[0]), warned(o)) for k in keys[step:]]
    assert sum(w for _, w in straight) >= 50, "the straight run must show warning glyphs"
    env2 = EngineEnv()
    env2.resume(tmp_path / "ck.json")
    for i, k in enumerate(keys[step:]):
        o2, _, _ = env2.step(k)
        bad = [f for f in W.FIELDS if f != "inv_glyphs" and W.digests(o2)[f] != straight[i][0][f]]
        assert not bad, (i, bad)
