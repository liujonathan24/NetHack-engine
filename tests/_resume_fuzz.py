"""Random-play checkpoint/resume fuzz (tests/test_resume_fuzz.py, tools/resume_fuzz.py).

A case is a role, a seed pair, a depth and how the hero got there (goto_depth
or the stairs).  The keys are a random mix of commands; an open prompt is
answered by the same stream.  One process plays the case: it checkpoints at a
random command boundary, plays the tail (the burst of tests/_resume_worker.py
first) and replays the tail from an in-memory snapshot taken at the same
point.  A second process (tests/_resume_worker.py resume) resumes the
checkpoint and replays the tail.  Every field of every step must match in
both replays.

    python tests/_resume_fuzz.py play <outdir>      play the case in <outdir>/spec.json
    python tests/_resume_fuzz.py features <outdir>  what the checkpoint's level holds
"""
import json
import os
import pathlib
import random
import subprocess
import sys
import time

import numpy as np

from nethack_core import glyphs
from nethack_core.engine_env import CheckpointAtPromptError, EngineEnv
from tests import _resume_worker as W

ROLES = {"Tourist": "Tou-hum-neu-fem", "Monk": "Mon-hum-neu-mal", "Wizard": "Wiz-elf-cha-mal",
         "Priest": "Pri-hum-neu-fem", "Valkyrie": "Val-hum-neu-fem", "Samurai": "Sam-hum-law-mal",
         "Healer": "Hea-gno-neu-fem", "Archeologist": "Arc-dwa-law-mal", "Barbarian": "Bar-orc-cha-fem",
         "Rogue": "Rog-hum-cha-mal", "Ranger": "Ran-elf-cha-fem", "Knight": "Kni-hum-law-mal",
         "Caveman": "Cav-dwa-neu-fem"}
TUNES = ({"reveal_map": 1.0, "dmg_by_player_scale": 2.0}, {"reveal_map": 1.0, "hp_regen_scale": 3.0})
DIRS = "hjklyubn"
ESC, CR, SPACE = 27, 13, 32
PM = {name: i for i, name in enumerate(glyphs.MONSTER_NAMES)}
ROOT = pathlib.Path(__file__).resolve().parent.parent


def make_case(i, seed, roles=None, depths=None, tune=False, unset=False):
    """Case i of a run: role and depth cycle, the rest comes from the run's seed."""
    r = random.Random(seed * 1000003 + i)
    roles, depths = list(roles or ROLES), list(depths or (1, 2, 3, 4))
    role = roles[i % len(roles)]
    depth = depths[(i // len(roles)) % len(depths)]
    entry = "stairs" if depth > 1 and r.random() < 0.4 else "goto"
    return {"id": f"{role.lower()}-d{depth}{entry[0]}-{seed}.{i}", "role": role,
            "character": None if unset else ROLES[role],
            "seeds": [r.randrange(1 << 31), r.randrange(1 << 31)], "depth": depth, "entry": entry,
            "tune": r.choice(TUNES) if tune else None, "ck_target": r.randint(60, 250), "after": 300,
            "case_seed": r.randrange(1 << 31)}


class Stream:
    """The random key stream: one command per move(), the answer to an open prompt per answer()."""

    def __init__(self, seed):
        self.r = random.Random(seed)

    def letter(self, inv):
        if inv and self.r.random() < 0.85:
            return ord(self.r.choice(list(inv)))
        return ord(self.r.choice("abcdefg-*?"))

    def answer(self, o):
        r, m = self.r, np.asarray(o.misc)
        if m[0]:
            return [ord(r.choice("yynnq")) if r.random() < 0.9 else ESC]
        if m[1]:
            return list(r.choice((b"\r", b"x\r", b"Elbereth\r", b"\x1b")))
        if r.random() < 0.7:
            return [r.choice((CR, SPACE, ESC))]
        return [ord(r.choice("abcdefgh")), CR]

    def move(self, o, inv):
        r = self.r
        d = lambda: ord(r.choice(DIRS))
        L = lambda: self.letter(inv)
        count = lambda: list(str(r.choice((2, 3, 5, 10, 20))).encode())
        moves = (
            (30, lambda: [d()]),
            (5, lambda: count() + [d()]),
            (8, lambda: [ord("s")]),
            (5, lambda: count() + [ord("s")]),
            (3, lambda: [ord(".")]),
            (3, lambda: [ord("F"), d()]),
            (3, lambda: [4, d()]),
            (3, lambda: [ord("t"), L(), d()]),
            (2, lambda: [ord("f"), d()]),
            (2, lambda: [ord("e"), L()]),
            (2, lambda: [ord("r"), L()]),
            (2, lambda: [ord("q"), L()]),
            (2, lambda: [ord("a"), L(), d()]),
            (2, lambda: [ord("E"), ord("-"), ord("x"), CR]),
            (1, lambda: [ord("E"), L()] + list(b"Elbereth\r")),
            (1, lambda: list(b"#pray\ry\r\r\r")),
            (1, lambda: list(b"#turn\r")),
            (1, lambda: list(b"#chat\r") + [d()]),
            (2, lambda: [ord(","), L(), CR]),
            (2, lambda: [ord("d"), L()]),
            (1, lambda: [ord("D"), ord("a"), CR, L(), CR]),
            (2, lambda: [ord(";")] + [d() for _ in range(r.randint(1, 4))] + [ord(".")]),
            (2, lambda: [ord("_"), ord(r.choice("<>_.")), ord(".")]),
            (1, lambda: [ord("_"), ord(">"), ord("."), ord(">")]),
            (1, lambda: [ord(r.choice("<>"))]),
            (1, lambda: [ord("Z"), ord(r.choice("ab")), d()]),
            (1, lambda: [ord("w"), L()]),
            (1, lambda: [ord("W"), L()]),
            (1, lambda: [ord("T"), L()]),
            (1, lambda: [ord(r.choice("oc")), d()]),
            (1, lambda: [ord("i"), ESC]),
            (1, lambda: [ord(":")]),
            (1, lambda: [24, ESC]),
            (3, lambda: [ESC]),
        )
        return r.choices([m for _, m in moves], [w for w, _ in moves])[0]()


def enter(env, spec):
    stairs = spec["entry"] == "stairs"
    modify = {"goto_depth": spec["depth"] - stairs} if spec["depth"] > 1 else {}
    o, _ = env.reset(seeds=tuple(spec["seeds"]), character=spec["character"], tune=spec["tune"], modify=modify)
    return o


def features(outdir):
    """What the checkpoint's level holds, in its own process: the straight keys are replayed to the
    checkpoint, then the reveal_map overlay shows every monster and feature of the level."""
    trace = json.loads((outdir / "trace.json").read_text())
    spec, ck = trace["spec"], trace["ckpts"]["ck"]["step"]
    env = EngineEnv()
    o = enter(env, spec)
    for t in trace["recs"][1:ck + 1]:
        o, _, _ = env.step(t["key"])
    assert W.digests(o) == trace["recs"][ck]["d"], "the straight replay differs at the checkpoint"
    env.tune.set(reveal_map=1.0)
    o, _, _ = env.step(18)
    g = np.asarray(o.glyphs)
    mons = {int(glyphs.glyph_to_mon(x)) for x in g[glyphs.glyph_is_normal_monster(g)]}
    ch = np.asarray(o.chars)
    dlvl0, dlvl = trace["recs"][0]["st"]["dlvl"], trace["recs"][ck]["st"]["dlvl"]
    tail = trace["recs"][ck + 1:]
    f = {"shop": PM["shopkeeper"] in mons, "temple": PM["aligned priest"] in mons,
         "fountain": bool((ch == ord("{")).any()), "altar": bool((ch == ord("_")).any()), "dlvl": dlvl,
         "stairs": spec["entry"] == "stairs" and dlvl0 == spec["depth"] - 1 and dlvl == spec["depth"],
         "death": trace["done_at"] is not None, "engraving": any("You read:" in t["msg"] for t in tail)}
    (outdir / "features.json").write_text(json.dumps(f))


def straight(env, spec, outdir, ck_target):
    """Play until `after` keys past the checkpoint (or death); the checkpoint is taken at the first
    command boundary at or after ck_target, after two ESCs that close any count or cursor prompt."""
    o = enter(env, spec)
    st = Stream(spec["case_seed"])
    trace = {"spec": spec, "ck_target": ck_target, "recs": [W.rec(o, 0, None)], "ckpts": {}, "done_at": None}
    queue = [ord(">")] if spec["entry"] == "stairs" else []
    step, ck, escaped, snap = 0, None, False, None
    while ck is None or step - ck < spec["after"]:
        if not queue:
            if any(o.misc):
                queue = st.answer(o) if ck is not None or step < ck_target else [ESC]
            elif ck is None and step >= ck_target and not escaped:
                queue, escaped = [ESC, ESC], True
            elif ck is None and escaped:
                try:
                    env.checkpoint(outdir / "ck.json")
                except CheckpointAtPromptError:
                    queue = [ESC]
                    continue
                snap, ck = env.snapshot(), step
                trace["ckpts"]["ck"] = {"step": step}
                queue = W.burst(W.inventory(o))
            else:
                queue = st.move(o, W.inventory(o))
        key = queue.pop(0)
        o, done, _ = env.step(key)
        step += 1
        trace["recs"].append(W.rec(o, step, key))
        if done:
            trace["done_at"] = step
            break
    return trace, snap


def replay(env, trace, ck):
    res = {"first_diff": None, "steps": 0}
    for t in trace["recs"][ck + 1:]:
        o, done, _ = env.step(t["key"])
        res["steps"] += 1
        bad = [f for f in W.FIELDS if W.digests(o)[f] != t["d"][f]]
        if bad and res["first_diff"] is None:
            res["first_diff"] = {"step": t["step"], "key": t["key"], "fields": bad, "straight": t["st"],
                                 "resumed": W.status(o), "straight_msg": t["msg"], "resumed_msg": W.msg(o)[:80]}
        if done:
            break
    return res


def play(spec, outdir):
    env = EngineEnv()
    r, target = random.Random(spec["case_seed"]), spec["ck_target"]
    for attempt in range(3):
        trace, snap = straight(env, spec, outdir, target)
        trace["retries"] = attempt
        if trace["ckpts"] or trace["done_at"] is None or trace["done_at"] <= 61:
            break
        target = r.randint(60, trace["done_at"] - 1)
    (outdir / "trace.json").write_text(json.dumps(trace))
    if trace["ckpts"]:
        env.restore(snap)
        trace["inmem"] = replay(env, trace, trace["ckpts"]["ck"]["step"])
        (outdir / "trace.json").write_text(json.dumps(trace))


def worker(script, *args, cwd):
    """One process; returns its stderr tail when it failed, else None."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")]))
    r = subprocess.run([sys.executable, str(script), *map(str, args)], cwd=cwd, env=env, capture_output=True,
                       text=True)
    return None if r.returncode == 0 else f"rc {r.returncode}: {r.stderr[-3000:]}"


def run_case(spec, outdir):
    """Play the case in one process and resume its checkpoint in another; return the comparison.

    status: exact (both replays exact), diverged, inmem_crash (the play process died in the
    in-memory replay; the cross-process result stands), died_before_ckpt, error."""
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "spec.json").write_text(json.dumps(spec))
    res, t0 = {"id": spec["id"], "spec": spec, "status": "error"}, time.time()
    res["error"] = worker(pathlib.Path(__file__), "play", outdir, cwd=outdir)
    if (outdir / "trace.json").exists():
        trace = json.loads((outdir / "trace.json").read_text())
        res.update(done_at=trace["done_at"], retries=trace["retries"])
        if not trace["ckpts"]:
            res["status"] = "died_before_ckpt"
        else:
            ck = res["ck_step"] = trace["ckpts"]["ck"]["step"]
            n_after = len(trace["recs"]) - 1 - ck
            res["inmem"] = trace.get("inmem") or {"crash": res["error"]}
            res["error"] = worker(pathlib.Path(W.__file__), "resume", "fuzz", outdir, "ck", cwd=outdir)
            if res["error"] is None:
                r = json.loads((outdir / "resume_ck.json").read_text())
                res.update(n_after=n_after, steps=r["steps"], after_resume_diff=r["after_resume_diff"],
                           first_diff=r["first_diff"], n_diff=r.get("n_diff", 0), resumed_done_at=r.get("done_at"))
                res["exact"] = r["first_diff"] is None and r["steps"] == n_after
                inmem = res["inmem"]
                res["inmem_exact"] = inmem.get("first_diff") is None and inmem.get("steps") == n_after
                res["status"] = ("exact" if res["exact"] and res["inmem_exact"] else
                                 "inmem_crash" if res["exact"] and "crash" in inmem else "diverged")
            if worker(pathlib.Path(__file__), "features", outdir, cwd=outdir) is None:
                res["features"] = json.loads((outdir / "features.json").read_text())
    res["time"] = round(time.time() - t0, 2)
    return res


def summarize(results, **meta):
    by = lambda key: {k: {"cases": sum(1 for x in results if x["spec"][key] == k),
                          "exact": sum(1 for x in results if x["spec"][key] == k and x["status"] == "exact")}
                      for k in sorted({x["spec"][key] for x in results}, key=str)}
    keys = ("id", "spec", "ck_step", "n_after", "steps", "done_at", "resumed_done_at", "after_resume_diff",
            "first_diff", "n_diff", "inmem", "features")
    pick = lambda status: [{k: x.get(k) for k in keys} for x in results if x["status"] == status]
    times = [x["time"] for x in results]
    return dict(meta, cases=len(results),
                status={s: sum(1 for x in results if x["status"] == s)
                        for s in ("exact", "diverged", "inmem_crash", "died_before_ckpt", "error")},
                by_role=by("role"), by_depth=by("depth"), by_entry=by("entry"),
                divergences=pick("diverged"), inmem_crashes=pick("inmem_crash"),
                errors=[{"id": x["id"], "error": x.get("error")} for x in results if x["status"] == "error"],
                time_per_case={"mean": round(sum(times) / max(len(times), 1), 2), "max": max(times, default=0)},
                results=results)


def _case(i, seed, expect=(), tune=False):
    return dict(make_case(i, seed, tune=tune), expect=list(expect))


DEFAULT_CASES = [_case(i, 7, expect=e) for i, e in ((119, ["shop"]), (34, ["stairs"]), (0, ["fountain"]), (40, []), (3, []), (17, []), (44, ["engraving"]), (45, []), (7, []), (22, ["death"]), (36, []), (50, []))] + [_case(64, 7, tune=True)]


if __name__ == "__main__":
    mode, out = sys.argv[1], pathlib.Path(sys.argv[2])
    play(json.loads((out / "spec.json").read_text()), out) if mode == "play" else features(out)
