"""One process of a cross-process checkpoint/resume test (tests/test_resume.py).

play   <scenario> <outdir>            play the scenario, write <outdir>/trace.json
                                      and a checkpoint per scheduled step
resume <scenario> <outdir> <ckname>   resume one checkpoint, replay the keys the
                                      straight game sent after it, write
                                      <outdir>/resume_<ckname>.json
"""
import hashlib
import json
import pathlib
import random
import sys

import numpy as np

from nethack_core.engine_env import EngineEnv

FIELDS = ("tty_chars", "tty_colors", "tty_cursor", "glyphs", "chars", "colors",
          "message", "inv_strs", "inv_letters", "inv_glyphs", "blstats", "misc")
BL = {"hp": 10, "dlvl": 12, "gold": 13, "ac": 16, "time": 20}
ESC = 27

# ck_from/ck_count: a checkpoint at each of the first ck_count prompt-free
# steps >= ck_from; after: keys sent after the last checkpoint (the same walk
# continued, or a fresh walk with seed tail, after the random burst if any).
SCENARIOS = {
    "valkyrie": dict(character="Val-hum-neu-fem", seeds=(42, 7), walk=2, ck_from=150, ck_count=1, after=150),
    "tourist": dict(character="Tou-hum-neu-fem", seeds=(7, 7), walk=5, ck_from=120, ck_count=1, after=260, burst=True),
    "monk_fast": dict(character=None, seeds=(42, 7), walk=2, ck_from=150, ck_count=8, after=100),
    "shop": dict(character="Val-hum-neu-fem", seeds=(42, 7), modify={"goto_depth": 2}, walk=2, ck_from=150, ck_count=1, after=270, shop_sound=True),
    "shop_b": dict(character="Val-hum-neu-fem", seeds=(42, 7), modify={"goto_depth": 2}, walk=2, ck_from=150, ck_count=1, after=270, tail=102),
    "fifty": dict(character=None, seeds=(11, 11), walk=4, ck_from=1, ck_count=50, after=30),
}
SHOP_SOUNDS = ("cursing shoplifters", "chime of a cash register", "Neiman and Marcus")


def walk(n, seed):
    r = random.Random(seed)
    return [ord(r.choice("hjklyubnhjklyubns.")) if r.random() > 0.05 else ESC for _ in range(n)]


def burst(inv):
    """Keys that draw on the random streams: throws, camera, kicks, searches, quaff, read, engrave, pray."""
    def letter(word):
        return next((l for l, s in inv.items() if word in s), None)
    dart, camera, potion, scroll = letter("dart"), letter("camera"), letter("potion"), letter("scroll")
    keys = []
    for d in "hjklyubn":
        if dart:
            keys += [ord("t"), ord(dart), ord(d)]
    for d in "hjkl":
        if camera:
            keys += [ord("a"), ord(camera), ord(d)]
        keys += [4, ord(d)]
    keys += [ord("2"), ord("0"), ord("s")]
    if potion:
        keys += [ord("q"), ord(potion)]
    if scroll:
        keys += [ord("r"), ord(scroll)]
    keys += [ord("E"), ord("-"), ord("x"), 13]
    keys += list(b"#pray\ry\r\r\r")
    return keys


def digests(o):
    return {f: hashlib.sha256(np.ascontiguousarray(getattr(o, f)).tobytes()).hexdigest()[:16] for f in FIELDS}


def msg(o):
    return bytes(np.asarray(o.message)).split(b"\0")[0].decode("latin-1")


def inventory(o):
    letters = bytes(np.asarray(o.inv_letters)).split(b"\0")[0].decode()
    strs = [bytes(s).split(b"\0")[0].decode("latin-1") for s in np.asarray(o.inv_strs)]
    return dict(zip(letters, strs))


def status(o):
    b = np.asarray(o.blstats)
    return {k: int(b[i]) for k, i in BL.items()}


def rec(o, step, key):
    return {"step": step, "key": key, "d": digests(o), "st": status(o), "msg": msg(o)[:80]}


def play(name, outdir):
    sc = SCENARIOS[name]
    env = EngineEnv()
    o, _ = env.reset(seeds=sc["seeds"], character=sc["character"], tune=sc.get("tune"), modify=sc.get("modify"))
    keys = walk(sc["ck_from"] + 60 + sc["after"], sc["walk"])
    trace = {"scenario": name, "recs": [rec(o, 0, None)], "ckpts": {}, "done_at": None}
    left, last_ck, step = sc["ck_count"], None, 0
    while step < len(keys):
        o, done, _ = env.step(keys[step])
        step += 1
        trace["recs"].append(rec(o, step, keys[step - 1]))
        if done:
            trace["done_at"] = step
            break
        if left and step >= sc["ck_from"] and not any(o.misc):
            env.checkpoint(outdir / f"ck{step}.json")
            trace["ckpts"][f"ck{step}"] = {"step": step}
            left -= 1
            last_ck = step
            if not left:
                tail = walk(sc["after"], sc["tail"]) if sc.get("tail") else keys[step:]
                keys = keys[:step] + (burst(inventory(o)) if sc.get("burst") else []) + tail
        if last_ck and not left and step >= last_ck + sc["after"]:
            break
    (outdir / "trace.json").write_text(json.dumps(trace))


def resume(name, outdir, ckname):
    trace = json.loads((outdir / "trace.json").read_text())
    ck_step = trace["ckpts"][ckname]["step"]
    env = EngineEnv()
    o = env.resume(outdir / f"{ckname}.json")
    at = trace["recs"][ck_step]
    res = {"ckname": ckname, "ck_step": ck_step, "character": getattr(env, "character", None),
           "after_resume_diff": [f for f in FIELDS if digests(o)[f] != at["d"][f]],
           "after_resume_status": status(o), "straight_status": at["st"],
           "first_diff": None, "steps": 0}
    for t in trace["recs"][ck_step + 1:]:
        o, done, _ = env.step(t["key"])
        res["steps"] += 1
        bad = [f for f in FIELDS if digests(o)[f] != t["d"][f]]
        if bad and res["first_diff"] is None:
            res["first_diff"] = {"step": t["step"], "fields": bad, "straight": t["st"], "resumed": status(o),
                                 "straight_msg": t["msg"], "resumed_msg": msg(o)[:80]}
        if done:
            break
    (outdir / f"resume_{ckname}.json").write_text(json.dumps(res))


if __name__ == "__main__":
    mode, name, outdir = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
    outdir.mkdir(parents=True, exist_ok=True)
    play(name, outdir) if mode == "play" else resume(name, outdir, sys.argv[4])
