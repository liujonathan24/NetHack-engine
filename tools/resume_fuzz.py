#!/usr/bin/env python
"""Random-play checkpoint/resume fuzz over any number of cases (tests/_resume_fuzz.py).

    tools/resume_fuzz.py --cases 200 --seed 1 --jobs 8 --out report.json
        [--workdir DIR] [--roles Tourist,Monk] [--depths 1,2] [--tune] [--unset-character]

The report lists every divergence (case, first differing step and key, fields,
both status dicts), the counts per role/depth/entry and the time per case.
"""
import argparse
import concurrent.futures
import json
import pathlib
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from tests import _resume_fuzz as F  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default="resume_fuzz.json")
    ap.add_argument("--workdir", help="per-case traces and checkpoints (default: a temp dir)")
    ap.add_argument("--roles", help="comma list, default: every role in tests/_resume_fuzz.py")
    ap.add_argument("--depths", help="comma list, default 1,2,3,4")
    ap.add_argument("--tune", action="store_true", help="reveal_map 1.0 plus a damage or regen knob")
    ap.add_argument("--unset-character", action="store_true", help="the engine's default role")
    a = ap.parse_args()
    roles = a.roles.split(",") if a.roles else None
    depths = [int(d) for d in a.depths.split(",")] if a.depths else None
    cases = [F.make_case(i, a.seed, roles, depths, a.tune, a.unset_character) for i in range(a.cases)]
    workdir = pathlib.Path(a.workdir or tempfile.mkdtemp(prefix="resume_fuzz_"))
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(a.jobs) as ex:
        results = list(ex.map(lambda c: F.run_case(c, workdir / c["id"]), cases))
    report = F.summarize(results, seed=a.seed, tune=a.tune, unset_character=a.unset_character,
                         workdir=str(workdir), elapsed=round(time.time() - t0, 1))
    pathlib.Path(a.out).write_text(json.dumps(report, indent=1))
    print(f"cases {report['cases']} status {report['status']} time/case {report['time_per_case']} "
          f"elapsed {report['elapsed']}s -> {a.out}")
    for d in report["divergences"]:
        print(f"  {d['id']}: ck {d['ck_step']} first diff {d['first_diff']} in-memory {d['inmem']}")
    for d in report["inmem_crashes"]:
        print(f"  {d['id']}: ck {d['ck_step']} game over at {d['done_at']}, the in-memory replay crashed")
    for e in report["errors"]:
        print(f"  {e['id']}: error {e['error'][-300:]!r}")
    return 1 if report["cases"] != report["status"]["exact"] else 0


if __name__ == "__main__":
    sys.exit(main())
