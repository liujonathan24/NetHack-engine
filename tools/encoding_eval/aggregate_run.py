"""Aggregate an encoding-sweep run into the Experiment-1 results table.

Per cell (encoding), over its 16 rollouts:
  - Depth Score  = mean +/- SE of max dungeon level reached (from the per-turn
    traces, which carry the authoritative `max_dlvl_reached` game-state field).
  - Alive@cap %  = fraction of rollouts that hit the 150-turn cap still playing
    (results.jsonl `is_truncated`), i.e. "could have gone deeper".
  - Death %      = fraction whose final hitpoints == 0.
  - Tokens/turn  = mean input & output tokens per LM turn (results `token_usage`).
  - move-attempts= count of `move(...)` tool calls the model emitted AND the
    `not available` rejections (from the `completion` history) — the withheld
    primitive it reached for; must be executed=0 (gate holds).
  - Tool mix     = mean calls of the main skills.

Depth per rollout comes from traces; everything else from results.jsonl. Both
are 16-rollout aggregates, so we don't need to join individual rollouts.

Usage: PYTHONPATH=.:environments/nethack .venv/bin/python \
         tools/encoding_eval/aggregate_run.py outputs/encoding_eval/run2
"""
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "environments", "nethack"))
from nethack_harness.prompt.balrog import balrog_progress  # noqa: E402

CELLS = ["B0", "JSON", "TOON", "IMG", "IMG_TTY"]
REPR = {
    "B0": "uncompressed ASCII", "JSON": "structured object",
    "TOON": "compact structured", "IMG": "pixel tileset (vision)",
    "IMG_TTY": "tty-raster (vision)",
}


def _trace_rollouts(cell_dir):
    """Per rollout: (max_dlvl, died, max_xp_level)."""
    out = []
    for f in sorted(glob.glob(os.path.join(cell_dir, "trace", "*.ndjson"))):
        md, died, mx = 1, False, 1
        for line in open(f):
            try:
                t = json.loads(line)
            except ValueError:
                continue
            md = max(md, t.get("max_dlvl_reached") or t.get("dlvl") or 1)
            st = t.get("status")
            if isinstance(st, dict):
                if st.get("hitpoints") == 0:
                    died = True
                mx = max(mx, st.get("experience_level") or 1)
        out.append((md, died, mx))
    return out


def _results_rows(cell_dir):
    fs = glob.glob(os.path.join(cell_dir, "evals", "*", "*", "results.jsonl"))
    if not fs:
        fs = glob.glob(os.path.join(cell_dir, "**", "results.jsonl"), recursive=True)
    if not fs:
        return []
    return [json.loads(l) for l in open(sorted(fs)[0])]


def _move_stats(rows):
    """Count move() tool calls the model emitted and 'not available' rejections
    across the completion message histories."""
    attempts = rejections = 0
    for r in rows:
        comp = r.get("completion") or []
        for m in comp:
            if not isinstance(m, dict):
                continue
            for tc in (m.get("tool_calls") or []):
                # tool_calls entries are JSON strings: {"id","name","arguments"}.
                if isinstance(tc, str):
                    try:
                        tc = json.loads(tc)
                    except ValueError:
                        continue
                name = tc.get("name") or (tc.get("function") or {}).get("name")
                if name == "move":
                    attempts += 1
            c = m.get("content")
            if isinstance(c, str) and "is not available" in c and "'move'" in c:
                rejections += 1
    return attempts, rejections


def _mean_se(xs):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    mu = sum(xs) / n
    if n == 1:
        return mu, 0.0
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return mu, math.sqrt(var / n)


def aggregate(run_dir):
    table = []
    for cell in CELLS:
        cd = os.path.join(run_dir, cell)
        if not os.path.isdir(cd):
            continue
        rollouts = _trace_rollouts(cd)
        depths = [d for d, _, _ in rollouts]
        rows = _results_rows(cd)
        if not depths and not rows:
            continue
        mu, se = _mean_se(depths)
        n = len(depths)
        died = sum(1 for _, d, _ in rollouts if d)
        # Real BALROG progression (%) per rollout = max milestone over (Dlvl, Xp).
        balrog = [100 * balrog_progress(md, mx) for md, _, mx in rollouts]
        balrog_mu, balrog_se = _mean_se(balrog)
        alive_cap = sum(1 for r in rows if r.get("is_truncated")) / len(rows) if rows else float("nan")
        in_tok = [r["token_usage"]["input_tokens"] / max(1, r.get("num_turns", 1)) for r in rows if r.get("token_usage")]
        out_tok = [r["token_usage"]["output_tokens"] / max(1, r.get("num_turns", 1)) for r in rows if r.get("token_usage")]
        m_attempts, m_reject = _move_stats(rows)
        def _toolmean(k):
            vs = [r.get(k, 0) for r in rows]
            return sum(vs) / len(vs) if vs else 0.0
        table.append({
            "cell": cell, "repr": REPR.get(cell, cell), "n": n,
            "depth_mean": mu, "depth_se": se,
            "balrog_pct_mean": balrog_mu, "balrog_pct_se": balrog_se,
            "balrog_pct_max": max(balrog) if balrog else None,
            "depth_hist": {d: depths.count(d) for d in sorted(set(depths))},
            "alive_at_cap_pct": 100 * alive_cap if alive_cap == alive_cap else None,
            "death_pct": 100 * died / n if n else None,
            "in_tok_per_turn": _mean_se(in_tok)[0] if in_tok else None,
            "out_tok_per_turn": _mean_se(out_tok)[0] if out_tok else None,
            "move_attempts": m_attempts, "move_rejections": m_reject,
            "move_to_calls": _toolmean("move_to_calls"),
            "explore_and_descend_calls": _toolmean("explore_and_descend_calls"),
            "attack_calls": _toolmean("attack_calls"),
            "search_calls": _toolmean("search_calls"),
            "wiki_calls": _toolmean("wiki_lookup_calls") + _toolmean("wiki_search_calls"),
            "scout_reward": _toolmean("scout_reward"),
            "descent_reward": _toolmean("descent_reward"),
        })
    return table


def to_markdown(table):
    lines = []
    lines.append("| Encoding | Repr | n | Depth Score (mean ± SE) | BALROG % (mean ± SE / max) | Depth dist | Alive@150 | Death % | in tok/turn | out tok/turn | move exec / attempt / reject |")
    lines.append("|---|---|:-:|:-:|:-:|---|:-:|:-:|:-:|:-:|:-:|")
    for r in sorted(table, key=lambda x: -(x["depth_mean"] if x["depth_mean"] == x["depth_mean"] else 0)):
        dist = " ".join(f"{k}:{v}" for k, v in r["depth_hist"].items())
        lines.append(
            f"| {r['cell']} | {r['repr']} | {r['n']} | "
            f"{r['depth_mean']:.2f} ± {r['depth_se']:.2f} | "
            f"{r['balrog_pct_mean']:.2f} ± {r['balrog_pct_se']:.2f} / {r['balrog_pct_max']:.2f} | {dist} | "
            f"{r['alive_at_cap_pct']:.0f}% | {r['death_pct']:.0f}% | "
            f"{r['in_tok_per_turn']:.0f} | {r['out_tok_per_turn']:.1f} | "
            f"0 / {r['move_attempts']} / {r['move_rejections']} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/encoding_eval/run2"
    table = aggregate(run_dir)
    md = to_markdown(table)
    print(md)
    out_json = os.path.join(run_dir, "table.json")
    out_md = os.path.join(run_dir, "table.md")
    json.dump(table, open(out_json, "w"), indent=1)
    open(out_md, "w").write(md + "\n")
    print(f"\nwrote {out_md} and {out_json}")
