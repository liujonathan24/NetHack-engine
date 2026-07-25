# Experiment 1 — Observation-encoding sweep (protocol)

**Status:** approved design, pre-launch. Author: Jonathan Liu. Date: 2026-07-24.
**Deliverable this spec defines:** the headline encoding-ablation table from
`docs/RESULTS_COLLATION_V3.md` §2, run rigorously on one fixed model.

---

## 1 · Question and hypotheses

Holding NetPlay's action surface fixed, **how deep does the agent get as a function of the
observation encoding alone?**

- **H1 (established, to reconfirm cleanly):** text > vision. Rendered pixels (IMG) and the
  tty raster (IMG_TTY) underperform the text encodings.
- **H2 (the one that failed before, to interrogate):** a structured object (JSON) should be
  at least as good as raw ASCII, because it is easier to parse. Prior runs measured JSON
  *below* uncompressed ASCII. This experiment re-tests it on one model with pinned games;
  the follow-on JSON-cell-content ablation (separate spec) decides whether the miss is JSON
  itself or an impoverished JSON schema.
- **H3 (control):** compression is not free on capability — but we do **not** run compacted
  ASCII (B1) here at all, per decision; B1 is excluded.

Primary axis: **Depth Score = `max_dlvl_reached`** (uncapped — see §3). Secondary:
alive-at-cap, unique-tile coverage, tokens/turn. BALROG progression is **not** reported
(the scorer is an unvalidated proxy — see `RESULTS_COLLATION_V3.md` §1).

---

## 2 · Fixed factors (identical in every cell)

| factor | value | rationale |
|---|---|---|
| model | `google/gemini-3-flash-preview` via Prime (`-p prime`) | a BALROG-bench model; Flash for experimentation, **upgrade to Gemini 3 Pro for final numbers** |
| action surface | `skill_set="netplay"` | NetPlay's high-level skills, no low-level `move(direction)`; navigation via `move_to` / `explore_and_descend` |
| tier | `full_nle` | **uncapped** — episode ends on death or the turn budget; no dlvl-6 success-milestone censoring |
| character | `Val-hum-neu-fem` (pinned) | strips character-RNG out of the encoding signal |
| max_turns | 150 | |
| n | 16 | per cell |
| seeds | 16 **pinned** explicit seeds, identical across all cells | makes this a *paired* comparison — encoding is the only thing that differs on a given game |
| compaction | `compact_obs=false` everywhere | no map compaction, per decision |

**Model note for final numbers.** Prime lists `google/gemini-3.1-pro-preview`, not a plain
`gemini-3-pro-preview`. Resolve the exact Gemini 3 Pro ID (Prime adds it, or route Pro
through the direct Google OpenAI-compat endpoint) before the final-number run.

---

## 3 · Cells (the encoding axis)

Five cells, n=16 each = 80 rollouts.

| cell | `variant` | representation |
|---|---|---|
| Uncompressed ASCII | `B0` | raw 21×79 grid |
| JSON | `JSON` | current single-schema structured map (baseline for the cell-content ablation) |
| TOON | `TOON` | compact structured text |
| IMG | `IMG` | rendered pixel tileset (vision) |
| IMG_TTY | `IMG_TTY` | rendered tty raster (vision) |

Excluded by decision: **B1** (compacted ASCII).

---

## 4 · Metrics logged per rollout

Logged for every rollout so the derived columns fall out of one run rather than needing
separate experiments:

- `max_dlvl_reached` (Depth Score), `xp_level` at end
- `alive_at_cap` (bool) + `num_turns` + death cause — answers "could it have gone deeper?"
- unique tiles seen (map-coverage proxy)
- observation tokens/turn and output tokens/turn (fills the token-cost column)
- per-rubric rewards (scout / descent / success / ascension)

The `alive_at_cap` flag is new — prior runs only logged death%. It must be written per
rollout (implementation task T3).

---

## 5 · Procedure (the babysat fan-out)

1. **Wire the runner.** `tools/encoding_eval/run.py`'s real runner is a `NotImplementedError`
   stub. Wire a per-cell launcher that shells to `vf-eval … -p prime` with the §2 `-a` args
   and `variant=<cell>`, `--save-results`, `trace_dir` under `outputs/`.
2. **Command validation (one cell, by hand).** Run a single B0 smoke (n=1, ~20 turns) to
   confirm the command shape, Prime routing to `google/gemini-3-flash-preview`, explicit-seed
   pinning, and Valkyrie pinning **before** any fan-out. A broken command must not fail five
   subagents identically.
3. **Smoke, one subagent per encoding** (n=1, ~20 turns). Each smoke subagent **verifies the
   trace actually rendered its intended encoding** — IMG shows pixels, JSON carries the cell
   attributes, seeds are pinned, the map isn't empty — and reports pass/fail. This is the gate
   that caught silently-wasted runs before.
4. **Fan-out** (gated on all smokes passing + a go from JL). n=16 per cell, one babysitting
   subagent per cell reading traces as they land: watch for stalls, Prime 429/402, degenerate
   loops, and empty-map renders. Cap concurrency to respect Prime rate limits.
5. **Aggregate.** `tools/encoding_eval/aggregate.py` → `outputs/encoding_eval/<run>/table.md`
   + `table.json`. Report Depth (mean ± SE), alive-at-cap %, tokens/turn per cell.

### Analysis

Because seeds **and** role are pinned, encodings are compared **paired** on identical games:
per-seed Depth deltas, a paired test (Wilcoxon signed-rank across the 16 seeds) for each
encoding-vs-B0 contrast, and the per-cell Depth distribution (count at each dlvl). Report the
paired deltas, not just the marginal means — that is the power win the pinning buys.

---

## 6 · Known implementation tasks (before fan-out)

- **T1 — runner wiring** (`tools/encoding_eval/run.py`): real per-cell `vf-eval -p prime` runner.
- **T2 — character pinning in `full_nle`**: confirm `character="Val-hum-neu-fem"` threads from
  `load_environment` through to the core-env `reset(character=…)` (`nethack_core/env.py:218`);
  `bootstrap_character` is a stub and full_nle "uses its own defaults," so this may need a
  small kwarg plumb.
- **T3 — per-rollout metric logging**: emit `alive_at_cap`, coverage, and tokens/turn into the
  saved results so §4 is populated without a second pass.
- **T4 — explicit-seed pinning**: `explicit_seeds=[…16…]` is supported (`nethack.py:1266`);
  confirm it survives the vf-eval `-a` round-trip (the nested-`task` round-trip bug bit us
  before — `RESULTS_COLLATION_V3.md` §6).

---

## 7 · Follow-on sub-experiments (separate specs, own build→smoke→run)

Each is its own protocol doc and PR; none blocks the base sweep.

- **1b · JSON cell-content ablation.** Parametric JSON schema; ablate one attribute at a time:
  glyph+knowledge, seen/unseen, lit/dark, visited/unvisited, reachability. Decides H2.
- **1c · Memory / journalling.** Journal vs belief-state vs summarize-and-reset (variant R) vs
  none, head to head — never isolated before.
- **1d · Observation timing + on-demand encoding.** (a) delayed map (action feedback every
  turn, full map re-sent only on request / material change) × {ASCII, JSON}; (b) a **new
  programmatic encoding**: JSON-like, but the map is exposed only when the agent calls a
  query — including a **bounding-box query** `reveal(x1,y1,x2,y2)` returning that sub-region.
  This is genuinely new code, so 1d gets a design pass before any eval.

---

## 8 · Repo / merge plan

- Runner wiring, this spec + the follow-on specs, and aggregation code → feature branch →
  **PR into clean `main`** (real harness capability).
- Bulky result artifacts (traces, `.jsonl`, rendered images) → `outputs/encoding_eval/` (kept
  out of clean `main`), per the main-clean / experimental split.
