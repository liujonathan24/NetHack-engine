# Experiment 1 follow-ons — 1b / 1c / 1d (protocol)

**Status:** design, pre-build. Builds on the verified Exp-1 spine (`exp/encoding-sweep`):
Gemini 3 Flash via Prime (team-billed), `netplay` skills, uncapped `full_nle`, pinned
Valkyrie, seeds 0–15, n=16, no compaction, the `move`-gate + banner fix. Each sub-experiment
holds that spine fixed and varies ONE axis. Same verify-first discipline as Exp 1
(assumptions → build → smoke → subagent trace-verify → run → aggregate).

Grounded in three code explorations (2026-07-24). Touch-points below are from those.

**Decisions (2026-07-24):** (1) **Build + smoke all three now; run no n=16 sweeps yet** — the
expensive runs are staged for a later model/scale decision (likely the Gemini 3 Pro finals).
Every run auto-reports the **real BALROG %** (now wired into `aggregate_run.py`). (2) 1c uses a
**real in-process `sub_lm`** for belief-state (tests summary quality, not just a status snapshot)
**and** a **code flag for a truly zero-journal no-memory baseline**. (3) 1b ablates exactly
**+seen, +visited, +reachability** on top of the current identification-rich JSON **base**
("knowledge"/identification is already in the base `desc`/`species` fields, so it is not a
separate arm); **lit/dark deferred** (no data source).

---

## 1b · JSON cell-content ablation — *what belongs in a cell*

**Question.** JSON underperformed in Exp 1 (1.44, worst). Is that JSON itself, or an
impoverished schema? Ablate the per-cell/entity attribute set, one attribute at a time.

**Code reality.** JSON path: `_structured_map_template("json")` (`prompt_spec.py:171`) →
`build_map_model(raw_obs)` (`nethack_core/map_model.py:62`, **stateless**) →
`json_encode` (`map_encoders.py:36`). Today: per-entity dicts `{kind,x,y,desc(+species,
is_pet,obj_class,detail)}` + one RLE terrain `grid` string. `map_detail` (full/minimal)
reaches the env via the `**kwargs` seam — a new `cell_schema` config rides the same seam.

**Attribute tractability:**
| attribute | source | cost |
|---|---|---|
| glyph/kind + known info (species, is_pet, obj_class, door/stair detail) | already on `Entity` | **free** |
| seen vs unseen | `pathfinding.is_truly_unseen` / `is_unknown` | derivable in the v2 template closure (has `state`) |
| distance / reachability from hero | `pathfinding.reachable_set` / `a_star` + player pos | derivable in the closure |
| visited (stood-on) vs unvisited | **new** per-step player-pos set on `state` | small tracker |
| lit vs dark | **no data source** — `CoreObservation` can't distinguish; needs an NLE `specials` obs or colors inference | **hard — deferred v1** |

**Build (harness layer only, `nethack_core` untouched — Option ii from the explore):**
- `map_encoders.py`: config-gated per-cell field selection (extend the full/minimal split).
- `prompt_spec.py`: `_structured_map_template_v2(cell_schema)` closure that computes the
  derivable attrs from `state` (seen/visited/reachability) and merges into the encoder output;
  register `JSON_v2` (and reuse for the ablation cells).
- `nethack.py`: `cell_schema` constructor kwarg + `state["cell_schema"]` write near
  `map_detail` (`:438`); a per-step visited tracker (`state["_visited_tiles"]`, keyed by dlvl).
- Arms (each = base + one attribute toggled on): **base** (kind+coord+desc) → +knowledge →
  +seen/unseen → +visited → +reachability. lit/dark deferred.

---

## 1c · Memory / journalling ablation — *how much memory helps*

**Question.** Journal vs belief-state vs summarize-and-reset vs none — head to head.

**Code reality (mostly existing knobs — cheapest sub-experiment).** All settable via `-a`.
Arms (hold `variant`=fixed encoding, `interface="skill"`, spine fixed):
| arm | knobs |
|---|---|
| **journal-only** | `belief_state_interval=0`, `summarize_and_reset=False`, journal tools in `skill_set` |
| **+ belief-state** | above + `belief_state_interval=25` |
| **summarize-and-reset (R)** | above + `summarize_and_reset=True` |
| **no-memory** | `belief_state_interval=0`, journal tools **excluded** from `skill_set`, `history_keep_full` small |

**Two caveats (decisions needed):**
1. **belief-state fidelity.** Via CLI the belief note is a *deterministic status snapshot*
   (HP/AC/Dlvl/Turn), **not** an LM summary — real summarization needs a non-Offline `sub_lm`
   injected in-process (not reachable via `-a`) (`helpers.py:616`, `code_mode.py:378`). v1
   tests "summarization mechanism present," not "LM-summary quality." *(Fork A.)*
2. **no-memory cleanliness.** The objective line is always pinned (`nethack.py:471,482`), so
   `=== JOURNAL === Objective:` renders every turn even in the no-memory arm. It's identical
   across arms (not a confound), but a *truly* zero-journal baseline needs a small code flag
   to skip the pin. v1: accept the fixed objective line; note it. *(Fork B.)*

**Fixed encoding for 1c:** recommend **B0** (the Exp-1 winner) so the memory effect is
measured on the strongest observation.

---

## 1d · Observation timing + on-demand / bounding-box encoding

**Question.** (a) Does a **delayed map** (action feedback every turn; full map re-sent only on
material change / request) help — and does it help regardless of encoding? (b) A new
**on-demand encoding**: the map is exposed only when the agent calls `reveal(x1,y1,x2,y2)`.

**Code reality.** Full map is re-sent every turn at the sole map-producer
`turn_template` (`nethack.py:1061`); `format_observation_as_chat` already supports
`include_map=False` (`rendering.py:626,672`) — the precedent the image/structured templates
use. Action feedback rides `prefix_parts` every turn independent of the map (`nethack.py:1086`)
— so "feedback always, map sometimes" is purely a `turn_template` change. Info-only skills
(no NLE step) have two proven patterns: journal-op short-circuit (A, `nethack.py:668`) and
empty-actions pass-through (B, wiki tools, `skills.py:1835`).

**Build:**
- **(a) delayed-map = new delivery VARIANT** `DM`: `_delayed_map_template` includes the map
  only when a cheap map-fingerprint changed (dlvl change / `map_view` hash, mirroring the
  journal/inventory fingerprint pattern) or `state["_force_map"]` is set; else emits
  `=== MAP (unchanged) ===`. Optional `request_map` info-only skill (pattern A) sets
  `_force_map`. Cross with {ASCII, JSON} to test encoding-independence.
- **(b) bbox = new TOOL + map-hidden VARIANT.** `reveal(x1,y1,x2,y2)` info-only skill
  (pattern B): crop `render_map_view`/`build_map_model` rows+entities to the rectangle
  (`tty` row = `y+1`; slice `tty_chars[y1+1:y2+2, x1:x2+1]`), return
  `SkillResult(actions=[], feedback=<region>)`. Pair with a variant whose template sets
  `include_map=False`. Gotcha: skills get `(env, obs, **kwargs)`, not `state` — reach
  `raw_obs` via `env`/`obs` or thread it as code-mode does (`nethack.py:655`). Bound the
  region size (feedback flows verbatim into the prompt).

**Files (1d):** `prompt_spec.py` (templates + registry), `rendering.py` (include_map gate +
map fingerprint), `skills.py` (`reveal`/`request_map` near the journal/wiki block :1781),
`nethack.py` (`setup_flags` at :465).

---

## Sequencing & cost

1c (config-only, ~1 optional code flag) → 1b (harness-layer schema + trackers) → 1d (new
templates + tools). Each arm is an n=16 sweep (~1–2 h on Flash). This is many cell-runs;
recommend building all the code first, smoke-verifying each, then running arms in waves.

**Deferred/hard:** 1b lit/dark (needs new NLE obs); real-LM belief-state (needs in-process
`sub_lm`). Both flagged, not blocking.
