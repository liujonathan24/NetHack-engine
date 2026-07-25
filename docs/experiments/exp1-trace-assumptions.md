# Experiment 1 — trace verification assumptions

Every assumption below must hold in the actual rollout traces before the results are
trustworthy. Each is stated so a subagent can **confirm or refute it from the trace files
alone** (`outputs/encoding_eval/<run>/<CELL>/…` — vf-eval `results.jsonl` + per-turn NDJSON
under `<CELL>/trace/`). A subagent returns, per assumption: PASS / FAIL / UNVERIFIABLE + the
evidence (file, line/turn, quoted snippet).

## Fixed-factor assumptions (must hold in EVERY cell)

- **A1 · Character pinned.** Every rollout plays `Val-hum-neu-fem`. Evidence: the bottom
  status line / welcome shows a Valkyrie; `meta.character == "Val-hum-neu-fem"` if present.
- **A2 · Seeds pinned & paired.** The 16 rollouts use exactly NLE seeds 0–15, one each, and
  the SAME seed set appears in every encoding cell (so seed *k* is the same game across
  encodings). Evidence: per-rollout seed field; the turn-1 map for seed *k* is identical
  across cells (same starting room glyphs).
- **A3 · No compaction.** `compact_obs=false` took effect: the map region is the full grid,
  not blank-row-stripped / RLE'd / gutted. Evidence: the rendered map in the trace is a full
  21×79-ish grid, not a near-empty block.
- **A4 · NetPlay action surface.** `skill_set="netplay"`: the exposed tool list contains
  `move_to`, `explore_and_descend`, `search`, `attack`, `descend`, … and **no low-level
  `move(direction=…)`** primitive and no `dir8` tools. Evidence: the tool schema / tool_calls
  in the trace.
- **A5 · Uncapped tier.** `tier="full_nle"`: episodes are NOT force-terminated at dungeon
  level 6. Any rollout that reaches dlvl 6 keeps going (ends on death or the turn cap), and
  no rollout shows `succeeded=True` fired by a dlvl-6 milestone. Evidence: termination reason;
  a rollout at dlvl≥6 that continued.
- **A6 · Turn budget.** `max_turns=150` — no rollout exceeds it; rollouts that end early end on
  death, not a lower cap.
- **A7 · Model & billing.** Every LM call is `google/gemini-3-flash-preview` and NONE returns
  `insufficient_funds` / 402 / 401 (i.e. the team-billing header held for the whole run).
  Evidence: no funds/auth errors anywhere in the cell's logs.

## Per-encoding rendering assumptions (the "did it render as intended" check)

- **A8 · B0** — raw uncompressed ASCII grid is present and legible in the user turn.
- **A9 · JSON** — the map arrives as a structured object / list of cells with attributes
  (not ASCII art); the attributes are actually populated.
- **A10 · TOON** — compact structured-text map present and well-formed.
- **A11 · IMG** — a rendered **pixel tileset image** is actually attached to the user turn
  (image content, not a path to a missing file / empty image).
- **A12 · IMG_TTY** — a rendered **tty-raster image** is attached to the user turn.

## Play-quality assumptions (unexpected-issue guards)

- **A13 · The agent actually plays.** Tool calls parse and are accepted by the env — the
  rollout is not dominated by malformed tool calls, empty completions, or no-ops. A healthy
  fraction of turns advance game state (dlvl / position / turn count move).
- **A14 · No degenerate loop.** No rollout is stuck repeating one identical action / oscillating
  on one tile for the majority of its turns (the known failure mode).
- **A15 · No silent voiding.** No mid-run env-server death / worker restart / `--More--` freeze
  / crash that would zero-out or truncate a rollout without it counting as a real death.
- **A16 · Metrics present.** Each rollout logs `max_dlvl_reached`, the reward components
  (scout/descent/success/ascension), turn count, and a death/alive indicator — enough to build
  the results table and the alive-at-cap column.

## What a subagent should do

1. Read the assigned cell's `results.jsonl` + a sample of its `trace/*.ndjson`.
2. For each assumption A1–A16, mark PASS/FAIL/UNVERIFIABLE with quoted evidence.
3. Flag ANY anomaly not covered above (surprising deaths, empty maps, repeated identical
   observations, token blowups, encoding bleed-through).
4. Return a compact structured verdict — do not summarize the game narratively.
