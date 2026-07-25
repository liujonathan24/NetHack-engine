# Results collation V3

Supersedes V2. Compiled 2026-07-22 by sweeping all 10 worktrees. Written to be walked
through with collaborators.

**Rules applied throughout.**

1. Every number traces to a file in §0. Nothing is estimated or carried over from the draft
   tables.
2. **Any number measured under a known harness bug is shown as `—`** and listed in §6.
3. Experiments are cut along the axis each one varies — **observation** (Exp 1), **action
   interface** (Exp 2), **agent architecture** (Exp 3), **late-game capability** (Exp 4).
4. In Exp 1 the top table is **best-known variant per encoding**; the sub-tables underneath
   are what justify each of those choices.

---

## 0 · Provenance index

| # | source | worktree | what it holds |
|---|---|---|---|
| S1 | `docs/netplay-parity-report.md` §6b/6c | main | encoding × pet matrix n=24; 5-encoding run n=6 |
| S2 | `docs/netplay-parity-writeup.md` | main | 5-encoding table, methodology, dlvl-2 cap bug |
| S3 | `.claude/worktrees/refactor/experiments/results/wave1_summary.md` | refactor | 7 harness variants × 5 seeds, CIs + Cohen's d |
| S4 | `.claude/worktrees/refactor/experiments/results/exp15_token_savings.json` | refactor | per-turn + cumulative observation tokens |
| S5 | `.../results/skill_ladder_baseline.md` | refactor | `dir8` / `move` / `full` skill ladder |
| S6 | `.../results/hosted_eval_code_mode_dynamic_subgoal.md` | refactor | code vs skill mode, tokens + cost |
| S7 | `.../results/hosted_eval_haiku_vs_qwen.md` | refactor | Haiku 4.5 vs Qwen3.5-9B; **wiki/recall call counts** |
| S8 | `.claude/worktrees/refactor/approaches/BENCHMARK_primitives.md` | refactor | Voyager / RLM / Go-Explore, primitives-only |
| S9 | `outputs/curriculum_experiments/reverse_curriculum/REPORT.md` + `scripted_nav/`, `glm5.2_partial/` | main | reverse-curriculum climb + scripted ceiling |
| S10 | `outputs/curriculum_experiments/{voyager,go_explore,go_explore_llm,voyager_attack}/*.json` | main | 6-floor curriculum, primitives-only |
| S11 | `.claude/worktrees/blog-v1/docs/CURRICULUM_RESULTS.md` | blog-v1 | GLM-5.2 curriculum tour, 6/6, skill-assisted |
| S12 | `.claude/worktrees/ch-curriculum-primitives/docs/CURRICULUM_PRIMITIVES_RESULTS.md` | ch-curric | CH loop, code-vs-skill, 90-trace base rate, deep segment |
| S13 | `.../docs/NAV_MODE_ITERATION_REPORT.md` | ch-curric | 11 nav bugs → 6/6 seeds beaten |
| S14 | `.../outputs/exp_*/run.log` | ch-curric | rubric rewards per interface / turn budget |
| S15 | `.claude/worktrees/blog-v1/docs/EXPERIMENTS.md` | blog-v1 | 21-experiment catalog with recalibration notes |
| S16 | `environments/nethack/nethack_harness/prompt/balrog.py` | main | the progression scorer (a proxy — see §1) |

---

## 1 · Metrics

**Depth Score = `max_dlvl_reached`** (or `max_curriculum_floor` on the 6-floor tour).
Measured everywhere; comparable to NetPlay's published 2.6.

**BALROG Progress Score — never measured.** What exists (S16) is an analytic proxy
`P(ascend) = (DL/50)^1.3 · (XL/30)^0.6`, calibrated to four headline points because the real
table isn't published. It returns **6.4%** where glyphbox reports **12.56%** at the same
(DL=10, XL=10), and it is computed from depth and XL rather than milestone achievement — a
different quantity. Every BALROG column stays empty. See §7.1.

---

## 2 · Experiment 1 — Observation encoding

> **NEW — rigorous run (2026-07-24):** a fresh, verified encoding sweep now exists on a
> different model and the *uncapped* tier: **Gemini 3 Flash via Prime, `full_nle` (uncapped),
> pinned Valkyrie, seeds 0–15, n=16, no compaction, `move`-gated action surface.** Result:
> **B0 2.00 ± 0.38 > IMG_TTY 1.88 > TOON 1.81 > JSON 1.44** — but **no pair is significant at
> n=16** (paired sign-p ≥ 0.18); JSON is the consistent laggard, B0 owns the only dlvl 5–6
> reaches and the lowest token cost. This run fixed three confounds the older numbers below
> silently carried (Prime team-billing, per-turn intro-banner bleed, and a `move` tool-gating
> leak that let the withheld primitive execute asymmetrically across encodings). Full write-up
> + full-corpus verification: **`docs/experiments/exp1-results.md`**. The older
> Qwen3-VL/`full_dungeon_easy` numbers below are retained for history but are censored at
> dlvl 6 and pre-date the gate/banner fixes — prefer the new run.

**Held fixed:** the `netplay` skill set (high-level actions, no low-level `move(direction)`;
navigation via `explore_and_descend` / `move_to` / `search` / `kick` + survival skills),
model `qwen/qwen3-vl-235b-a22b-instruct`, `max_turns=150`. **Only the observation varies.**
This is NetPlay's own design — hold the action set constant, vary the observation. Contrast
with §3d, where the *harness* is what varies.

> ### ⚠ Known design defect in every Exp-1 run so far: the depth ceiling
>
> All of these ran on tier `full_dungeon_easy`, whose success milestone **terminates the
> episode at dungeon level 6**. Every mean below is therefore a **censored** measurement —
> we cannot see how deep the good rollouts would have gone, and the top of the distribution
> is clipped exactly where the interesting encodings differ (B0 already puts rollouts at
> dlvl 5 and 6). This plausibly compresses the differences between encodings and is a
> leading suspect for why the ranking looks the way it does. **Re-runs must use an
> uncapped depth objective**, with the episode ending on death or turn budget only.

### 2a. Headline — best-known variant per encoding

Each row is the **best variant we have measured for that encoding**; §2b–§2e are the
sub-tables that justify those choices.

| Encoding | Representation | Depth Score | Alive at turn 150 | Avg token cost | n | Notes *(column will be removed)* |
|---|---|:-:|:-:|:-:|:-:|---|
| **Uncompressed ASCII (B0)** | raw 21×79 grid | **2.74 ± 0.41** | 52% (death 48%) | 295 obs tok/turn | 23 | best overall; best-of-ASCII = uncompressed + pet |
| Compacted ASCII (B1) | blank-strip + RLE + journal-diff | 2.29 ± 0.24 | 8% (death 92%) | 214 obs tok/turn | 24 | compaction gutted the map region |
| **JSON — best-of-JSON** | structured map object | 1.96 ± 0.27 | 54% (death 46%) | — | 24 | **only one cell-schema ever tried** — see §2c |
| JSON | structured object | 2.33 | — | — | 6 | n=6 run, taken under the compacted default |
| IMG_TTY | rendered image of the tty raster | 1.83 | — | — | 6 | n=6 run, compacted default |
| TOON | compact structured text | 1.50 | — | — | 6 | n=6 run, compacted default |
| IMG | rendered pixel tileset | 1.33 | — | — | 6 | n=6 run, compacted default |

Two power blocks: rows 1–3 are **n=24**; rows 4–7 are the **n=6** run that is the only place
TOON / IMG / IMG_TTY were ever measured, and it was taken under the compacted default, so it
understates ASCII. Merging them into one publishable ranking requires re-running the vision
and TOON arms at n=24 uncompressed and uncapped (§8, run 6).

**"Alive at turn 150"** answers *could it have gone deeper?* — the number shown is
`1 − death%`, which is the closest thing currently logged; it conflates "hit the turn cap
still playing" with "terminated early on the NLE step budget." The true metric needs a
per-rollout `alive_at_cap` flag (§7.3). Even as a proxy it is informative: **compacted ASCII
kills 92% of its rollouts** — the depth number understates how badly that encoding fails.

**Avg token cost** is per-turn observation tokens (§2b). Output tokens per turn were never
logged separately; whole-run totals exist only for the skill-vs-code comparison (§3b).

**The ordering, and the part that doesn't fit.** Text beats vision decisively — pixel tiles
are worst (1.33), the tty raster next-worst (1.83). Within text, the raw grid beats every
compression. **But JSON should have won and didn't.** The stated hypothesis — a structured
object parses more reliably than ASCII art — predicts JSON ≥ ASCII. We measure JSON *below*
uncompressed ASCII (1.96 vs 2.74). The most likely explanation is not that JSON is a bad
idea but that **we have only ever tried one JSON schema**, and it is thin. §2c is the
experiment that decides this.

### 2b. Token cost (S4)

Seed 42, 60 turns, tokenizer-counted observation payload:

| Config | obs tokens / turn | cumulative @ turn 60 | saving |
|---|---:|---:|---|
| Raw baseline (uncompressed obs, full history) | 295.0 | 17,700 | — |
| Obs compaction only | 214.2 | 12,854 | 27.4% per turn |
| Obs + history compaction | 214.2 | 1,835 | **89.6% cumulative** |

**Map compaction buys 27% of per-turn tokens and costs ~0.4 dlvl. History compaction buys
89.6% cumulative and has never been shown to cost anything.** Two different levers bundled
under one word. Per-encoding token costs (JSON vs TOON vs IMG) were never logged.

### 2c. Sub-table — what goes in a JSON cell **[not yet run]**

The JSON encoding is a list of cells, each with a list of attributes. **We have never varied
the attribute set.** Ablate one at a time against a fixed base:

| Attribute | Rationale | Result |
|---|---|:-:|
| glyph / item type + what the agent knows about it | the minimum; identity beyond a raw char | — |
| seen vs unseen | the blank `' '` glyph currently means **both** dark floor and solid stone — a documented cause of routing into walls | — |
| lit vs dark | separates "explored and empty" from "not yet looked at" | — |
| visited vs unvisited | frontier reasoning the agent currently has to infer | — |
| distance / reachability from the hero | offloads the thing agents demonstrably fail at | — |

The last row is a boundary case: reachability shades into *locating for* the agent, which the
honest-perception principle in Exp 4 deliberately removes. Run it, but report it separately.

### 2d. Sub-table — memory and journalling variants **[partially run]**

Which memory mechanism the agent gets is an observation choice, and we have several
(journal `add_note`/`recall`, pinned objective, belief-state summaries, history compaction)
that have never been compared head to head.

| Memory variant | What it is | Result |
|---|---|:-:|
| Journal (`add_note` / `recall`) + pinned objective | current default | baseline of every Exp-1 row |
| Belief-state summarization | periodic rewritten state summary | folded into B1; never isolated |
| Summarize-and-reset (variant R) | hard-truncate history before the last checkpoint | 0.111 vs B1 0.082 on the rubric (§3d) — a token win at capability parity |
| History-compaction on/off | keep-full / drop-after thresholds | 89.6% cumulative token saving (§2b); capability cost never measured |
| No memory at all | control | — |

The one thing we can say: history compaction is a large token win with no measured capability
cost, and R is a further token win at parity. The rest is unmeasured.

### 2e. Sub-table — when the observation is exposed **[not yet run]**

The **delayed / on-demand map** is the current final variant: the agent gets **action
feedback every turn**, and the full map is re-issued **only on request or when it materially
changes**. Not an encoding — a change in *when* the observation is delivered. Zero
measurements exist. It must be crossed with both text modes, because the interesting claim
is that it helps regardless of encoding:

| arm | ASCII (B0) | JSON |
|---|:-:|:-:|
| map re-sent every turn (current baseline) | 2.74 ± 0.41 | 1.96 ± 0.27 |
| **map delayed / on-demand** | **—** | **—** |

Predicted mechanism: cuts per-turn tokens the way compaction does but *without* gutting
spatial context, since the full map is still available on request — i.e. it should break the
token-vs-depth tradeoff §2b exposes.

Also unmeasured on this axis: **image size / resolution** for the IMG and IMG_TTY arms. Both
vision encodings were run at one resolution, so "vision loses to text" is currently a claim
about one particular rendering.

### 2f. Are encoding effects independent of the algorithm? **[not yet run]**

The claim we want to make is architectural. Three outcomes are distinguishable and we
currently cannot tell them apart:

- **encoding dominates** — the ranking holds under every algorithm; encoding is a
  free-standing result;
- **algorithm dominates** — one algorithm wins under every encoding; encoding is second-order;
- **they co-depend** — e.g. code-mode agents prefer JSON (they can index it) while skill-mode
  agents prefer ASCII (they read it as a picture). Most likely, and most interesting to write up.

Design: a full square, **{2 algorithms} × {6 encodings}**, n=24 on pinned seeds, uncapped
depth. Read the interaction term, not the margins.

### 2g. *(optional note — not for the main writeup)* Standing against NetPlay

NetPlay (Jeurissen et al., CoG 2024) reports **2.6 mean max dungeon level on GPT-4**. Our
best measured config is 2.74 ± 0.41 (n=23) — nominally above, but the comparison differs in
harness, model, tier and turn budget simultaneously, and the SE is wide. What the data says
about *why* we're weak, in order of evidence strength: (1) agents **never level up** — mean
XL **~1.4**, most games end at the starting XL 1 with ~0.4 kills/game, so they meet floor-2-4
monsters at ~14 HP; (2) **death, not the clock**, ends ~⅔–¾ of rollouts, only ~1 in 30 hits
the turn cap; (3) every condition is **bimodal** — ⅓–½ stall at dlvl 1 while the rest reach
3–6. This looks like tuning distance, not a structural gap. **Action item:** check whether
NetPlay published open-model results — if so, that is the fair comparison rather than
benchmarking a mid-tier open model against GPT-4.

---

## 3 · Experiment 2 — Action interface and skills

Everything here varies **what the agent can do**, holding the observation fixed.

### 3a. Background on the baselines and the harness variants

**NetPlay** (Jeurissen et al., CoG 2024). An LLM agent built on a high-level skill API over
`autoascend`, a complete rule-based NetHack bot. Its core skill, `explore_level`, explores a
whole floor over a persistent per-floor map and hands control back to the LLM at decision
points. Its design principle — hold the action set fixed and vary only the observation — is
the one Exp 1 copies. Published headline: **2.6 mean max dungeon level, GPT-4**.

**GlyphBox** (Wang 2026). Filters the raw glyph grid into a structured entity list and pairs
it with a **code interface**, which is why it matters here: its headline (GPT-5.2 to dlvl 10)
is a *combined* encoding + interface result, exactly the interaction §2f is designed to
separate. Our variant G is n=1 and unusable, so **we have no GlyphBox baseline.**

**BALROG** (Paglieri et al., ICLR 2025). The benchmark whose progression score we cannot yet
compute (§1). Its no-ASCII condition is our variant B, and it is the one clean negative in
§3d.

**Continual Harness** (arXiv:2605.09998). A mid-rollout self-refinement directive: the agent
is periodically prompted to revise its own approach and write to a journal. Implemented as
variant P (every 20 turns) and as the CH loop in Exp 4, where a **separate teacher model**
emits CRUD edits across prompt / sub-agents / skills / memory every `refine_interval` turns.
On 200-turn rollouts it does not move the needle (§3d); the standing hypothesis is that 200
turns is too short to amortize.

### 3b. Pet tactics — can you hack the game?

An *action* variant, not an encoding: the harness either instructs the model to let the pet
fight or a skill does it — wait ≤8 turns on the downstairs for the pet to be adjacent so it
follows, kite (≤4 steps) when a hostile is adjacent and the pet is engaging, melee only when
cornered. n=24 per cell.

| Observation held fixed | pet ON | pet OFF | **pet Δ** | death % on/off |
|---|:-:|:-:|:-:|:-:|
| Uncompressed ASCII (B0) | 2.74 ± 0.41 | 2.21 ± 0.37 | **+0.53** | 48% / 71% |
| Compacted ASCII (B1) | 2.29 ± 0.24 | 1.83 ± 0.18 | **+0.46** | 92% / 58% |
| JSON | 1.96 ± 0.27 | 2.29 ± 0.28 | **−0.33** | 46% / 54% |

**The question is whether degenerate short-run behaviour helps: yes on ASCII (+0.5 on both),
no on JSON (−0.33), for reasons that are not clear.** Averaged across encodings it reads
"net-neutral," which is JSON cancelling ASCII — the kind of false null that per-encoding
breakout catches. Each Δ is ~1–1.5 SE.

A caveat that feeds §2g: pet-kill **suppresses player XP** — the pet takes the kills — which
plausibly worsens the XL-1.4 weakness that is our leading explanation for being weak overall.
Depth up, strength down.

Depth distribution, n=24 each:

| run | dlvl 1 | 2 | 3 | 4 | 5 | 6 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| B0 ASCII, pet | 10 | 2 | 4 | 2 | 3 | 1 |
| B0 ASCII, no pet | 14 | 3 | 2 | 1 | 2 | 1 |
| B1 compacted, pet | 8 | 6 | 6 | 3 | 1 | 0 |
| B1 compacted, no pet | 11 | 7 | 5 | 1 | 0 | 0 |
| JSON, pet | 12 | 7 | 2 | 1 | 1 | 1 |
| JSON, no pet | 8 | 8 | 5 | 0 | 2 | 1 |

*(The dlvl-6 column is the censoring ceiling, not a distribution tail — see the §2 warning.)*

### 3c. Skill mode vs code mode — **the key experiment of this section**

The question is *how should automation be exposed*: one function-tool per skill, or a single
sandboxed `code(source=…)` tool over an `nh` namespace where the agent writes Python and
plays many steps per round-trip. Measured on the 6-game primitives curriculum, GLM-5.2, **no
descend/ascend skill** (depth = curriculum floor of 6):

| Action interface | turn budget | games reaching floor 4 | floor distribution | rubric reward |
|---|:-:|:-:|---|---|
| Skill mode (CH loop, best iteration) | 150 | **2 / 6** | `[2,1,4,2,4,3]`, mean 2.67 | — |
| Code mode | 150 | 0 / 6 | `{1:2, 2:3, 3:1}` | 2.19 ± 1.29 |
| **Code mode** | **300** | **3 / 6** | `{2:1, 3:2, 4:3}` | 26.66 ± 23.11 |
| Code mode | 600 | 3 / 6 | — | 30.98 ± 19.80 |
| Code + manual-move stuck hint | 300 | 3 / 6 (neutral) | — | 30.19 ± 20.63 |
| Code + forced auto-unstuck | 300 | 2 / 6 (**hurt**) | — | 24.06 ± 19.89 |
| **Code, capped at ~8 game turns per call + explicit `continue`** | 300 | **—** | — | — |

**The turn budget is the confound and it dominates.** Code mode is strictly worse than skill
mode at 150 turns and strictly better at 300, because it spends turns *perceiving* — printing
the map, inspecting cells — before acting. Any code-vs-skill claim that doesn't control the
budget is measuring the budget.

**New arm to add (last row): bounded delegation.** Cap any single skill or code invocation at
~8 game turns; to keep going the model must explicitly emit `continue`. The motivation is
that an unbounded call lets the model check out for hundreds of in-game turns with no
decision point — which is both a capability problem (no re-planning on new information) and
an evaluation problem (we stop measuring the agent and start measuring the script). This
directly tests whether code mode's advantage is genuine reasoning throughput or just budget
absorption.

Cost side (S6, Qwen3.5-9B, 1 rollout each): skill mode 4.3M input / 46K output over 146
turns; code mode 3.1M / 49K over 136 turns — **27% fewer input tokens**, because one code
call dispatches ~3× more game actions per round-trip.

Base rate from the full corpus: across **all 90 traces** (inference-free, judged by
`max_curriculum_floor`), **21/90 reached floor 4 = 23% per-game rate**. At 23% i.i.d.,
P(all 6 seeds succeed) is 2.6% at best-of-3, 15% at best-of-5, 63% at best-of-10 — optimistic,
since seeds are not i.i.d. *Caveat from the same analysis: the eval **reseeds every episode**,
so those 90 traces are 90 distinct seeds, not 6 repeated.*

### 3d. Skill ladder — does the mega-skill do all the work? (S5)

| skill set exposed | model | n | descents | ever saw `>` | mean reward |
|---|---|:-:|:-:|:-:|---:|
| `dir8` (8 compass tools only) | Qwen3.5-9B | 8 | **0/8** | 0/8 | 0.064 |
| `move(direction)` + survival | Qwen3.5-9B | 8 | **0/8** | 0/8 | 0.065 |
| `full` (14 skills incl. mega-skills) | Qwen3.5-9B | 24 | **0/24** | 0/24 | 0.079 |
| `full` | Haiku 4.5 | 4 | 0/4 | — | — |

Eight primitive tools to a 14-skill mega-stack moves mean reward 0.064 → 0.079 and unlocks
**zero** descents. Not model-specific: Haiku 4.5 also goes 0/4 while calling
`find_and_descend` 9–17 times.

### 3e. Harness variants — model fixed, harness swapped (S3)

Contrast with §2, where the harness was fixed and the observation varied. 7 variants × 5
seeds (22–26), 200-turn cap, `Qwen/Qwen3.5-9B`, `nethack@0.0.64`. Metric is Prime's
rubric-weighted `avg_score` (scout 1.0, descent 10.0/dlvl, success 100, ascension 1000) —
**not depth**, and not decomposable (hosted `prime eval get` doesn't return the per-function
breakdown).

| variant | what it is | n | mean | SD | 95% CI | Cohen's d vs B1 |
|---|---|:-:|---:|---:|---|---:|
| **N** | NetPlay skill-only (no `move(dir)`) | 4 | **1.137** | 1.235 | [0.068, 2.206] | +1.21 |
| R | summarize-and-reset | 4 | 0.111 | 0.061 | [0.070, 0.168] | +0.60 |
| P | Continual-Harness self-refinement | 3 | 0.103 | 0.005 | [0.100, 0.108] | +0.84 |
| B0 | no compaction (calibration) | 3 | 0.102 | 0.025 | [0.078, 0.127] | +0.68 |
| G | GlyphBox + code mode | 1 | 0.095 | — | — | — |
| B1 | current default (compaction + belief) | 4 | 0.082 | 0.035 | [0.051, 0.112] | 0.00 |
| B | BALROG no-ASCII | 5 | 0.056 | 0.018 | [0.042, 0.071] | −0.92 |

**Stripping the ASCII grid (variant B) is the only clean negative** — M-W U=15, p=0.286 at
n=5, Cohen's d=−0.92 — corroborating §2a from another direction: the grid does work a
natural-language scene description cannot replace. N is bimodal (2.155, 2.257, 0.039, 0.097):
removing `move(direction)` raises variance in both directions. **G is n=1 and unusable**
(4 of 5 seeds killed at a 130-min stuck timeout).

### 3f. Cross-model at matched tier, and tool-usage rates (S7)

| metric | Qwen3.5-9B | Claude Haiku 4.5 |
|---|---:|---:|
| scout_reward | **0.132** | 0.077 |
| descent_reward | 0 | 0 |
| turns | 183 | 201 |
| autoexplore calls | 47 | 17 |
| **wiki_search calls** | **0** | **1** |
| **recall calls** | **0** | **1** |
| pin_objective calls | 1 | 1 |
| input / output tokens | 4.27M / 31K | 5.61M / 22K |
| estimated cost | **$0.79** | $5.72 |

Two very different models both scoring **0 descents** is the signature of a *harness* ceiling.
And note the wiki rows — see §4b.

---

## 4 · Experiment 3 — Continual harness and agent architectures

### 4a. What the three architectures are

**Voyager** (Wang et al., arXiv:2305.16291). An LLM lifelong-learning loop with three parts:
an **automatic curriculum** proposing the next objective, a **skill library** of executable
code the agent writes and keeps, and a **verify** step deciding whether a new skill is
retained. Here: each iteration the model proposes an objective (usually "increase dungeon
level"), writes a macro composed of primitives, runs it, and keeps it in the library on
success. The claim to test is *compounding* — later objectives should get cheaper as the
library grows.

**Go-Explore** (Ecoffet et al., arXiv:1901.10995). "First return, then explore." Archive
interesting states keyed by a cell descriptor, **return** to a promising one
deterministically, then explore from there. NetHack fits unusually well because the engine
provides byte-exact `snapshot()` / `restore()`, so returning is free rather than replayed.
Here: cells keyed `(progress, dnum, x//3, y//3)`, selection weighted
`(1+progress)/(1+visits)`, exploration sampling weighted primitives (compass steps, run
macros, `search`, real `>`/`<`) with **no forced stair-take**. Two arms: *keyless* (random —
the control) and *guided* (an LLM picks the action).

**RLM — Recursive Language Models.** The top agent drives the game through the code/REPL
interface rather than one tool call per turn, and decomposes long-horizon reasoning by
calling **sub-LMs** over slices of context (`nh.summarize`, `nh.plan`, `nh.recall_lm`). The
claim is that recursion handles horizon length better than a flat context.

All three share a wiki snapshot (`wiki_lookup` / `wiki_search`) so knowledge access is not a
confounder.

### 4b. The wiki is essentially unused

The shared wiki was built so knowledge access wouldn't confound the comparison. In practice
agents barely touch it:

| run | wiki_search | wiki_lookup | recall | over |
|---|:-:|:-:|:-:|---|
| Qwen3.5-9B, corridor_explore (S7) | **0** | 0 | **0** | 183 turns |
| Claude Haiku 4.5, same tier (S7) | **1** | — | **1** | 201 turns |

Roughly **0–1 knowledge lookups per 200-turn rollout.** Two readings, and we cannot yet
distinguish them: either the wiki isn't on the critical path for early-dungeon play (likely —
these tiers reward tile coverage, not knowledge), or the tool is badly surfaced and the agents
don't think to reach for it. Either way, **no current result is confounded by wiki access**,
and the "knowledge grounding" story in the Voyager arms is doing less work than the design
assumed. Worth an explicit on/off ablation (§8) — and note the contrast with §5e, where
Claude agents *did* construct the invocation ritual correctly, which is exactly the kind of
task the wiki should be load-bearing for.

### 4c. Primitives-only, no vision (S8) — everything ties a random walk

`poolside/laguna-m.1`, 5 seeds {2, 9, 19, 42, 123}, 100 iterations, mega-skills removed.

| Architecture | evolving machinery | built a nav tool? | Depth Score |
|---|---|---|:-:|
| Voyager | skill library + auto-curriculum | yes — 5–11 skills/seed | 1.0 *(see caveat)* |
| RLM | recursive code mode | attempted; all errored | 1.0 *(see caveat)* |
| Go-Explore (guided) | LLM exploration + archive | yes — archive | 1.0 *(see caveat)* |
| Go-Explore (keyless, control) | random exploration + archive | n/a | 1.0 |

**The diagnostics are the result, not the depth number** — which is contaminated (§6), since
this ran on the pre-fix harness. What survives inspection: Voyager proposed **418** descend
objectives and succeeded **0** times; RLM's end position equalled its spawn position
**exactly on all 5 seeds** after 100 turns (laguna's navigation code errored and fell back to
a single wall-blocked step); guided Go-Explore discovered exactly **1 cell** — the start — so
its archive had nothing to return to and it was indistinguishable from the random control.

### 4d. Turn vision on (S8) — the architectures come alive

Same benchmark, `reveal_map` overlay on, 5 seeds. **Treat the depth numbers as directional
only** (§6); the *contrast* is the finding.

| Architecture | per-seed depths {2, 9, 19, 42, 123} | mean | vision-off |
|---|---|:-:|:-:|
| **Voyager** | 1 · 5 · 7 · 1 · 1 | ~3.0 | 1.0 |
| **RLM** | 1 · 3 · 2 · 1 · 4 | ~2.2 | 1.0 |
| Go-Explore keyless (control) | 1 · 1 · 1 · 1 · 1 | 1.0 | 1.0 |
| Go-Explore guided | stopped — timeout-dominated at depth 1 | — | 1.0 |

**This is the interesting one.** Give the map-reading agents the information and they use it:
Voyager synthesizes and reuses a `navigate_and_descend` skill floor-to-floor and reaches
dlvl 5 and 7; RLM writes valid `move_to`→stairs code once the `>` is on screen. The control
stays flat at 1.0 — vision helps agents that can *act on* a known stair location and does
nothing for one that cannot express "path to the visible stairs." With `reveal_map` off, the
agents are, metaphorically, in the dark; with it on, the machinery they build actually
compounds.

Three things this established: (1) the knob alone reaches nobody — `reveal_map` fills
`chars`/`glyphs` but not `tty_chars`, and the harness rendered from `tty_chars`, so each
agent had to be rewired to read the reveal-filled array; (2) descent is agent × seed
dependent — Voyager cracks {9, 19}, RLM cracks {9, 19, 123}, neither dominates; (3) laguna is
a reasoning model, which surfaced three latent bugs that would have silently corrupted the
comparison — empty content under low `max_tokens` (guided Go-Explore was capped at 20 output
tokens and was therefore *always* falling back to random), a 30 s client timeout causing a
retry storm, and the wrong `nethack_harness` being imported from the main checkout.

### 4e. The 20-seed extension — **retracted, do not quote**

| Architecture (vision on, 20 seeds) | reported | status |
|---|---|---|
| Voyager | 6/20 descended, mean 2.0, max 7 | **— contaminated** |
| RLM | 6/20 descended, mean 1.7, max 5 | **— contaminated** |

The reported failure mode — on ~70% of seeds "the stairs need corridor exploration, which
`reveal_map` breaks: filling the map makes the frontier detector see everything as explored,
so the explore fallback stalls," with 200 searches opening no path — **is the bug, not a
property of the algorithms.** `reveal_map` was a render-only overlay that displayed terrain
while leaving secret doors (SDOOR) and corridors (SCORR) **impassable in `levl[][]`**, and
didn't render SCORR at all: the downstair genuinely sat behind an invisible, un-walkable
passage. Fixed in the engine (`winrl.cc`: convert SDOOR→DOOR, SCORR→CORR when `reveal_map>0`
and re-render), after which all six curriculum seeds' stairs became navigable, up from 3/6
stuck. Re-run required (§8, run 5).

---

## 5 · Experiment 4 — The six-floor curriculum (late-game capability)

**Why this experiment exists.** Experiments 1–3 all bottom out in the same place:
navigation, made harder by combat, which the model has not learned to do well. This
curriculum sets that aside and asks a different question: **if traversal is granted, can the
agent actually play the game?** Can it descend to the bottom and climb back? Does it know
what to do at dungeon level 50 — a place no agent in the literature reaches — where the task
stops being "find the stairs" and becomes Gehennom, the Invocation level, and Moloch's
Sanctum?

| curriculum floor | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| dungeon level | DoD 1 | DoD 2 | DoD 3 | Gehennom 48 | Gehennom 49 | Gehennom 50 |

The DoD 3 → Gehennom 48 transition fires **only** when the hero genuinely stands on the real
boundary staircase and takes the real `>`. There is no teleport command the agent can call.

### 5a. Upper bound — descent handed over (S11)

GLM-5.2, `curriculum` tier, B0 raw ASCII, 3 seeds × 2 rollouts, `max_turns=40`, with
`descend`/`ascend` available:

| metric | value |
|---|---|
| success (reached the Elemental Planes) | **6/6 = 100%** |
| total reward | 55.05 ± 1.16 (descent 48.50, scout 5.55) |
| turns | 18.8 ± 2.8 |
| `move` / `move_to` calls | **≈ 0** |

The agent never navigates; it presses stair skills. A **substrate demonstration** — it proves
the tour is wired end to end and places agents in late-game content. Never quote it as an
agent capability result.

### 5b. Descent skill removed, pre-navigation-fix — **retracted**

| Architecture (3 seeds, full vision, 45 turns / 600 iters) | reported deepest floor | status |
|---|---|---|
| Go-Explore (random primitives) | 2, 2, 2 | **— contaminated** |
| Voyager (GLM-5.2) | 1, 1, 1 | **— contaminated** |
| Voyager-attack variant | 1, 1, 1 | **— contaminated** |
| Go-Explore-LLM | 1, 1, 1 | **— contaminated** |

All ran before the 11 harness navigation fixes and the engine de-secret. The contemporaneous
conclusion — "neither method descends; the descend skills were doing all the work" — was
**substantially wrong about the cause**: `nh.map.rows` didn't exist (518 errors across 78
traces, so the map-read idiom was broken every code-mode run), code mode discarded navigation
feedback, A* violated NetHack's diagonal rules into and out of doorways, `move_to` reported
*predicted* rather than *actual* outcomes, doors were never opened or kicked, and secret
passages were impassable.

### 5c. Reverse curriculum and scripted ceiling — **retracted for the same reason**

Both measured **climb reachability of the up-stair** on the pre-fix harness and pre-de-secret
engine — exactly the quantity the fixes moved. The failure attribution in that report (89% of
failures monster-related; 51% a blocking monster the climber can't engage) matches a bug
identified independently later: `move_to`'s reachability **gate** uses a strict
`reachable_set` excluding monsters and boulders, so a single blocker at a chokepoint makes a
genuinely reachable target report "no route." Numbers withheld; worth re-running because it
is free and it is the right control.

*(Two operational findings survive: Prime Inference returned **HTTP 402** mid-run and the loop
silently swallowed the errors into no-op turns, so 37 of 72 episodes burned their full budget
in 8–14 s and looked like genuine zero-success data; and an intermittent `--More--`
welcome-prompt freeze made **every** command a silent no-op at process start, producing a
fake single-seed win pattern.)*

### 5d. After the navigation harness was fixed — the result that stands

Eleven harness bugs and two engine bugs fixed (restored `nh.map.rows`; surfaced nav feedback;
closed-loop honest `move_to` that opens and kicks doors and attacks through corridor monsters;
NetHack-correct diagonal rules; prompt-settle after descent; engine `reveal_map` de-secret;
engine `hero_on_stair` recognising the branch staircase).

**Sample: 6 seeds × 1 Claude subagent each = 6 runs, no descend/ascend skill.**

| seed | reached floor 4 (Gehennom, dlvl 48)? | descents |
|:-:|:-:|:-:|
| 19 | yes | 3 |
| 20 | yes | 4 |
| 21 | yes | 4 |
| 22 | yes | 3 |
| 23 | yes | 3 (found the hidden floor-2 passage) |
| 24 | yes | 3 |

**6/6 runs (n=6) reached floor 4 by genuine LLM navigation**, 3–4 real descents each — agents
reading the map and routing to real staircases. This replaces every dashed row in §5b/§5c.
Caveats to state when presenting: n=6, one model family (Claude subagents), hand-driven
through the step driver rather than a batch eval, and floor 4 (dlvl 48) is the *entry* to the
deep segment, not the bottom.

### 5e. The deep segment — what the agent does at dungeon level 48–50 (S12)

**Sample: 10 seeds (19–28) × 1 Claude subagent each = 10 runs**, started at the jump (floor 4)
with the invocation kit.

| Seed | Reached | Seed max | Note |
|:-:|:-:|:-:|---|
| 19 | **6** | 6 | 48→49, ritual flawless → Moloch's Sanctum |
| 20 | 5 | 6 | floor-5 maze down-stair walled off by a master-lich / Aleax nest |
| 21 | 4 | 4 | short Gehennom — the jump lands on its Sanctum (at max) |
| 22 | 4 | 6 | boxed into a 10-tile pocket by 2 boulders + a hostile `f` |
| 23 | 4 | 6 | master lich + bone devil (wand of fire) jammed the dlvl-48 corridor |
| 24 | 4 | 6 | mind flayer + vrock pack on the path to the stair |
| 25 | 4 | 6 | boxed into a **1-tile** pocket by a single hostile `f` |
| 26 | **5** | 5 | auto-seated on its Invocation level, ritual → its Sanctum |
| 27 | 4 | 4 | short Gehennom (at max) |
| 28 | 4 | 4 | short Gehennom (at max) |

**Totals: n=10. Floor 6 (dlvl 50, the true bottom) ×1 · floor 5 ×2 · floor 4 ×7. 5/10 reached
their seed's achievable maximum** (3 of the floor-4 stops are *at* max, because those seeds
have a short Gehennom and the jump lands directly on their Sanctum). Across §5d + §5e that is
**16 agent-driven runs total, of which 1 reached dungeon level 50.**

**The answer to "does it know what to do at the bottom" is yes, for the hardest single thing
in the game.** Dungeon level 49 is the Invocation level — NetHack places **no down-staircase
there by design**; the only route to the Sanctum is the invocation ritual on the vibrating
square (apply the Candelabrum-lit Bell of Opening, read the Book of the Dead, wait out the
multi-turn recitation). Every subagent that reached an Invocation level and was seated
performed the ritual **flawlessly — 2/2**. Given the wiki and some experimentation, agents
construct that sequence correctly. **The ritual is not a blocker.**

What still caps floors 5–6 is the same `move_to` reachability gate as §5c: every dlvl-48
Gehennom maze is **one connected component** (683 / 696 / 572 walkable tiles) and the
down-stair is **always** reachable via `a_star(pass_monsters=True)` (path length 20–149) — but
one hostile or boulder at a chokepoint makes the strict gate report "no route," so the agent
concludes "disconnected maze" and searches walls instead of attacking the blocker. Gating with
`pass_monsters` should lift most of seeds 20/22/23/24/25 toward their real maximum.

---

## 6 · Contaminated and retracted results

Everything here was measured under a bug that plausibly determines the result, and is shown
as `—` above.

| Result | Bug | Evidence it matters |
|---|---|---|
| 20-seed Voyager / RLM vision-on (4e) | `reveal_map` render-only: SDOOR/SCORR impassable, SCORR unrendered | after the engine de-secret, 6/6 curriculum stairs became navigable, up from 3/6 |
| Primitives-only benchmark depth scores (4c, 4d) | same, plus the pre-fix nav stack | the diagnostics survive; the depth numbers do not |
| Primitives-only curriculum, 3 seeds (5b) | 11 harness nav bugs incl. missing `nh.map.rows` (518 errors / 78 traces), discarded nav feedback, illegal diagonals, no door handling, predicted-not-actual reporting | same seeds went floor 1–2 → 6/6 floor 4 on the fixed harness |
| Reverse curriculum + scripted ceiling (5c) | same nav bugs + `move_to` strict reachability gate | the "monster blocks the path" failure (51%) *is* the gate; mazes are provably connected |
| All depth numbers before commit `7c3e17e` | verifiers didn't round-trip the nested `task` dict → tier fell back to `corridor_explore` → **every eval terminated at dlvl 2** | proven with `--state-columns`: `succeeded=True` at `max_dlvl=2, num_turns=2` |
| Pre-fix cross-model check (Qwen3-VL 1.38 vs Sonnet-4.5 1.25) | dlvl-2 cap + death detector catching 3 of ~22 deaths | both taken under the cap |
| Reverse-curriculum episodes after the credit cutoff | Prime HTTP 402 swallowed into no-op turns | 37 of 72 episodes ran their full budget in 8–14 s |
| Any run hit by the welcome-prompt freeze | undismissed `--More--` makes every command a silent no-op | produced a fake single-seed win pattern; intermittent across launches |
| Variant G / GlyphBox (3e) | code-mode perf bug — rollouts >2 h | 4 of 5 seeds killed at a 130-min timeout; n=1 survives |
| **All Exp-1 depth means** | tier terminates the episode at **dlvl 6** | censored, not wrong — but the top of the distribution is clipped where encodings differ |

---

## 7 · What is missing

### 7.1 Benchmarks we cannot yet compare against **[blocking]**

- **BALROG Progress Score.** Never measured; our scorer is a proxy reading ~2× low against
  the one external anchor, computed from `(depth, XL)` rather than milestones. Until we get
  the real table or run the official scorer, all cross-paper claims use `max_dlvl`.
- **NetPlay.** Only their published GPT-4 number (2.6), across four simultaneous differences.
  Fixes: a matched `explore_level` arm inside our harness, **and** check whether NetPlay
  published open-model results — if so, switch to that comparison.
- **GlyphBox.** n=1, unusable — and their headline is a combined encoding + code-interface
  result, exactly the interaction §2f targets.

### 7.2 Models **[fractured is acceptable for this draft]**

No Gemini and no GPT-5.5 rollout exists anywhere. Present roster: Qwen3-VL-235B (encoding),
Qwen3.5-9B (variant sweep, skill ladder, cross-model), Qwen3.5-35B/4B (cost probes),
Haiku 4.5, Sonnet 4.5 (pre-fix), GLM-5.2 + GLM-5.1 teacher (curriculum, CH loop),
laguna-m.1 (architectures), Claude subagents (navigation and deep segment). **Every headline
rests on a different single model.** Fine for the first draft with the model named in every
caption; not sufficient for "the effect is architectural, not model-specific."

### 7.3 Metrics not logged

- **`alive_at_cap` per rollout** — needed to make the §2a "alive at turn 150" column real
  rather than `1 − death%`.
- **Uncapped depth** — the dlvl-6 episode termination censors every Exp-1 mean.
- **Seed variance (σ) per architecture** — computable from per-seed depth vectors once the
  underlying runs are un-contaminated.
- **Map coverage %** — recoverable from traces; scout reward is only a proxy.
- **Per-encoding token cost** — only raw-vs-compacted measured; and output tokens per turn
  were never separated.
- **Survival rate (>500 steps)** — **deprioritized.** Survival is a poor capability signal:
  autoascend survives indefinitely while accomplishing nothing. Death % is enough.

### 7.4 The counter-result that must stay visible

"LLM agents cannot navigate NetHack" is **not a safe claim** and this document should not be
read as supporting it. After 11 harness bugs and a render-only `reveal_map` were fixed, 6/6
seeds were beaten to Gehennom with no descend skill and 3–4 genuine descents each; at the
bottom of the dungeon the invocation ritual went 2/2. **The binding constraint is roughly half
harness quality and half agent capability**, and every pre-fix measurement understated the
agents.

---

## 8 · Run queue

Ordered by (unblocks-a-table × cheapness). Runs 1–2 are free.

| # | Run | Unblocks | Cost |
|:-:|---|---|---|
| 1 | Obtain the real BALROG table / official scorer | every BALROG column; all external comparisons | none — sourcing |
| 2 | Re-run the free scripted navigation ceiling on the fixed harness + de-secret engine | §5c; the standing control for Exps 3–4 | free (no API) |
| 3 | **Uncap episode depth** and re-run the three n=24 encoding cells | §2 censoring defect — gates every Exp-1 number | 3 × 24 |
| 4 | **Delayed / on-demand map × {ASCII, JSON}** | §2e — the current final variant, entirely unmeasured | 4 cells |
| 5 | **JSON cell-content ablation** — glyph+knowledge, seen/unseen, lit/dark, visited/unvisited, reachability | §2c — leading explanation for why JSON underperforms | 5–6 arms |
| 6 | Full encoding ranking at n=24, uncompressed and uncapped, incl. TOON / IMG / IMG_TTY (+ an image-resolution arm) | §2a — merges the two power blocks into one publishable table | 6+ × 24 |
| 7 | Re-run Voyager / RLM / Go-Explore at 20 seeds on the fixed harness | §4e retraction; σ per architecture | 60 runs |
| 8 | **Bounded delegation**: cap skill/code calls at ~8 game turns with explicit `continue` | §3c — separates reasoning throughput from budget absorption | 2–3 cells |
| 9 | Skill vs code at **matched turn budget** across 3 models | §3c — currently GLM-5.2 only and budget-confounded | 6 cells |
| 10 | **Encoding × algorithm square** — {skill, code} × 6 encodings | §2f — the independence-vs-co-dependence claim | 12 cells |
| 11 | Memory ablation — journal / belief-state / summarize-and-reset / none | §2d — never compared head to head | 4 arms |
| 12 | Wiki on/off ablation | §4b — is knowledge access load-bearing at all? | 2 arms |
| 13 | Matched NetPlay arm (`explore_level` reimplemented in-harness) + open-model published numbers if they exist | §2g, §7.1 | build + 24 |
| 14 | Fix `move_to` to gate with `pass_monsters`, re-run the deep segment | §5e — should lift seeds 20/22/23/24/25 | small |
| 15 | XP-before-descent policy knob (agents die at XL ~1.4) | §2g — highest-leverage untried tactic | 24 |
