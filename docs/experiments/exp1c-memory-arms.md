# Sub-experiment 1c — memory / journalling ablation (ready-to-run arms)

Head-to-head of four memory conditions, holding the **Exp-1 spine** fixed:
netplay skill surface, `full_nle`, Valkyrie (`Val-hum-neu-fem`), seeds 0–15
(`n=16`), **B0** encoding, Gemini 3 Flash via the Prime **team** endpoint.

Only the memory mechanism varies across arms. Two pieces of code make a clean
ablation possible (both landed on `exp/encoding-sweep`):

- **`sub_lm_model`** (`load_environment`): builds a real in-process Prime-backed
  `PrimeSubLM` from a model id so belief-state distillation calls the model
  (`nethack_harness/tools/code_mode.py:PrimeSubLM`) instead of the deterministic
  `OfflineSubLM` status-snapshot stub. Without it, belief notes are just an
  HP/AC/Dlvl snapshot (`helpers.py:_maybe_belief_state_summary`).
- **`pin_objective_on_setup`** (`load_environment` → env): when `False`, the tier
  description is not pre-pinned as the journal objective at setup
  (`nethack.py`, formerly the unconditional `journal.pin_objective(spec.description)`).
  With journal tools excluded and `belief_state_interval=0`, the `Journal` stays
  empty → `format_observation_as_chat` suppresses the `=== JOURNAL ===` block
  (`prompt/rendering.py:636`, gated on `journal.is_empty()`). This is the only
  way to reach a *truly* no-memory arm.

## The four arms (`-a` JSON, drop into `vf-eval nethack -a '<...>'`)

Shared spine (every arm): `"tier":"full_nle"`, `"variant":"B0"`,
`"interface":"skill"`, `"character":"Val-hum-neu-fem"`,
`"explicit_seeds":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]`. Model resolved from
`configs/endpoints.toml` (prime-team block) with `-m google/gemini-3-flash-preview`
— do **not** pass `-p prime` (that drops the `X-Prime-Team-ID` billing header).

### 1. journal-only
Journal tools available (netplay includes `add_note`/`recall`/`pin_objective`);
no periodic belief distillation; no chat reset.
```json
{"tier":"full_nle","variant":"B0","interface":"skill","character":"Val-hum-neu-fem","skill_set":"netplay","belief_state_interval":0,"summarize_and_reset":false,"explicit_seeds":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
```

### 2. + belief-state
journal-only **plus** a real-LM belief note every 25 turns.
```json
{"tier":"full_nle","variant":"B0","interface":"skill","character":"Val-hum-neu-fem","skill_set":"netplay","belief_state_interval":25,"sub_lm_model":"google/gemini-3-flash-preview","summarize_and_reset":false,"explicit_seeds":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
```

### 3. summarize-and-reset (variant R)
belief-state **plus** hard-drop of every chat turn older than the most recent
`belief_state:tN` checkpoint — "the belief state IS the memory, chat is disposable."
```json
{"tier":"full_nle","variant":"B0","interface":"skill","character":"Val-hum-neu-fem","skill_set":"netplay","belief_state_interval":25,"sub_lm_model":"google/gemini-3-flash-preview","summarize_and_reset":true,"explicit_seeds":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
```

### 4. no-memory
No journal tools (netplay **minus** `add_note`/`recall`/`pin_objective`), no
belief distillation, no setup objective pin → journal never renders. Small
full-history window so old turns are dropped rather than re-summarized.
```json
{"tier":"full_nle","variant":"B0","interface":"skill","character":"Val-hum-neu-fem","skill_set":"move_to,explore_and_descend,attack,throw,descend,search,pickup,engrave_elbereth,pray,eat,quaff,read,kick,wiki_lookup,wiki_search","belief_state_interval":0,"pin_objective_on_setup":false,"history_keep_full":2,"explicit_seeds":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
```

The `skill_set` above is exactly the netplay whitelist
(`helpers.py:_build_skill_adapter_callables`, `elif skill_set == "netplay"`)
with the three journal tools removed:

```
netplay = move_to, explore_and_descend, attack, throw, descend, search, pickup,
          engrave_elbereth, pray, eat, quaff, read, kick,
          add_note, recall, pin_objective, wiki_lookup, wiki_search
no-memory = netplay − {add_note, recall, pin_objective}
```

## Smoke verification

`tools/exp1c_memory/_smoke.sh <arm> <outdir>` runs one seed, ~6 turns, B0, via
the Prime team endpoint, then greps the trace (`rendered_user_content`). For the
belief-state arms it sets `belief_state_interval=3` so a note fires inside 6
turns. Expected per arm:

- **journal-only**: `=== JOURNAL ===` block with an `Objective:` line.
- **+belief-state / summarize-and-reset**: a `belief_state:t*` note that is a real
  LM sentence (not `[offline-summary]` / the HP/AC/Dlvl status snapshot).
- **no-memory**: NO `=== JOURNAL ===` block, and the tool schema exposes no
  `add_note`/`recall`/`pin_objective`.

`n=16` sweeps are intentionally **not** run here — build + smoke only.
