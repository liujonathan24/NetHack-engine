# Sub-experiment 1d — observation delivery / timing (ready-to-run arms)

Head-to-head of how the map is *delivered* to the agent, holding the **Exp-1
spine** fixed: netplay skill surface, `full_nle`, Valkyrie (`Val-hum-neu-fem`),
seeds 0–15 (`n=16`), Gemini 3 Flash via the Prime **team** endpoint. Only the
map-delivery mechanism varies; **action feedback rides `prefix_parts` every turn
in all arms** (`nethack.py`, `prefix_parts`), independent of the map.

Two new delivery variants landed on `exp/encoding-sweep`
(`prompt/prompt_spec.py`):

- **Delayed / on-demand map (`DM`, `DM_JSON`).** The FULL map is re-sent only
  when it **materially changed** (dungeon level or a cheap map-view fingerprint,
  `_delayed_map_fingerprint`) **or** the agent forced a refresh via
  `request_map` / `reveal` (`state["_force_map"]`, set in `env_response`).
  Otherwise a one-line placeholder stands in:
  `=== MAP (unchanged; call request_map to refresh) ===`. `DM` renders the full
  map as the **B0 ASCII** grid; `DM_JSON` renders it as the **JSON** map — the
  delivery policy composes with either encoding (`_delayed_map_template(base)`).
- **Bounding-box on-demand (`BBOX`).** The map is **hidden** every turn
  (`=== MAP (hidden; call reveal(x1,y1,x2,y2) to view a region) ===`); the agent
  sees it only by calling `reveal(x1,y1,x2,y2)`, which returns that ASCII
  sub-rectangle as tool feedback and **consumes no game turn**
  (`_bbox_template`, `skills.py:reveal`).

Both new tools are info-only (empty `actions`, no NLE step) and are exposed
through an explicit comma-`skill_set` = the netplay tools **plus** `reveal,request_map` (these are deliberately NOT in the default netplay set, so
includes `reveal`, `request_map`).

## The arms (`-a` JSON, drop into `vf-eval nethack -a '<...>'`)

Shared spine (every arm): `"tier":"full_nle"`, `"interface":"skill"`,
`"character":"Val-hum-neu-fem"`, `"skill_set":"move_to,explore_and_descend,attack,throw,descend,search,pickup,engrave_elbereth,pray,eat,quaff,read,kick,add_note,recall,pin_objective,wiki_lookup,wiki_search,reveal,request_map"`,
`"explicit_seeds":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]`. Model resolved from
`configs/endpoints.toml` (prime-team block) with `-m google/gemini-3-flash-preview`
— do **not** pass `-p prime` (that drops the `X-Prime-Team-ID` billing header).

### Baselines (already shipped — for reference)
`variant":"B0"` (full ASCII map every turn) and `"variant":"JSON"` (full JSON
map every turn) are the every-turn deliveries these arms are measured against.

### 1. delayed-map × B0
Full **ASCII** map only on material change or `request_map`; placeholder
otherwise.
```json
{"tier":"full_nle","variant":"DM","interface":"skill","character":"Val-hum-neu-fem","skill_set":"move_to,explore_and_descend,attack,throw,descend,search,pickup,engrave_elbereth,pray,eat,quaff,read,kick,add_note,recall,pin_objective,wiki_lookup,wiki_search,reveal,request_map","explicit_seeds":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
```

### 2. delayed-map × JSON
Full **JSON** map only on material change or `request_map`; placeholder
otherwise.
```json
{"tier":"full_nle","variant":"DM_JSON","interface":"skill","character":"Val-hum-neu-fem","skill_set":"move_to,explore_and_descend,attack,throw,descend,search,pickup,engrave_elbereth,pray,eat,quaff,read,kick,add_note,recall,pin_objective,wiki_lookup,wiki_search,reveal,request_map","explicit_seeds":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
```

### 3. bounding-box on-demand
Map hidden; only `reveal(x1,y1,x2,y2)` exposes a region (no game turn consumed).
```json
{"tier":"full_nle","variant":"BBOX","interface":"skill","character":"Val-hum-neu-fem","skill_set":"move_to,explore_and_descend,attack,throw,descend,search,pickup,engrave_elbereth,pray,eat,quaff,read,kick,add_note,recall,pin_objective,wiki_lookup,wiki_search,reveal,request_map","explicit_seeds":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
```

## In-process verification

`tools/exp1d_obs/verify_1d.py` builds each env in-process and drives synthetic
tool calls (no LM, no Prime — cannot hang):

```
PYTHONPATH=.:environments/nethack .venv/bin/python tools/exp1d_obs/verify_1d.py
```

Asserts, from the returned user message:

- **DM / DM_JSON**: turn 1 shows the full `=== MAP ===` (`=== MAP (JSON) ===`);
  a no-material-change turn (`add_note`) shows `=== MAP (unchanged...`; after
  `request_map` the full map returns. Action feedback present every turn.
- **BBOX**: normal turns carry only the hidden-map placeholder (no inline grid);
  `reveal(...)` returns an ASCII crop and consumes **no** game turn (`dlvl` and
  in-game `time` unchanged before/after).

`n=16` sweeps are intentionally **not** run here — build + verify only.
