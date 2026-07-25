# Sub-experiment 1b — JSON cell-content ablation (ready-to-run arms)

The **JSON** encoding underperformed in Exp 1. This sub-experiment tests whether
enriching each map cell with **SPATIAL / EXPLORATION** metadata — on top of the
current identification-rich JSON base — closes the gap.

The JSON base already carries per-entity *identification* attributes
(`kind` / `x,y` / `desc` / `species` / `door` state / `stair` direction) plus one
RLE terrain `grid`. Those stay fixed. We ablate exactly **three** per-cell
attributes, each added as one compact RLE **0/1 mask layer**:

| attribute | JSON key        | meaning (1 = …)                                  | source |
|-----------|-----------------|--------------------------------------------------|--------|
| `seen`    | `seen_grid`     | tile revealed (rendered char ≠ space)            | `raw_obs.chars` |
| `visited` | `visited_grid`  | hero has stood on this tile (this dlvl)          | `state["_visited_tiles"]` |
| `reach`   | `reach_grid`    | walkable-reachable from the hero's tile *now*    | `pathfinding.reachable_set` |

`lit/dark` is **DEFERRED** — there is no clean per-tile light data source in the
current obs, so shipping it would mean a fabricated layer.

Each mask reuses the terrain RLE format (`nethack_core/map_model.py:_rle_grid`),
so a 21×79 layer stays bounded (a fully-empty row is a single run, e.g. `0x79`).
A **disabled** attribute is **absent** from the JSON entirely — the layers are
strictly gated by `cell_schema`, so the four arms differ only in which keys the
map object contains.

## Code (all on `exp/encoding-sweep`, harness-layer only; `nethack_core` untouched)

- **`cell_schema`** (`load_environment` → env `__init__`, `nethack.py`): a set/list
  or `"seen+visited"`-style string of enabled attrs, normalized by
  `_normalize_cell_schema` (unknown tokens dropped). Written to
  `state["cell_schema"]` in `setup_state` next to `state["map_detail"]`.
  Empty/None = base JSON, byte-identical to before.
- **Visited tracker** (`nethack.py`): `state["_visited_tiles"]` is
  `dlvl -> set[(x,y)]`; seeded with the hero's start tile in `setup_state` and
  appended in the per-step loop next to `scout_tiles_seen` (same
  `max_dlvl_reached` coordinate frame).
- **JSON template** (`prompt/prompt_spec.py:_structured_map_template`): when
  `state["cell_schema"]` is non-empty, computes the enabled layers via
  `map_encoders.build_cell_layers(chars, player, visited_xy, cell_schema)` and
  merges them into the JSON object (`json_encode(..., cell_layers=...)`). TOON is
  unaffected.
- Selected with `variant="JSON"` + a `cell_schema` arg — no new variant names.

## The five arms (`-a` JSON, drop into `vf-eval nethack -a '<...>'`)

Shared spine (every arm), matching the Exp-1 encoding sweep: `"tier":"full_nle"`,
`"variant":"JSON"`, `"interface":"skill"`, `"skill_set":"netplay"`,
`"character":"Val-hum-neu-fem"`, `"map_detail":"full"`,
`"explicit_seeds":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]`. Model resolved from
`configs/endpoints.toml` (prime-team block) with `-m google/gemini-3-flash-preview`
— do **not** pass `-p prime` (that drops the `X-Prime-Team-ID` billing header).

### 1. base (identification-rich JSON, no cell metadata)
```json
{"tier":"full_nle","variant":"JSON","interface":"skill","skill_set":"netplay","character":"Val-hum-neu-fem","map_detail":"full","cell_schema":[],"explicit_seeds":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
```

### 2. + seen
```json
{"tier":"full_nle","variant":"JSON","interface":"skill","skill_set":"netplay","character":"Val-hum-neu-fem","map_detail":"full","cell_schema":["seen"],"explicit_seeds":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
```

### 3. + visited
```json
{"tier":"full_nle","variant":"JSON","interface":"skill","skill_set":"netplay","character":"Val-hum-neu-fem","map_detail":"full","cell_schema":["visited"],"explicit_seeds":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
```

### 4. + reach
```json
{"tier":"full_nle","variant":"JSON","interface":"skill","skill_set":"netplay","character":"Val-hum-neu-fem","map_detail":"full","cell_schema":["reach"],"explicit_seeds":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
```

### 5. + all (seen + visited + reach)
```json
{"tier":"full_nle","variant":"JSON","interface":"skill","skill_set":"netplay","character":"Val-hum-neu-fem","map_detail":"full","cell_schema":["seen","visited","reach"],"explicit_seeds":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
```

## Smoke verification

`tools/encoding_eval/_smoke_1b.sh <cell_schema> <outdir>` runs one seed, ~6
turns, `variant=JSON`, via the Prime team endpoint, then the trace's
`rendered_user_content` is inspected. Expected per arm:

- **base** (`cell_schema=[]`): the JSON map object has NONE of
  `seen_grid`/`visited_grid`/`reach_grid`.
- **+seen**: `seen_grid` present, and NOT all-`1` (some unseen `0` runs exist
  early). `visited_grid`/`reach_grid` absent.
- **+visited**: `visited_grid` present and non-trivial (grows as the hero moves).
  `seen_grid`/`reach_grid` absent.
- **+reach**: `reach_grid` present and non-trivial. `seen_grid`/`visited_grid`
  absent.
- **+all**: all three present; the JSON map block still `json.loads`-parses.

`n=16` sweeps are intentionally **not** run here — build + smoke only.
