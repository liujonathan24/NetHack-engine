# environments/nethack/nethack_harness/prompt/map_encoders.py
"""Serialize the canonical MapModel as JSON or TOON, at a selectable detail.

`full`  -> rich entity attributes + the RLE grid.
`minimal` -> entity kind/coord/description only; no grid, no rich attrs.
Both project the SAME model, so JSON and TOON cannot diverge.
"""
from __future__ import annotations

import json
from typing import Any

_RICH_FIELDS = ("species", "is_pet", "obj_class", "detail")


def _entity_dict(e: Any, detail: str) -> dict:
    d = {"kind": e.kind, "x": e.x, "y": e.y, "desc": e.description}
    if detail == "full":
        for f in _RICH_FIELDS:
            v = getattr(e, f, None)
            if v is not None:
                d[f] = v
    return d


def _model_dict(model: Any, detail: str) -> dict:
    d = {
        "player": list(model.player) if model.player else None,
        "entities": [_entity_dict(e, detail) for e in model.entities],
    }
    if detail == "full":
        d["grid"] = model.grid
    return d


# ---------- sub-experiment 1b: per-cell SPATIAL / EXPLORATION layers ----------
#
# The identification-rich JSON base (kind/coord/desc/species/door/stair + the
# RLE terrain grid) is enriched with optional, strictly-gated 0/1 mask layers:
#   seen_grid    -- 1 = tile revealed (non-space char), 0 = unseen
#   visited_grid -- 1 = hero has stood on this tile this level
#   reach_grid   -- 1 = walkable-reachable from the hero's tile right now
# Each mask reuses the terrain RLE format (`_rle_grid`) so a 21x79 layer stays
# bounded (a single run "0x79" per empty row). Only the enabled attributes are
# emitted; a disabled one is ABSENT from the JSON entirely.

_CELL_ATTRS = ("seen", "visited", "reach")


def build_cell_layers(chars, player, visited_xy, cell_schema) -> dict:
    """Return ``{layer_name: rle_string}`` for each enabled cell_schema attr.

    ``chars`` is the NLE 21x79 char grid, ``player`` is (x, y), ``visited_xy``
    is the set of (x, y) tiles the hero has stood on this level, ``cell_schema``
    is a set/collection drawn from ``_CELL_ATTRS``. Each layer is a 0/1 mask
    RLE-encoded exactly like the terrain grid.
    """
    import numpy as np
    from nethack_core.map_model import _rle_grid

    schema = set(cell_schema or ())
    chars = np.asarray(chars)
    h, w = chars.shape
    layers: dict = {}

    if "seen" in schema:
        # Revealed iff the rendered char is not the unseen/rock sentinel (space).
        seen = (chars != ord(" ")).astype(int)
        layers["seen_grid"] = _rle_grid(seen)

    if "visited" in schema:
        vis = np.zeros((h, w), dtype=int)
        for (x, y) in visited_xy or ():
            if 0 <= y < h and 0 <= x < w:
                vis[y, x] = 1
        layers["visited_grid"] = _rle_grid(vis)

    if "reach" in schema:
        from nethack_harness.navigation.pathfinding import reachable_set
        reach = np.zeros((h, w), dtype=int)
        for (x, y) in reachable_set(chars, tuple(player)):
            if 0 <= y < h and 0 <= x < w:
                reach[y, x] = 1
        layers["reach_grid"] = _rle_grid(reach)

    return layers


def json_encode(model: Any, *, detail: str = "full", cell_layers: dict | None = None) -> str:
    d = _model_dict(model, detail)
    if cell_layers:
        d.update(cell_layers)
    return json.dumps(d, separators=(",", ":"))


def toon_encode(model: Any, *, detail: str = "full") -> str:
    """Token-frugal line-oriented encoding of the same model.

    Format (deterministic):
        @ x,y
        <kind> x,y desc[ k=v ...]
        ...
        grid: <rle>            # full detail only
    """
    lines = []
    if model.player:
        lines.append(f"@ {model.player[0]},{model.player[1]}")
    for e in model.entities:
        parts = [e.kind, f"{e.x},{e.y}", e.description]
        if detail == "full":
            for f in _RICH_FIELDS:
                v = getattr(e, f, None)
                if v is not None:
                    parts.append(f"{f}={v}")
        lines.append(" ".join(str(p) for p in parts))
    if detail == "full":
        lines.append(f"grid: {model.grid}")
    return "\n".join(lines)
