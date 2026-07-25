"""The prompt factory.

A :class:`PromptSpec` bundles the four things that define how a NetHack rollout
talks to the model:

  1. **observation form + processors** — :class:`ObsSpec` (``ascii`` / ``img``
     plus the per-turn obs blocks toggled via ``setup_flags``).
  2. **system prompt** — the instruction block.
  3. **per-turn template** — ``turn_template(structured, journal, state, *,
     compact, journal_max_chars) -> str | list`` renders the user message each
     turn (f-string-style assembly of the observation).
  4. **tools** — :class:`ToolSpec` (which skill set + any extra tool factories).

:func:`build_prompt` is the single constructor; :data:`VARIANT_REGISTRY` defines
every shipped variant as a composition of the rendering functions in
``nethack_harness.prompt.rendering``. ``NetHackVerifiersEnv`` holds one resolved
``PromptSpec`` and dispatches through it instead of scattering ``self.variant ==``
checks across the rollout loop.

This module reproduces the *exact* behaviour of the legacy ``self.variant``
branches — it changes control flow (registry dispatch), not rendered bytes.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from nethack_harness.prompt.rendering import (
    SYSTEM_PROMPT,
    format_observation_as_chat,
    _format_obs_balrog,
    _format_obs_glyphbox,
    _format_obs_summarize_reset,
)
from nethack_harness.helpers import (
    _refinement_directive,
    _ch_build_window,
    _drop_before_last_belief,  # noqa: F401  (kept for symmetry / external use)
    _ch_inject_system,
    _make_run_macro_adapter,
)

# ---------- the four-part spec ----------


@dataclass(frozen=True)
class ObsSpec:
    """Observation form + per-turn obs-block processors.

    ``mode`` selects how the observation is presented (``ascii`` text grid or a
    rendered ``img``). ``setup_flags`` are the per-rollout state flags that gate
    the optional obs blocks (descent-salience, E1 frontier surface, E2 paint);
    ``setup_state`` writes them and the rendering functions read them.
    """

    mode: str = "ascii"
    setup_flags: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ToolSpec:
    """Which tools the agent gets.

    ``skill_set`` is ``None`` to honour the caller's ``skill_set`` kwarg (the
    default), or a concrete set name to force it. ``extra_tools`` is a tuple of
    zero-arg factories that each return one extra tool callable (e.g. the CH
    ``run_macro`` adapter).
    """

    skill_set: Optional[str] = None
    extra_tools: tuple = ()


@dataclass(frozen=True)
class PromptSpec:
    """A fully-specified harness prompt: the four parts + cross-cutting hooks."""

    name: str
    system_prompt: str
    obs: ObsSpec
    turn_template: Callable[..., Any]
    tools: ToolSpec
    # Cross-cutting hooks discovered in the variant inventory. Each is a tuple
    # so a variant can compose several. turn_hooks mutate the per-turn prefix
    # (P refinement directive, CH refiner + sub-agents); history_transforms
    # rewrite the chat message list in get_prompt_messages (CH system inject).
    turn_hooks: tuple = ()
    history_transforms: tuple = ()


def build_prompt(
    *,
    name: str,
    system_prompt: str,
    obs: ObsSpec,
    turn_template: Callable[..., Any],
    tools: ToolSpec,
    turn_hooks: tuple = (),
    history_transforms: tuple = (),
) -> PromptSpec:
    """Construct a :class:`PromptSpec` from the four parts (+ optional hooks).

    This is the one place a harness author assembles a prompt: pick an
    observation form, a system prompt, a per-turn template, and a tool set.
    """
    return PromptSpec(
        name=name,
        system_prompt=system_prompt,
        obs=obs,
        turn_template=turn_template,
        tools=tools,
        turn_hooks=turn_hooks,
        history_transforms=history_transforms,
    )


# ---------- per-turn templates (ascii) ----------


def _canonical_template(structured, journal, state, *, compact, journal_max_chars):
    """The default per-turn renderer (B1 and most variants)."""
    return format_observation_as_chat(
        structured, journal, state=state,
        compact=compact, journal_max_chars=journal_max_chars,
    )


def _formatter_template(formatter):
    """Adapt a legacy ``_format_obs_*`` formatter to the turn_template signature.

    The legacy formatters take ``(structured, journal, state, journal_max_chars)``
    and apply their own compaction policy, so ``compact`` is ignored here — this
    matches the pre-refactor call sites exactly.
    """

    def _render(structured, journal, state, *, compact, journal_max_chars):
        return formatter(structured, journal, state, journal_max_chars)

    return _render


def _image_template(render_name):
    """Per-turn template that returns a multimodal [image_url, text] content list.

    ``render_name`` selects the strict render path: "glyph" → GlyphMapper tiles,
    "tty" → tty-text raster. The text block is journal + status + inventory only
    (the image is the sole spatial channel).
    """

    def _render(structured, journal, state, *, compact, journal_max_chars):
        from nethack_harness.prompt.image_render import (
            glyphs_to_png_b64, tty_to_png_b64, to_data_uri,
        )

        raw = state["raw_obs"]
        b64 = glyphs_to_png_b64(raw) if render_name == "glyph" else tty_to_png_b64(raw)
        text = format_observation_as_chat(
            structured, journal, state,
            compact=compact, journal_max_chars=journal_max_chars,
            include_map=False, include_local=False,
        )
        return [
            {"type": "image_url", "image_url": {"url": to_data_uri(b64)}},
            {"type": "text", "text": text},
        ]

    return _render


def _structured_map_template(fmt):
    """Per-turn template emitting a structured-text map (JSON or TOON).

    ``fmt`` is "json" or "toon". The map_detail level is read from
    ``state["map_detail"]`` (default "full"). The status/inventory block is
    appended via the canonical formatter with the ASCII map gated off.

    Sub-experiment 1b (JSON only): when ``state["cell_schema"]`` names any of
    {"seen","visited","reach"}, the JSON base is enriched with a COMPACT 0/1
    mask layer per enabled attribute (``seen_grid`` / ``visited_grid`` /
    ``reach_grid``), computed here from ``state`` and RLE-encoded like the
    terrain grid. A disabled attribute is ABSENT from the JSON. TOON is
    unaffected (empty cell_schema leaves JSON bit-identical to before).
    """

    def _render(structured, journal, state, *, compact, journal_max_chars):
        from nethack_core import build_map_model
        from nethack_harness.prompt.map_encoders import (
            json_encode, toon_encode, build_cell_layers,
        )

        detail = state.get("map_detail", "full")
        model = build_map_model(state["raw_obs"])
        if fmt == "json":
            cell_schema = state.get("cell_schema") or set()
            cell_layers = None
            if cell_schema:
                import numpy as np
                raw = state["raw_obs"]
                bl = np.asarray(raw.blstats)
                player = (int(bl[0]), int(bl[1]))
                # Key visited tiles by the hero's ACTUAL depth (blstats[12]) so
                # this read matches how the tracker files them (max_dlvl_reached
                # lags a descent turn — see setup_state/env_response).
                dlvl_key = int(bl[12])
                visited_all = state.get("_visited_tiles") or {}
                visited_xy = visited_all.get(dlvl_key, set())
                cell_layers = build_cell_layers(
                    raw.chars, player, visited_xy, cell_schema,
                )
            map_text = json_encode(model, detail=detail, cell_layers=cell_layers)
        else:
            map_text = toon_encode(model, detail=detail)
        status = format_observation_as_chat(
            structured, journal, state, compact=compact,
            journal_max_chars=journal_max_chars,
            include_map=False, include_local=False,
        )
        return f"=== MAP ({fmt.upper()}) ===\n{map_text}\n\n{status}"

    return _render


# ---------- sub-experiment 1d: observation-DELIVERY variants ----------

# Placeholder that stands in for the map block on a delayed-map turn where the
# map is elided (no material change and no forced refresh).
_DELAYED_MAP_PLACEHOLDER = "=== MAP (unchanged; call request_map to refresh) ==="
# Placeholder shown by the bounding-box variant, where the map is hidden and
# only reachable through the reveal() tool.
_BBOX_MAP_PLACEHOLDER = "=== MAP (hidden; call reveal(x1,y1,x2,y2) to view a region) ==="


def _delayed_map_fingerprint(structured, state):
    """Cheap material-change signal for the delayed-map variants.

    Any move or newly-seen tile changes ``structured.map_view`` (the @ position
    and revealed glyphs are baked into it), so the map is re-sent; a no-op turn
    (info-only skill, blocked action, in-place search that reveals nothing)
    leaves it byte-identical and the map is elided. Also keyed on the dungeon
    level so a stair transition always forces a refresh. Mirrors the
    journal/inventory fingerprint pattern in rendering.py.
    """
    try:
        map_view = getattr(structured, "map_view", "") or ""
    except Exception:
        map_view = ""
    dlvl = state.get("max_dlvl_reached", 1) if state is not None else 1
    return (dlvl, hash(map_view))


def _delayed_map_template(base_fmt):
    """Delayed / on-demand map delivery (variants DM, DM_JSON).

    The FULL map is re-sent only when it materially changed (dlvl or map
    fingerprint) OR the agent forced a refresh via request_map/reveal
    (``state['_force_map']``, set in env_response). Otherwise a one-line
    placeholder stands in for the map block. Action feedback + status render
    unchanged every turn (they ride the caller's prefix_parts / the status
    block here), independent of whether the map is shown.

    ``base_fmt`` selects the encoding the FULL map uses so delayed delivery
    composes with either baseline: "ascii" (B0-style grid via
    format_observation_as_chat) or "json" (structured JSON map via the same
    builder the JSON variant uses, including any 1b cell layers).
    """
    _json_full = _structured_map_template("json")

    def _render(structured, journal, state, *, compact, journal_max_chars):
        force = bool(state.pop("_force_map", False)) if state is not None else False
        cur_fp = _delayed_map_fingerprint(structured, state)
        prev_fp = state.get("_delayed_map_fp") if state is not None else None
        show_map = force or prev_fp is None or cur_fp != prev_fp
        if state is not None:
            state["_delayed_map_fp"] = cur_fp

        if base_fmt == "json":
            if show_map:
                return _json_full(
                    structured, journal, state,
                    compact=compact, journal_max_chars=journal_max_chars,
                )
            status = format_observation_as_chat(
                structured, journal, state, compact=compact,
                journal_max_chars=journal_max_chars,
                include_map=False, include_local=False,
            )
            return f"{_DELAYED_MAP_PLACEHOLDER}\n\n{status}"

        # ascii
        if show_map:
            return format_observation_as_chat(
                structured, journal, state,
                compact=compact, journal_max_chars=journal_max_chars,
            )
        text = format_observation_as_chat(
            structured, journal, state, compact=compact,
            journal_max_chars=journal_max_chars, include_map=False,
        )
        return _splice_placeholder(text, _DELAYED_MAP_PLACEHOLDER)

    return _render


def _bbox_template(structured, journal, state, *, compact, journal_max_chars):
    """Bounding-box on-demand map (variant BBOX).

    The map is never rendered inline; the agent sees it only by calling
    ``reveal(x1,y1,x2,y2)``, which returns the requested sub-rectangle as tool
    feedback (no NLE step). Everything else (journal, status, inventory,
    under-player) renders normally.
    """
    text = format_observation_as_chat(
        structured, journal, state, compact=compact,
        journal_max_chars=journal_max_chars, include_map=False,
    )
    return _splice_placeholder(text, _BBOX_MAP_PLACEHOLDER)


def _splice_placeholder(text: str, placeholder: str) -> str:
    """Insert a map placeholder where the (omitted) map block would have been.

    The canonical renderer puts the map immediately before ``=== STATUS ===``;
    with include_map=False that block is absent, so we splice the placeholder in
    just ahead of STATUS to keep the layout familiar. Falls back to prepending.
    """
    marker = "=== STATUS ==="
    if marker in text:
        return text.replace(marker, f"{placeholder}\n\n{marker}", 1)
    return f"{placeholder}\n\n{text}"


# ---------- cross-cutting hooks (verbatim from the legacy env_response) ----------


def _p_refinement_hook(env_self, state, prefix_parts):
    """Variant P: inject a self-refinement directive every ``refine_interval``."""
    if (
        env_self.refine_interval > 0
        and state.get("turn_count", 0) > 0
        and state["turn_count"] % env_self.refine_interval == 0
        and not state.get("_refine_emitted_this_turn")
    ):
        prefix_parts.append(_refinement_directive(state))
        state["_refine_emitted_this_turn"] = True
    else:
        state["_refine_emitted_this_turn"] = False


def _ch_refiner_hook(env_self, state, prefix_parts):
    """Variant CH: run the configured Refiner and apply its CRUD edits."""
    if (
        env_self.refiner is not None
        and env_self.refine_interval > 0
        and state.get("turn_count", 0) > 0
        and state["turn_count"] % env_self.refine_interval == 0
        and not state.get("_ch_refined_this_turn")
    ):
        try:
            from nethack_harness.refiner import snapshot_components, apply_edits
            window = _ch_build_window(state.get("trajectory") or [], n_turns=env_self.refine_interval)
            edits = env_self.refiner.refine(
                window=window,
                components=snapshot_components(state),
            )
            applied = apply_edits(state, edits)
            state["_ch_last_edits"] = edits.to_trace_dict()
            state["_ch_last_applied"] = applied
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("CH refiner failed: %s", e)
        state["_ch_refined_this_turn"] = True
    else:
        state["_ch_refined_this_turn"] = False
        state.pop("_ch_last_edits", None)


def _ch_subagents_hook(env_self, state, prefix_parts):
    """Variant CH: prepend any sub-agent directive whose trigger fires (cap 3)."""
    if not state.get("_ch_subagents"):
        return
    from nethack_harness.refiner import trigger_fires
    fired = []
    for name, spec in state["_ch_subagents"].items():
        if trigger_fires(spec.get("trigger", ""), state.get("structured_obs")):
            fired.append(f"[subagent:{name}] {spec.get('text', '')}")
            if len(fired) >= 3:
                break
    if fired:
        prefix_parts.extend(fired)


def _ch_history_inject(env_self, messages, state):
    """Variant CH: append the Refiner addendum + macro list to the system msg."""
    return _ch_inject_system(messages, state)


# The refiner-machinery bundle that the CH variant wires onto its spec: the
# per-turn refiner + sub-agent hooks, the system-inject history transform, and
# the run_macro tool adapter. Factored out so it can be attached to ANY obs
# variant (e.g. JSON) when refine=True, decoupling the teacher-refiner from the
# CH observation format.
_CH_REFINER_TURN_HOOKS = (_ch_refiner_hook, _ch_subagents_hook)
_CH_REFINER_HISTORY_TRANSFORMS = (_ch_history_inject,)
_CH_REFINER_EXTRA_TOOLS = (_make_run_macro_adapter,)


def attach_refiner(spec: "PromptSpec") -> "PromptSpec":
    """Return ``spec`` with the CH teacher-refiner machinery attached.

    Adds the refiner + sub-agent turn-hooks, the ``_ch_history_inject`` system
    addendum, and the ``run_macro`` tool — the exact bundle the CH variant uses
    — onto whatever obs/template ``spec`` already carries. Idempotent: hooks /
    tools already present (e.g. the CH spec built them in directly) are not
    duplicated.
    """
    turn_hooks = tuple(spec.turn_hooks) + tuple(
        h for h in _CH_REFINER_TURN_HOOKS if h not in spec.turn_hooks
    )
    history_transforms = tuple(spec.history_transforms) + tuple(
        t for t in _CH_REFINER_HISTORY_TRANSFORMS if t not in spec.history_transforms
    )
    existing_tools = tuple(spec.tools.extra_tools)
    extra_tools = existing_tools + tuple(
        mk for mk in _CH_REFINER_EXTRA_TOOLS if mk not in existing_tools
    )
    new_tools = dataclasses.replace(spec.tools, extra_tools=extra_tools)
    return dataclasses.replace(
        spec,
        turn_hooks=turn_hooks,
        history_transforms=history_transforms,
        tools=new_tools,
    )


# ---------- the registry ----------

_CANONICAL_OBS = ObsSpec(mode="ascii")
_FULL_TOOLS = ToolSpec()


def _spec(name, *, system_prompt, obs=_CANONICAL_OBS, turn_template=_canonical_template,
          tools=_FULL_TOOLS, turn_hooks=(), history_transforms=()):
    return build_prompt(
        name=name, system_prompt=system_prompt, obs=obs,
        turn_template=turn_template, tools=tools,
        turn_hooks=turn_hooks, history_transforms=history_transforms,
    )


def _build_registry(system_prompt: str) -> dict:
    """Build the variant→PromptSpec table against a given system prompt."""
    canonical = lambda name, **kw: _spec(name, system_prompt=system_prompt, **kw)  # noqa: E731
    return {
        # Default + close cousins: canonical render, full tools.
        "B1": canonical("B1"),
        "B0": canonical("B0"),
        "N": canonical("N"),
        # BALROG: natural-language scene, no ASCII grid.
        "B": canonical("B", turn_template=_formatter_template(_format_obs_balrog)),
        # Glyphbox: canonical render, paired with interface=code by the caller.
        "G": canonical("G", turn_template=_formatter_template(_format_obs_glyphbox)),
        # Summarize-and-reset: canonical-equivalent render; the drop is driven by
        # the orthogonal summarize_and_reset kwarg, not the variant.
        "R": canonical("R", turn_template=_formatter_template(_format_obs_summarize_reset)),
        # Descent-salience block (ND, FD).
        "ND": canonical("ND", obs=ObsSpec(setup_flags={"_descent_salient": True})),
        "FD": canonical("FD", obs=ObsSpec(setup_flags={"_descent_salient": True})),
        # E1 frontier-surface blocks.
        "E1": canonical("E1", obs=ObsSpec(setup_flags={"_e1_obs": True})),
        # E2 paint frontiers onto the map.
        "E2": canonical("E2", obs=ObsSpec(setup_flags={"_e2_obs": True})),
        # Image observation: rendered tiles (IMG) or tty raster (IMG_TTY).
        "IMG": canonical("IMG", obs=ObsSpec(mode="img"),
                         turn_template=_image_template("glyph")),
        "IMG_TTY": canonical("IMG_TTY", obs=ObsSpec(mode="img"),
                             turn_template=_image_template("tty")),
        # Structured-text map: JSON or in-repo TOON, built from the canonical
        # map model; map_detail (full/minimal) rides on state["map_detail"].
        "JSON": canonical("JSON", turn_template=_structured_map_template("json")),
        "TOON": canonical("TOON", turn_template=_structured_map_template("toon")),
        # Sub-experiment 1d (observation DELIVERY): delayed / on-demand map.
        # The FULL map is re-sent only on a material change or an explicit
        # request_map/reveal; otherwise a placeholder stands in. Action feedback
        # is unchanged every turn. Composes with the ASCII (DM) and JSON
        # (DM_JSON) encodings.
        "DM": canonical("DM", turn_template=_delayed_map_template("ascii"),
                        obs=ObsSpec(setup_flags={"_delayed_map": True})),
        "DM_JSON": canonical("DM_JSON", turn_template=_delayed_map_template("json"),
                             obs=ObsSpec(setup_flags={"_delayed_map": True})),
        # Bounding-box on-demand: the map is hidden; the agent views regions via
        # reveal(x1,y1,x2,y2), which returns an ASCII crop as tool feedback.
        "BBOX": canonical("BBOX", turn_template=_bbox_template,
                          obs=ObsSpec(setup_flags={"_bbox_map": True})),
        # Continual-harness adaptation: periodic self-refinement directive.
        "P": canonical("P", turn_hooks=(_p_refinement_hook,)),
        # Full Continual Harness: refiner + sub-agents + system inject + run_macro.
        # Built by attaching the refiner bundle to a canonical (ASCII) spec, so
        # CH and JSON+refine share one wiring path.
        "CH": attach_refiner(canonical("CH")),
    }


VARIANT_REGISTRY = _build_registry(SYSTEM_PROMPT)


def resolve_spec(variant: str, system_prompt: str) -> PromptSpec:
    """Return the PromptSpec for ``variant`` with ``system_prompt`` injected.

    ``system_prompt`` is passed explicitly (rather than read from a module
    global) so callers can hand in the value *after* the NETHACK_HARNESS overlay
    has mutated it — keeping the overlay seam intact.
    """
    base = VARIANT_REGISTRY.get(variant) or VARIANT_REGISTRY["B1"]
    return dataclasses.replace(base, system_prompt=system_prompt)
