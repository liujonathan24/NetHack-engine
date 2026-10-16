"""
nethack_core.trace_schema
==========================

Single, versioned source of truth for the per-turn rollout **trace** format —
the NDJSON stream the env writes (one JSON object per line, per LM turn) and the
rollout viewer / web console read back to replay a game exactly as the model saw
it.

Both sides import this module so the on-disk contract lives in one place:

    * writer  -> ``environments/nethack/nethack_harness/helpers._write_trace_entry``
    * readers -> ``tools/rollout_view/stats.load_trace`` and friends

This module deliberately depends on the standard library only (``json``). It
must NOT import ``verifiers``, ``flask`` or ``pufferlib`` so it stays usable
from every layer (engine, hub, console) without dragging in heavy deps.

On-disk field names are frozen (do not rename them). Bumping the format means
bumping :data:`TRACE_SCHEMA_VERSION` and documenting the change below.

Schema history
--------------
* ``1`` — first explicitly-versioned schema. Field set frozen from the
  pre-versioning writer; adds the ``schema_version`` stamp. Records written
  before versioning simply lack the key and are read as ``version 0`` (legacy).
* ``2`` — reproducibility fields. All additive, all optional, so ``version 0``
  and ``version 1`` records still validate unchanged:

  - ``t_mono``   — the writer has emitted this since before versioning and
    ``tools/cli_harness_eval/aggregate.py`` consumes it; the *schema* was the
    stale side, so it is documented here rather than dropped from the writer.
  - ``lm_turn``  — 1-based LM-turn ordinal. Unlike ``turn`` (which counts only
    turns that reached the engine) there is exactly ONE record per LM turn, so
    ``len(records) == metrics.num_turns == metrics.total_tool_calls``. This is
    the authoritative per-record counter; ``turn`` is kept verbatim for readers
    written against version 0/1.
  - ``applied``  — bool: did this turn's tool call actually reach the engine?
  - ``actions``  — the full ordered low-level command stream (see
    :func:`empty_action_record`).
  - ``all_messages`` — every in-game message produced this turn, in order,
    including the intermediates a multi-step skill would otherwise swallow.
    ``messages`` (last message only) is unchanged for old readers.
  - ``tool_results`` — machine-readable per-call outcome (status / game message
    / whether the game clock advanced).
  - ``gt_obs``   — the engine's ground-truth observation planes, packed and
    compressed (see :func:`encode_obs_blob`).
* ``3`` — provenance of the *decision*, not the game. Also additive/optional,
  so version 0-2 records still validate unchanged:

  - ``dispatch_route`` — which route delivered this tool call: ``"harness"``
    (the in-process v0 rollout loop parsed it out of an assistant message) or
    ``"mcp"`` (a CLI agent called the tool server directly). The two routes
    have structurally different visibility into the model, and the field says
    which one a record came from instead of leaving a reader to infer it from
    an empty ``assistant_message``. See :data:`DISPATCH_ROUTES`.
  - ``reasoning``  — an explicit record of whether the agent's own words for
    this turn are available, where they came from, and -- when they are not --
    WHY. Measured before this existed: ``assistant_message`` was ``""`` on
    100% of CLI-arm turns with nothing distinguishing "the model said nothing"
    from "this route cannot see what the model said". See
    :func:`reasoning_record` / :func:`unavailable_reasoning`.
"""
from __future__ import annotations

import base64
import json
import zlib
from pathlib import Path
from typing import IO, Iterator, Union

# Current on-disk schema version. Bump when the field set changes.
TRACE_SCHEMA_VERSION = 3

# The key under which the version is stamped in each record.
SCHEMA_VERSION_KEY = "schema_version"

# ---------------------------------------------------------------------------
# Field catalogue (documentation + validation). Names are frozen — the writer
# and every reader agree on exactly these keys.
# ---------------------------------------------------------------------------

#: Fields every well-formed record carries.
REQUIRED_FIELDS: tuple[str, ...] = (
    "turn",                    # int: LM turn index within the rollout
    "raw_grid",                # list[str]: raw 24x80 tty rows (rstripped)
    "status",                  # dict: parsed status line (depth, hp, ...)
    "rendered_user_message",   # str: the literal text observation sent to the LM
    "tool_calls",              # list[{"name","arguments"}]: parsed tool calls
    "action_indices",          # list[int]: NLE action indices applied this turn
    "reward",                  # float: reward accrued this turn
)

#: Fields that may be present (writer emits them; readers must tolerate absence).
OPTIONAL_FIELDS: tuple[str, ...] = (
    SCHEMA_VERSION_KEY,        # int: schema version stamp (absent => legacy/0)
    "t_wall",                  # float: wall-clock timestamp
    "variant",                 # str: harness variant id
    "dlvl",                    # int|None: dungeon level (mirrors status depth)
    "hp",                      # int|None: hitpoints (mirrors status)
    "max_hp",                  # int|None: max hitpoints
    "max_dlvl_reached",        # int|None: deepest dlvl reached so far
    "continual_life",          # int: continual-harness life counter
    "rendered_user_content",   # str|list: multimodal user content (images->paths)
    "assistant_message",       # str: assistant message consumed this turn
    "messages",                # list[str]: in-game messages this turn
    "ch_edits",                # optional: refiner per-interval edits (variant CH)
    "checkpoint",              # str|None: level checkpoint path (web-console recorder)
    # --- schema version 2 (all additive/optional) ---------------------------
    "t_mono",                  # float: monotonic clock (latency analysis)
    "lm_turn",                 # int: 1-based LM-turn ordinal; one record per LM turn
    "applied",                 # bool: did this turn's tool call reach the engine?
    "actions",                 # dict: full low-level command stream (see below)
    "all_messages",            # list[str]: every in-game message this turn, in order
    "tool_results",            # list[dict]: machine-readable per-call outcomes
    "gt_obs",                  # dict|None: packed ground-truth observation planes
    # --- schema version 3 (all additive/optional) ---------------------------
    "dispatch_route",          # str: "harness" | "mcp" (see DISPATCH_ROUTES)
    "reasoning",               # dict: the agent's words + their provenance
)

#: Every documented field name.
ALL_FIELDS: tuple[str, ...] = REQUIRED_FIELDS + OPTIONAL_FIELDS


# ---------------------------------------------------------------------------
# version 2: ground-truth observation blob
# ---------------------------------------------------------------------------

#: Observation planes persisted as ground truth, in pack order.
#:
#: SIZE/FIDELITY TRADEOFF. The full ``CoreObservation`` is ~11.7 KB of raw
#: ndarray bytes per turn (glyphs 3318 + chars 1659 + colors 1659 + inv_strs
#: 4400 + blstats 216 + ...). Written as JSON integer lists that is ~45 KB/turn,
#: which would have made the pilot's 99-turn NDJSON ~4.5 MB per seed instead of
#: 645 KB -- a 7x blow-up on a file that already sits beside a 59 MB
#: traces.jsonl. Three cheaper options were considered and rejected:
#:
#:   * store nothing (status quo)      -- analysis stays trapped in rendered text;
#:   * store only ``glyphs``           -- loses the inventory and status ground
#:                                        truth, which is most of what an
#:                                        analysis actually wants;
#:   * delta-encode against the previous turn -- smallest, but a single dropped
#:                                        or reordered record makes every later
#:                                        turn undecodable, and these files are
#:                                        appended to live from a crashy rollout.
#:
#: What we do instead: concatenate every plane into ONE byte buffer, zlib it as
#: a single stream (so the large all-zero runs in ``inv_strs`` and the
#: repetitive rock glyphs compress against each other), then base64. Measured on
#: a live probe this lands at ~1.0-1.5 KB/turn, i.e. every plane kept, ~20%
#: added to the NDJSON, and each record still independently decodable.
#: ``tty_chars`` is deliberately EXCLUDED: ``raw_grid`` already carries it
#: losslessly as rstripped text.
GT_OBS_PLANES: tuple[str, ...] = (
    "glyphs", "chars", "colors", "blstats", "message",
    "inv_strs", "inv_letters", "inv_glyphs", "tty_colors", "tty_cursor", "misc",
)

#: Codec id stamped into each blob so a future change can be detected.
GT_OBS_CODEC = "concat+zlib+b64"


def encode_obs_blob(obs, planes: "tuple[str, ...]" = GT_OBS_PLANES) -> Union[dict, None]:
    """Pack the named ndarray planes of ``obs`` into one compressed blob.

    ``obs`` is any object exposing the planes as attributes (a
    ``nethack_core.env.CoreObservation``) or a mapping. Returns ``None`` if no
    plane could be read -- callers treat that as "ground truth unavailable"
    rather than as an error. numpy is imported lazily so this module stays
    importable (and this function unused) in stdlib-only contexts.
    """
    try:
        import numpy as np
    except Exception:
        return None
    if obs is None:
        return None
    # The manifest is a list of compact ``[name, dtype, shape]`` triples rather
    # than a list of objects: offsets and byte counts are implied by dtype x
    # shape in pack order, and spelling them out as JSON keys cost ~1 KB/record
    # -- as much as the compressed pixel data itself.
    manifest: list = []
    chunks: list[bytes] = []
    offset = 0
    for name in planes:
        try:
            val = obs.get(name) if isinstance(obs, dict) else getattr(obs, name, None)
            if val is None:
                continue
            arr = np.ascontiguousarray(val)
            raw = arr.tobytes()
        except Exception:
            continue
        manifest.append([name, str(arr.dtype), list(arr.shape)])
        chunks.append(raw)
        offset += len(raw)
    if not manifest:
        return None
    packed = zlib.compress(b"".join(chunks), 6)
    return {
        "codec": GT_OBS_CODEC,
        "planes": manifest,
        "raw_bytes": offset,
        "data": base64.b64encode(packed).decode("ascii"),
    }


def decode_obs_blob(blob: Union[dict, None]) -> dict:
    """Inverse of :func:`encode_obs_blob`: blob -> ``{plane_name: ndarray}``.

    Returns ``{}`` for a missing/unknown blob so readers can stay tolerant.
    """
    if not blob or blob.get("codec") != GT_OBS_CODEC:
        return {}
    import numpy as np

    buf = zlib.decompress(base64.b64decode(blob["data"]))
    out = {}
    offset = 0
    for name, dtype, shape in blob.get("planes") or []:
        dt = np.dtype(dtype)
        nbytes = int(dt.itemsize)
        for dim in shape:
            nbytes *= int(dim)
        out[name] = np.frombuffer(
            buf[offset:offset + nbytes], dtype=dt).reshape(tuple(shape))
        offset += nbytes
    return out


# ---------------------------------------------------------------------------
# version 2: action stream
# ---------------------------------------------------------------------------

#: How the action stream was obtained. ``engine_step_hook`` means every call to
#: the engine's ``step()`` during this LM turn was intercepted, so the list is
#: complete regardless of whether the actions came from the harness step loop,
#: from inside a closed-loop skill (``np_explore_level`` etc.), from the menu
#: auto-dismiss drain, or from the deadlock-breaker.
ACTION_SOURCE_ENGINE_HOOK = "engine_step_hook"


def empty_action_record(reason: str) -> dict:
    """An explicitly-unobservable action record.

    The point of ``recorded: False`` is that it is NOT the same value as "this
    turn genuinely executed zero actions" (which is ``recorded: True, n: 0``).
    Version 0/1 traces could not tell those apart: both were ``[]``.
    """
    return {"recorded": False, "source": None, "reason": reason, "n": 0,
            "bytes": [], "replayable": False}


def action_record(actions, source: str = ACTION_SOURCE_ENGINE_HOOK,
                  replayable: bool = True, reason=None) -> dict:
    """A complete action record: the ordered keystroke bytes the engine consumed.

    ``bytes`` is directly replayable -- ``NetHackCoreEnv.step`` takes exactly
    these values (the semantic action enums ARE keystroke bytes, see
    ``nethack_harness.helpers._to_action_indices``). ``keys`` is the same
    sequence rendered for humans and is derived, never authoritative.

    ``replayable`` is False when the turn ALSO mutated engine state by some
    route other than ``step`` -- today that means the ``rollback`` skill, which
    calls ``engine.restore(handle)`` on a heap snapshot. Replaying the byte
    stream past such a turn would diverge, and a trace that did not say so
    would be quietly wrong rather than usefully incomplete.
    """
    vals = [int(a) for a in actions]
    return {
        "recorded": True,
        "source": source,
        "reason": reason,
        "n": len(vals),
        "bytes": vals,
        "keys": "".join(key_repr(v) for v in vals),
        "replayable": bool(replayable),
    }


def key_repr(byte: int) -> str:
    """Render one keystroke byte for human reading (round-trips via ``bytes``)."""
    b = int(byte)
    if b == 27:
        return "<ESC>"
    if b == 13:
        return "<CR>"
    if b == 10:
        return "<LF>"
    if b == 32:
        return "<SP>"
    if 33 <= b <= 126:
        return chr(b)
    if 0 < b < 27:
        return "^" + chr(b + 64)
    return "<%d>" % b


# ---------------------------------------------------------------------------
# version 2: tool results
# ---------------------------------------------------------------------------

#: Status vocabulary for ``tool_results[i]["status"]``.
#:
#: ``completed`` deliberately does NOT mean "made progress" -- the vendored
#: NetPlay skills report ``Step.completed()`` without reading the result, so a
#: move that stopped against a wall says "completed" too. ``clock_advanced`` is
#: the field that separates them.
TOOL_STATUSES: tuple[str, ...] = (
    "completed",       # skill ran to completion (says nothing about progress)
    "failed",          # the skill itself reported failure
    "interrupted",     # skill stopped itself early (new glyph, HP drop, spoiling food)
    "refused",         # harness safety net declined the call (pet/peaceful)
    "error",           # the skill raised
    "rejected",        # tool not in this rollout's exposed set (hallucinated)
    "no_op",           # journal/info-only call; no engine step by design
    "no_tool_call",    # the model emitted no tool call this turn
    "dead",            # character already dead; call refused
    "not_applied",     # emitted by the model but the rollout ended first
    "unknown",
)


# ---------------------------------------------------------------------------
# version 3: dispatch route + reasoning provenance
# ---------------------------------------------------------------------------

#: How this turn's tool call reached the engine.
#:
#: ``harness`` -- the in-process v0 rollout loop parsed it out of the assistant
#: message it had just received, so the model's own text is in hand at write
#: time. ``mcp`` -- a CLI agent (Claude Code, Prime Agent) called the tool
#: server over MCP; the server sees the call and its arguments but never the
#: message that produced it. Both routes run the SAME ``_apply_tool_call``, so
#: everything about the game is identical; only the visibility into the model
#: differs, and that is what this field records.
DISPATCH_ROUTES: tuple[str, ...] = ("harness", "mcp")

#: Where a record's ``reasoning.text`` came from.
#:
#: ``assistant_message`` -- captured inline at write time from the message the
#: harness had just parsed (the control arm).
#: ``trace_nodes`` -- backfilled after the rollout from the sampled assistant
#: messages the verifiers interception server recorded on the rollout trace
#: (``traces.jsonl``'s ``nodes``). This is the only source available to the
#: MCP route, because the tool server is a different process from the one the
#: model talks to.
REASONING_SOURCES: tuple[str, ...] = ("assistant_message", "trace_nodes")

#: How a backfilled ``reasoning`` block was matched to its turn record.
#:
#: ``inline`` -- no matching needed; the message was in hand when the record
#: was written. ``tool_call_sequence`` -- the sampled assistant turns' game
#: tool-call names were matched, in order, against the dispatched calls, and
#: the two sequences agreed. ``ordinal`` -- the counts agreed 1:1 and the
#: records were paired by position (used when names cannot be compared).
REASONING_ALIGNMENTS: tuple[str, ...] = ("inline", "tool_call_sequence", "ordinal")


def reasoning_record(text, source: str, alignment: str,
                     reasoning_text: str = "") -> dict:
    """One available ``reasoning`` block.

    ``text`` is the assistant's visible content for the turn; ``reasoning_text``
    is the provider's separate reasoning channel (``reasoning_content``) when
    one was returned, kept apart because it is not always replayable and is
    priced differently.
    """
    return {
        "available": True,
        "source": source,
        "alignment": alignment,
        "text": text or "",
        "reasoning_text": reasoning_text or "",
        "reason": "",
    }


def unavailable_reasoning(reason: str) -> dict:
    """One ``reasoning`` block for a turn whose agent-side text is NOT available.

    The reason is mandatory and is the whole point of the field: an explicit
    "unavailable because <the tool server never sees the model's message>"
    is a different fact from "the model chose to say nothing", and an empty
    ``assistant_message`` string cannot tell them apart.
    """
    return {
        "available": False,
        "source": None,
        "alignment": None,
        "text": "",
        "reasoning_text": "",
        "reason": reason,
    }


def record_version(record: dict) -> int:
    """Return the schema version stamped on ``record`` (0 if unstamped/legacy)."""
    try:
        return int(record.get(SCHEMA_VERSION_KEY, 0))
    except (TypeError, ValueError):
        return 0


def stamp(record: dict) -> dict:
    """Return ``record`` with the current schema version stamped in place."""
    record[SCHEMA_VERSION_KEY] = TRACE_SCHEMA_VERSION
    return record


def to_json_line(record: dict) -> str:
    """Serialize one record to a single NDJSON line (trailing newline included).

    Stamps the current :data:`TRACE_SCHEMA_VERSION`. On-disk field names are left
    untouched; only the version key is added.
    """
    return json.dumps(stamp(record)) + "\n"


def write_record(fp: IO[str], record: dict) -> None:
    """Append one stamped record as an NDJSON line to an open text file object."""
    fp.write(to_json_line(record))


def parse_line(line: str) -> Union[dict, None]:
    """Parse one NDJSON line into a record dict, or ``None`` if it is blank,
    not valid JSON, or not a JSON object. Readers skip ``None`` results."""
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def iter_records(path: Union[str, Path]) -> Iterator[dict]:
    """Yield each valid record from an NDJSON trace file, skipping bad lines."""
    for line in Path(path).read_text().splitlines():
        rec = parse_line(line)
        if rec is not None:
            yield rec


def read_trace(path: Union[str, Path]) -> list[dict]:
    """Read a whole NDJSON trace file into a list of record dicts."""
    return list(iter_records(path))


def validate_record(record: dict) -> list[str]:
    """Return a list of human-readable problems with ``record`` (empty == valid).

    Checks that ``record`` is a dict, carries every required field, and has no
    unknown keys. Intended for tests / defensive callers; the writer and readers
    are tolerant by design and do not call this on the hot path.
    """
    problems: list[str] = []
    if not isinstance(record, dict):
        return ["record is not a dict"]
    for f in REQUIRED_FIELDS:
        if f not in record:
            problems.append(f"missing required field: {f!r}")
    known = set(ALL_FIELDS)
    for k in record:
        if k not in known:
            problems.append(f"unknown field: {k!r}")
    return problems


def is_valid(record: dict) -> bool:
    """True if ``record`` has all required fields and no unknown keys."""
    return not validate_record(record)
