"""Harness runtime helpers extracted verbatim from nethack.py (no logic change):
chat-history compaction, continual-harness reset, refiner/CH glue, belief-state
distillation, terminal-outcome detection, reward functions, and the verifiers
code/skill tool adapters.
"""
from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any, Optional

import verifiers as vf

from nethack_core import NetHackCoreEnv, trace_schema
from nethack_harness.memory.journal import Journal
from nethack_core import shape as shape_observation
from nethack_harness.tools.skills import registry as skill_registry, list_skills

def _continual_reset(state: dict, env, env_self) -> None:
    """Continual harness: reseed and reset the underlying NLE so the chat
    session continues into a new life. Preserves journal + belief notes;
    bumps a `_continual_life` counter; records the death into the journal."""
    lives_left = state.get("_continual_lives_left", env_self.continual_lives)
    life_no = state.get("_continual_life", 1)
    # Snapshot death context into the journal so the agent can remember it.
    s = state.get("structured_obs")
    death_note = "died"
    if s is not None:
        death_note = (
            f"life {life_no}: died at Dlvl {s.status.get('depth','?')} "
            f"on turn {s.status.get('time','?')} "
            f"(max Dlvl reached {state.get('max_dlvl_reached','?')})"
        )
    journal = state.get("journal")
    if journal is not None:
        try:
            journal.add_note(f"death:life{life_no}", death_note)
        except Exception:
            pass
    # Reseed deterministically from the original seed + life number.
    orig_seed = state.get("_orig_seed")
    if orig_seed is None:
        # Recover from env metadata if not stored yet.
        orig_seed = (env.current_seeds or (0, 0))[0]
        state["_orig_seed"] = orig_seed
    new_seed = (int(orig_seed) * 1_000_003 + life_no) & 0x7FFFFFFF
    env.seed(core=new_seed, disp=new_seed)
    obs, _meta = env.reset()
    from nethack_harness.tools.skills import bootstrap_character
    character = bootstrap_character(env)
    state["character"] = character
    state["raw_obs"] = obs
    state["structured_obs"] = shape_observation(obs, character)
    state["max_dlvl_reached"] = max(state.get("max_dlvl_reached", 1), 1)
    state["died"] = False
    state["ascended"] = False
    state["_continual_life"] = life_no + 1
    state["_continual_lives_left"] = lives_left - 1
    # Reset per-life ephemera but keep cross-life memory (journal, belief).
    state["_seen_stairs_down"] = set()


def _capture_user_content(content, out_dir, *, run_id: str, turn: int):
    """Return a trace-safe copy of the per-turn user content.

    Strings pass through. For a multimodal list, each image_url data URI is
    decoded and written to ``<out_dir>/images/<run_id>_<turn>_<idx>.png`` and the
    entry is rewritten to reference the relative path instead of the inline
    base64, so the exact image the model saw is replayable without bloating the
    NDJSON.
    """
    if isinstance(content, str):
        return content
    import base64 as _b64
    images_dir = Path(out_dir) / "images"
    out = []
    idx = 0
    for entry in content:
        if entry.get("type") == "image_url":
            url = (entry.get("image_url") or {}).get("url", "")
            if url.startswith("data:") and "base64," in url:
                images_dir.mkdir(parents=True, exist_ok=True)
                b64 = url.split("base64,", 1)[1]
                fname = f"{run_id}_{turn}_{idx}.png"
                (images_dir / fname).write_bytes(_b64.b64decode(b64))
                out.append({"type": "image_url", "image_url": {"path": f"images/{fname}"}})
                idx += 1
            else:
                out.append(entry)
        else:
            out.append(entry)
    return out


def _write_trace_entry(env_self, state: dict, assistant_msg, tool_calls,
                       action_indices, total_reward: float, obs_text: str,
                       obs_content=None) -> None:
    """Write one NDJSON line per env_response turn. Best-effort; never raises.

    Captures everything needed by the replay viewer to render the game as
    the model saw it: raw 24x80 tty grid, structured obs, the literal user
    message we will send back, the assistant message we just consumed, the
    parsed tool calls, the NLE action indices applied, reward, dlvl, hp.
    """
    if not env_self.trace_dir:
        return
    try:
        import os as _os
        import time as _time
        out_dir = Path(env_self.trace_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        run_id = state.get("_trace_run_id")
        if run_id is None:
            seeds = state.get("env").current_seeds if state.get("env") else (0, 0)
            run_id = f"{seeds[0]}_{_os.getpid()}_{int(_time.time())}"
            state["_trace_run_id"] = run_id
        path = out_dir / f"{run_id}.ndjson"
        raw = state.get("raw_obs")
        grid = []
        if raw is not None:
            try:
                grid = [
                    "".join(chr(int(c)) for c in row).rstrip()
                    for row in raw.tty_chars
                ]
            except Exception:
                pass
        s = state.get("structured_obs")
        status = dict(s.status) if s is not None else {}
        assist_content = ""
        if assistant_msg is not None:
            if isinstance(assistant_msg, dict):
                assist_content = assistant_msg.get("content", "") or ""
            else:
                assist_content = getattr(assistant_msg, "content", "") or ""
        tc_serial = []
        for tc in (tool_calls or []):
            if isinstance(tc, dict):
                fn = tc.get("function") or {}
                tc_serial.append({
                    "name": fn.get("name") or tc.get("name"),
                    "arguments": fn.get("arguments") or tc.get("arguments"),
                })
            else:
                fn = getattr(tc, "function", None)
                tc_serial.append({
                    "name": getattr(fn, "name", None) if fn is not None
                            else getattr(tc, "name", None),
                    "arguments": getattr(fn, "arguments", None) if fn is not None
                                 else getattr(tc, "arguments", None),
                })
        entry = {
            "turn": state.get("turn_count", 0),
            "t_wall": _time.time(),
            "variant": env_self.variant,
            "raw_grid": grid,
            "status": status,
            "dlvl": status.get("depth"),
            "hp": status.get("hitpoints"),
            "max_hp": status.get("max_hitpoints"),
            "max_dlvl_reached": state.get("max_dlvl_reached"),
            "continual_life": state.get("_continual_life", 1),
            "rendered_user_message": obs_text,
            "rendered_user_content": _capture_user_content(
                obs_content if obs_content is not None else obs_text,
                out_dir, run_id=run_id, turn=state.get("turn_count", 0)),
            "assistant_message": assist_content,
            "tool_calls": tc_serial,
            "action_indices": list(action_indices) if action_indices else [],
            "reward": float(total_reward),
            "messages": list(s.messages) if s and s.messages else [],
        }
        # Variant CH: capture the refiner's per-interval edits (set by
        # _ch_refiner_hook on refinement turns) so the trace records exactly
        # what the teacher changed this turn.
        if state.get("_ch_last_edits"):
            entry["ch_edits"] = state["_ch_last_edits"]
        with path.open("a") as f:
            # Route through the versioned trace schema: stamps schema_version and
            # serializes the (frozen) on-disk field set. See nethack_core.trace_schema.
            trace_schema.write_record(f, entry)
    except Exception:
        # Tracing must never break a rollout.
        pass


def _drop_before_last_belief(messages, state) -> list:
    """Variant R: hard-drop every message before the chat-position corresponding
    to the most-recent belief_state:tN checkpoint.

    Heuristic: we don't track which chat turn produced each belief note, so
    instead we drop everything older than (refine_distance_from_end) where
    refine_distance is set to the belief_state_interval. The journal block
    inside the *current* user message already carries the belief notes, so
    no semantic info is lost.

    Leaves the system message and the most recent user/assistant pair fully
    intact. Inserts a single elision marker so the model knows context was
    dropped.
    """
    if not state:
        return messages
    journal = state.get("journal")
    if journal is None:
        return messages
    # If no belief checkpoint has fired yet, do nothing.
    has_belief = any(k.startswith("belief_state:") for k in (journal.notes or {}).keys())
    if not has_belief:
        return messages
    # Find indices of user messages and keep only the last K (here K=2 — the
    # last user obs and its preceding assistant exchange suffice once the
    # belief state carries the rest).
    keep_window = 2
    user_idx = [i for i, m in enumerate(messages) if _msg_role(m) == "user"]
    if len(user_idx) <= keep_window:
        return messages
    cut_at = user_idx[-keep_window]
    out = []
    for i, m in enumerate(messages):
        if i == 0 and _msg_role(m) == "system":
            out.append(m)
            continue
        if i >= cut_at:
            out.append(m)
    # Insert elision marker right after the system message.
    insert_at = 1 if out and _msg_role(out[0]) == "system" else 0
    n_dropped = len(messages) - len(out)
    if n_dropped > 0:
        out.insert(insert_at, vf.UserMessage(
            role="user",
            content=f"[variant=R: {n_dropped} prior turns dropped; see JOURNAL belief_state notes for context]",
        ))
    return out


def _refinement_directive(state: dict) -> str:
    """Variant P (Continual Harness, arXiv:2605.09998) periodic self-refinement
    prompt. Injected every refine_interval turns; asks the agent to reflect on
    the last window of play and update its objective and/or write a lesson
    note. Because `pin_objective` and `add_note` are journal ops that don't
    consume an NLE step, the agent can spend this turn editing its own
    persistent memory without losing a game action."""
    turn = state.get("turn_count", 0)
    max_dlvl = state.get("max_dlvl_reached", 1)
    cur_dlvl = state["structured_obs"].status.get("depth", 1) if state.get("structured_obs") else 1
    return (
        f"[self-refinement turn (variant=P, t={turn})] "
        f"You are at Dlvl {cur_dlvl} (max reached {max_dlvl}). "
        f"Before your next action, reflect: is your current objective still "
        f"the right one? What pattern from the last {state.get('_refine_window', 20)} "
        f"turns should you remember? Call `pin_objective(text=...)` to update "
        f"the goal if it has shifted, or `add_note(key='lesson:t{turn}', "
        f"text=...)` to record a short lesson. These calls do NOT consume a "
        f"game turn. If nothing needs updating, take your normal action."
    )


def _ch_build_window(trajectory: list, n_turns: int) -> list[dict]:
    """Slice the last `n_turns` of chat history into {role, content} dicts
    for the Refiner. Handles both dict-shape and verifiers pydantic msgs."""
    out: list[dict] = []
    for msg in trajectory[-(2 * n_turns):]:
        if isinstance(msg, dict):
            role = msg.get("role", "?")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "role", "?")
            content = getattr(msg, "content", "")
        out.append({"role": role, "content": content if isinstance(content, str) else str(content)})
    return out


def _ch_inject_system(messages, state: dict):
    """Append CH prompt addendum + macro list onto the system message."""
    addendum = (state.get("_ch_prompt_addendum") or "").strip()
    macros = state.get("_ch_skills") or {}
    if not addendum and not macros:
        return messages
    extra_lines: list[str] = []
    if addendum:
        extra_lines.append("\n[continual-harness addendum]\n" + addendum)
    if macros:
        # Surface available macros so the agent can call run_macro(name=...).
        macro_names = ", ".join(sorted(macros.keys()))
        extra_lines.append(
            f"\n[continual-harness macros] You may also call "
            f"`run_macro(name=...)` with one of: {macro_names}."
        )
    extra = "".join(extra_lines)
    out = list(messages)
    for i, m in enumerate(out):
        if _msg_role(m) == "system":
            out[i] = _replace_content(m, _msg_content(m) + extra)
            return out
    # No system message found (shouldn't happen, dataset prepends one) —
    # prepend a fresh system block.
    out.insert(0, vf.SystemMessage(role="system", content=extra.lstrip()))
    return out


def _ch_save_bootstrap(env_self, state: dict) -> None:
    """Persist the four CH components to <bootstrap_dir>/seed<N>.json on
    terminal. Best-effort; never raises."""
    try:
        if not getattr(env_self, "bootstrap_dir", None):
            return
        import os, json
        from nethack_harness.refiner import snapshot_components
        os.makedirs(env_self.bootstrap_dir, exist_ok=True)
        seed = state.get("_orig_seed", 0)
        path = os.path.join(env_self.bootstrap_dir, f"seed{seed}.json")
        with open(path, "w") as f:
            json.dump(snapshot_components(state), f, indent=2)
    except Exception:
        pass


def _compact_chat_history(messages, keep_full: int = 5, drop_after: int = 100):
    """Compact older user (env-response) messages in-place so the chat doesn't
    grow without bound. Assistant messages are kept verbatim — their tool_calls
    matter for downstream replay. Only the *content* of user messages older
    than `keep_full` turns is rewritten.

    Compaction tiers:
      - turn distance ≤ keep_full: full fidelity (unchanged)
      - keep_full < distance ≤ drop_after: one-line summary
      - distance > drop_after: message dropped entirely

    Returns a NEW list so we never mutate state["trajectory"].
    """
    # Count from the end. "Turn distance" = how many user messages back this is.
    out = list(messages)
    # Walk backwards over user messages to compute distance.
    user_indices = [i for i, m in enumerate(out) if _msg_role(m) == "user"]
    if len(user_indices) <= keep_full:
        return out
    # Older user messages we want to compact (oldest first up to threshold).
    to_compact = user_indices[: -keep_full]
    to_drop_indices = set()
    for distance_from_end, idx in enumerate(reversed(to_compact), start=keep_full + 1):
        if distance_from_end > drop_after:
            to_drop_indices.add(idx)
            continue
        # One-line summary: keep prefix tags ([autohalt: ...], [feedback: ...])
        # and HP/Turn/Dlvl status line; drop everything else.
        summary = _one_line_summary(_msg_content(out[idx]), distance_from_end)
        out[idx] = _replace_content(out[idx], summary)
    if to_drop_indices:
        # Replace dropped chunks with a single elision marker if there's anything to drop.
        n_dropped = len(to_drop_indices)
        out = [m for i, m in enumerate(out) if i not in to_drop_indices]
        # Insert a single elision marker at the start (after system prompt if present).
        from typing import cast
        insert_at = 1 if out and _msg_role(out[0]) == "system" else 0
        out.insert(insert_at, vf.UserMessage(
            role="user",
            content=f"[elided {n_dropped} older turns; see journal for context]",
        ))
    # Second pass: collapse consecutive compacted user messages with the same
    # status signature. In the 9071d001 trace, 90 user msgs were literally
    # "[turn -X] HP: 14/14 AC: 4 Dlvl: 1 ..." with the same HP/AC/Dlvl — pure
    # token noise. We shrink runs to "[turn -Y] (status unchanged)".
    out = _dedupe_compacted_runs(out)
    return out


_STATUS_SIG_RE = re.compile(r"HP:\s*(\d+/\d+)\s+AC:\s*(-?\d+)\s+Dlvl:\s*(\d+)")


def _compacted_status_signature(content: str) -> Optional[tuple]:
    """Return (hp_ratio, ac, dlvl, has_feedback) for a compacted user message,
    or None if the message isn't recognized as compacted-only (i.e., still has
    a MAP block, or is an elision marker, or has unique feedback)."""
    if "=== MAP ===" in content or "elided" in content[:30]:
        return None
    if not content.lstrip().startswith("[turn -"):
        return None
    # Any non-trivial bracketed feedback (e.g. [Moved S.], [Attack hit]) makes
    # this turn unique — don't collapse.
    head = content.split("\n", 1)[0]
    # Strip leading [turn -N]
    rest = re.sub(r"^\[turn -\d+\]\s*", "", head)
    fb_match = re.match(r"\[([^\[\]]{1,80})\]", rest)
    has_feedback = bool(fb_match and "turn -" not in fb_match.group(1))
    sig = _STATUS_SIG_RE.search(content)
    if not sig:
        return None
    return (sig.group(1), sig.group(2), sig.group(3), has_feedback)


def _dedupe_compacted_runs(messages):
    """Collapse consecutive compacted user messages with identical
    HP/AC/Dlvl signatures and no per-turn feedback into terse `[turn -Y]
    (unchanged)` placeholders. Keeps the first message in each run intact
    so the model can read the actual status; later messages just mark
    that nothing changed.

    Does not change message count or assistant messages — purely shrinks
    redundant content.
    """
    out = list(messages)
    last_sig: Optional[tuple] = None
    for i, m in enumerate(out):
        if _msg_role(m) != "user":
            continue
        content = _msg_content(m)
        sig = _compacted_status_signature(content)
        if sig is None:
            last_sig = None
            continue
        # If feedback present, keep full and reset run.
        _, _, _, has_feedback = sig
        if has_feedback:
            last_sig = sig
            continue
        if last_sig is not None and sig[:3] == last_sig[:3]:
            # Same status as previous compacted msg: shrink.
            turn_match = re.match(r"\[turn -(\d+)\]", content)
            label = turn_match.group(0) if turn_match else "[turn -?]"
            out[i] = _replace_content(m, f"{label} (unchanged)")
        last_sig = sig
    return out


def _msg_role(m) -> str:
    """Pull role from either a dict-shaped message or a pydantic Message."""
    if isinstance(m, dict):
        return str(m.get("role", ""))
    return str(getattr(m, "role", ""))


def _msg_content(m) -> str:
    if isinstance(m, dict):
        c = m.get("content", "")
    else:
        c = getattr(m, "content", "")
    return c if isinstance(c, str) else ""


def _replace_content(m, new_content: str):
    """Return a copy of `m` with content swapped, preserving role + pydantic class."""
    if isinstance(m, dict):
        out = dict(m)
        out["content"] = new_content
        return out
    # pydantic: use model_copy(update=...).
    try:
        return m.model_copy(update={"content": new_content})
    except Exception:
        return vf.UserMessage(role=_msg_role(m), content=new_content)


def _msg_get(m, key, default=None):
    if isinstance(m, dict):
        return m.get(key, default)
    return getattr(m, key, default)


def _sanitize_assistant_content(messages: list) -> list:
    """Coerce any assistant message with null/empty content and no tool_calls
    to a non-null string so strict OpenAI-compatible endpoints accept the
    history.

    A "thinking" model (e.g. Qwen3.5) can emit a turn that is pure
    ``reasoning_content`` with ``content=None`` and ``tool_calls=None``. When
    that message is re-sent as history, Prime Inference returns HTTP 422:
    "content is required unless an assistant message includes tool_calls or
    function_call". We replace the null content with the message's
    ``reasoning_content`` (so no signal is lost) or a single space placeholder.
    Messages that already carry content or tool_calls are returned untouched.
    """
    out = []
    for m in messages:
        if _msg_role(m) == "assistant":
            content = _msg_get(m, "content", None)
            tool_calls = _msg_get(m, "tool_calls", None)
            func_call = _msg_get(m, "function_call", None)
            empty = content is None or (isinstance(content, str) and content.strip() == "")
            if empty and not tool_calls and not func_call:
                reasoning = _msg_get(m, "reasoning_content", None)
                replacement = reasoning if isinstance(reasoning, str) and reasoning.strip() else " "
                out.append(_replace_content(m, replacement))
                continue
        out.append(m)
    return out


def _one_line_summary(content: str, turn_distance: int) -> str:
    """Squash a full obs_text into one line. Heuristics:
       - Keep the STATUS line ("HP: x/y AC: z Dlvl: d Turn: t ...") if present.
       - Keep any [autohalt: ...] / [...] feedback prefix.
       - Otherwise just emit a placeholder.

    IMPORTANT (bug-fix 2026-05-16): `get_prompt_messages` walks the FULL
    chat history every turn, so already-compacted messages get re-fed into
    this function each turn. Previously, the loop would re-pick the
    `[turn -N]` label as `feedback` and prepend a new `[turn -K]` to it
    each round — after many turns, the message becomes a useless chain
    like "[turn -92] [turn -91] [turn -90] ... [turn -7]" with no content.
    Now we detect already-compacted messages and emit a single fresh
    label, dropping the chain. Idempotent.
    """
    stripped_content = content.strip()
    looks_compacted = (
        stripped_content.startswith("[turn -")
        and "=== " not in stripped_content
        and "MAP" not in stripped_content
    )
    if looks_compacted:
        # Already compacted: extract whatever feedback/status we saved earlier
        # and re-emit with the fresh distance label. Without this, every
        # subsequent compaction round drops the [Moved S.] / [Picked up]
        # marker, erasing the agent's action audit-log.
        feedback_part = ""
        hp_part = ""
        # Drop the leading "[turn -N] " then scan the remainder.
        remainder = re.sub(r"^\[turn -\d+\]\s*", "", stripped_content)
        # Feedback is a short bracketed token like "[Moved S.]" or "[Picked up]"
        fb_match = re.match(r"(\[[^\[\]]{1,80}\])\s*", remainder)
        if fb_match and "[turn -" not in fb_match.group(1):
            feedback_part = fb_match.group(1)
            remainder = remainder[fb_match.end():]
        # Status line: "HP: x/y AC: z ..."
        hp_match = re.search(r"HP:\s*\d+/\d+[^\n]*", remainder)
        if hp_match:
            hp_part = hp_match.group(0).strip()
        parts = [f"[turn -{turn_distance}]"]
        if feedback_part: parts.append(feedback_part)
        if hp_part: parts.append(hp_part)
        return " ".join(parts)

    status_line = ""
    feedback = ""
    for line in content.splitlines()[:25]:  # cap scan; obs is short prefix
        line = line.strip()
        # Only treat short bracketed lines as feedback — not chained turn labels.
        if line.startswith("[") and line.endswith("]") and len(line) < 200 and "[turn -" not in line:
            feedback = line
        if line.startswith("HP: "):
            status_line = line
            break
    parts = [f"[turn -{turn_distance}]"]
    if feedback:
        parts.append(feedback)
    if status_line:
        parts.append(status_line)
    return " ".join(parts)


def _check_halt_condition(raw_obs, hp_before: int) -> Optional[str]:
    """Per-step halt for multi-action skills. Returns a short reason string
    if the model should regain control NOW, or None to continue.

    Conditions:
      - HP dropped by ≥25% of the pre-skill value (we're being hit).
      - Hunger blstat indicates Weak/Fainting (need to eat).
      - HP/maxHP < 0.3 (precarious situation).
    """
    try:
        # NLE blstats indices: 10=HP, 11=maxHP, 21=hunger (0=Satiated/1=Normal/...)
        # See nle/nethack/nethack.py:BLStats.
        blstats = raw_obs.get("blstats") if isinstance(raw_obs, dict) else None
        if blstats is None:
            return None
        hp = int(blstats[10])
        max_hp = max(int(blstats[11]), 1)
        hunger = int(blstats[21]) if len(blstats) > 21 else 1
    except (KeyError, IndexError, TypeError):
        return None

    if hp_before > 0 and hp <= hp_before * 0.75:
        return f"HP dropped {hp_before}→{hp}"
    if hp <= max_hp * 0.3:
        return f"HP critical ({hp}/{max_hp})"
    if hunger >= 4:  # Weak (4) or Fainting (5) or Starving (6)
        return f"hunger level {hunger}"
    return None


BELIEF_STATE_INTERVAL = 25
"""Every N turns, call SubLM.summarize on the journal+status and store the
result as a `belief_state:<turn>` note. Allows history-compaction to drop
older turns without losing semantic context. Survey recommendation #3."""


def _maybe_belief_state_summary(state: dict) -> None:
    """Periodic belief-state distillation. Best-effort — silently skips on
    SubLM error so it never breaks a rollout.

    When the configured sub_lm is the OfflineSubLM stub (default), we skip
    the stub call and record a concrete status snapshot instead. The stub's
    "[offline-summary] ..." output isn't useful to the agent; a status
    snapshot at least surfaces HP/dlvl/turn at a known prior moment.
    """
    journal = state.get("journal")
    if journal is None:
        return
    try:
        s = state.get("structured_obs")
        turn = state.get("turn_count", 0)
        # Concrete status snapshot — useful regardless of SubLM backend.
        if s is not None:
            status_snap = (
                f"HP {s.status.get('hitpoints','?')}/{s.status.get('max_hitpoints','?')} "
                f"AC {s.status.get('armor_class','?')} "
                f"Dlvl {s.status.get('depth','?')} "
                f"Turn {s.status.get('time','?')} "
                f"max_dlvl={state.get('max_dlvl_reached','?')} "
                f"descents={state.get('descent_count',0)}"
            )
        else:
            status_snap = "(no obs)"

        # If a real (non-Offline) SubLM is wired, use its richer summary.
        sub_lm = state.get("sub_lm")
        from nethack_harness.tools.code_mode import OfflineSubLM
        if sub_lm is not None and not isinstance(sub_lm, OfflineSubLM):
            ctx_lines = [status_snap]
            for k, v in journal.notes.items():
                ctx_lines.append(f"- {k}: {v}")
            ctx = "\n".join(ctx_lines)
            try:
                summary = sub_lm.summarize(ctx, query=f"belief state at turn {turn}")
                journal.add_note(f"belief_state:t{turn}", summary)
                return
            except Exception:
                pass  # fall through to status snapshot

        journal.add_note(f"belief_state:t{turn}", status_snap)
    except Exception:
        pass


def _maybe_distill(state: dict, prior_dlvl: int) -> None:
    """Belief-state distillation hook. Calls the SubLM (default: Offline)
    to summarize what happened on `prior_dlvl` and adds it to the journal
    as `dlvl_<n>_summary`. Cheap when the SubLM is offline; nontrivial
    when wired to a real inference server.
    """
    journal = state.get("journal")
    if journal is None:
        return
    try:
        from nethack_harness.tools.code_mode import _default_sub_lm
        sub_lm = state.get("sub_lm") or _default_sub_lm()
        ctx_lines = []
        for k, v in journal.notes.items():
            ctx_lines.append(f"- {k}: {v}")
        ctx = "\n".join(ctx_lines) if ctx_lines else "(no notes recorded on this level)"
        summary = sub_lm.summarize(ctx, query=f"key events on dlvl {prior_dlvl}")
        journal.add_note(f"dlvl_{prior_dlvl}_summary", summary)
    except Exception:
        # Distillation is best-effort; don't break the rollout.
        pass


def _to_action_indices(env: NetHackCoreEnv, actions: list[int]) -> list[int]:
    """Normalize skill action values to the keystroke bytes the engine consumes.

    The semantic action enums ARE keystrokes (CompassDirection.N == 107 ==
    ord('k'), MiscAction.MORE == 13, ...) and ``EngineEnv.step`` takes those
    bytes directly -- so there is no longer an action-index translation layer.
    This is an identity pass-through (kept as a named seam, and so callers don't
    have to change). ``env`` is unused but retained for signature stability.
    """
    return [int(a) for a in actions]




# ---------- rewards ----------

@vf.reward(weight=1.0)
async def scout_reward(state: vf.State) -> float:
    """
    Total normalized tiles scouted across the rollout.

    Verifiers' `Rubric.score_rollout` runs once at end of rollout. Returning
    the last step's `scout_delta` alone effectively reports only the very
    last action's exploration — every prior tile discovery is invisible to
    the eval harness. So env_response accumulates `scout_reward_total` each
    step (delta/1000) and we return that running sum. Old call sites that
    only set `scout_delta` (unit tests) still work as a fallback.
    """
    if "scout_reward_total" in state:
        return float(state["scout_reward_total"])
    return float(state.get("scout_delta", 0)) / 1000.0


@vf.reward(weight=10.0)
async def descent_reward(state: vf.State) -> float:
    """+1 per new dungeon level reached this episode.

    env_response increments `descent_count` whenever max_dlvl advances; we
    return the running tally. (A per-step comparison here would always read
    `depth == max_dlvl_reached` because env_response updates max_dlvl first.)
    """
    if "descent_count" in state:
        return float(state["descent_count"])
    # Back-compat for unit tests that set max_dlvl_reached + structured_obs.
    s = state.get("structured_obs")
    if s is None:
        return 0.0
    dlvl = s.status.get("depth", 1)
    if dlvl > state["max_dlvl_reached"]:
        state["max_dlvl_reached"] = dlvl
        return 1.0
    return 0.0


@vf.reward(weight=100.0)
async def success_reward(state: vf.State) -> float:
    """
    +1 if the tier's success_milestone fired this episode. Distinct from
    `ascension_reward` (which weight=1000) because milestone success is a
    rung on the curriculum, not the endgame.
    """
    return 1.0 if state.get("succeeded") else 0.0


@vf.reward(weight=1000.0)
async def ascension_reward(state: vf.State) -> float:
    """
    The big one. +1 when the player ascends (escapes the dungeon with the
    Amulet of Yendor), 0 otherwise. _detect_terminal_outcome in env_response
    sets state["ascended"]=True when the standard ascension messages appear.
    """
    return 1.0 if state.get("ascended") else 0.0


# ---------- terminal outcome detection ----------

# NetHack-3.6 prints these phrases on the final screen. We treat any of them
# as proof of ascension; the death case is symmetric (a player who didn't
# ascend but did die has state["died"]=True so we can attribute the outcome).
_ASCENSION_MARKERS = (
    "ascended to demigod",
    "ascended to demigoddess",
    "with the Amulet",
    "offered the Amulet",
)
_DEATH_MARKERS = (
    "killed by",
    "starved to death",
    "petrified by",
    "drowned",
    "quit the game",
    "Do you want your possessions identified",
)


def _decode_tty(obs) -> str:
    """Render the full tty into a single string for marker-scanning."""
    return "\n".join(
        "".join(chr(c) for c in row) for row in obs.tty_chars
    )


def _detect_terminal_outcome(obs, state: dict) -> None:
    """
    Inspect the last observation to determine death vs ascension. Mutates
    state in place. Called every step (cheap) so the reward functions can
    read precomputed booleans.
    """
    state.setdefault("ascended", False)
    state.setdefault("died", False)
    if state["ascended"] or state["died"]:
        return  # terminal outcomes are absorbing

    screen = _decode_tty(obs)
    if any(m in screen for m in _ASCENSION_MARKERS):
        state["ascended"] = True
        state["terminated"] = True
        return
    if any(m in screen for m in _DEATH_MARKERS):
        state["died"] = True
        state["terminated"] = True




def _code_tool_adapter():
    """The single 'code' tool exposed in interface='code' mode."""
    def code(source: str) -> str:
        """Execute Python against the `nh` namespace.

        Available: nh.move/attack/descend/search/pickup/move_to/autoexplore,
        nh.add_note/recall, nh.wiki_lookup/wiki_search, nh.status/inventory/
        map_view/character. Constants: Direction.{N,NE,E,SE,S,SW,W,NW,WAIT},
        Position(x, y). Imports and dunder access are blocked. Stdout returns
        as the tool result. 5s wallclock cap.
        """
        return ""  # never called directly; env_response routes the source.

    return code


def _build_skill_adapter_callables(skill_set: str = "full") -> list:
    """
    Build one callable per registered skill with the right __name__, doc, and
    annotations so verifiers' tool-schema introspection works.

    The callables are stubs: calling them raises (we want a loud error if
    something ever bypasses `env_response` and tries to invoke them).
    """
    import inspect
    from typing import Optional as _Opt

    # Skills the harness owns (never exposed as agent tools). Menu/inventory
    # selection is auto-dismissed in env_response; eat/quaff/read take an
    # `item` arg and bundle the selection in-skill. Exposing these as agent
    # tools caused Qwen3.5-9B to spend 42% of turns on spurious menu calls.
    _HARNESS_OWNED = {"inventory_item", "menu_option"}

    # skill_set: 'full' (default), 'move' (only move + survival), 'dir8'
    # (8 single-direction tools + survival, no `move` aggregator), or a
    # comma-separated whitelist e.g. 'move,descend,search'. The ladder
    # exists to measure how much "free reasoning" each helper-skill
    # offloads from the agent. dir8 is the most-faithful NLE baseline.
    if skill_set == "dir8":
        # Single-direction tools (N/NE/.../NW) + descend + search +
        # pickup + attack + survival. NO move/move_to/autoexplore/
        # find_and_descend/kick aggregators. Strips all "free" pathfinding.
        keep = {"descend", "search", "pickup", "attack",
                "engrave_elbereth", "pray", "eat", "quaff", "read",
                "add_note", "recall", "pin_objective",
                "wiki_lookup", "wiki_search"}
        out = []
        # Generate 8 direction skill-adapters by binding `move(direction=...)`
        # to a fixed direction. Naming: `north`, `northeast`, etc.
        _DIR_NAMES = [("north","N"),("northeast","NE"),("east","E"),
                      ("southeast","SE"),("south","S"),("southwest","SW"),
                      ("west","W"),("northwest","NW")]
        for tname, dir_canon in _DIR_NAMES:
            out.append(_make_fixed_direction_adapter(tname, dir_canon))
        for name, schema in skill_registry.all_schemas().items():
            if name in _HARNESS_OWNED: continue
            if name not in keep: continue
            params = schema.get("parameters", {}) or {}
            out.append(_make_skill_adapter(name, schema.get("description", ""), params))
        return out
    elif skill_set == "netplay":
        # NetPlay (Jeurissen, CoG 2024): a skill-only action surface with NO
        # low-level `move(direction=...)` primitive — the agent acts through
        # high-level pathfinding (move_to/autoexplore/find_and_descend) plus
        # interactions. The standardized action set for cross-encoding
        # benchmarks (hold actions fixed, vary the observation).
        # explore_and_descend supersedes the weaker open-loop autoexplore /
        # find_and_descend (which bump on doors/corridors and don't loop), so we
        # drop those — otherwise the LLM defaults to the familiar weak tools and
        # never calls the robust one. move_to stays for precise single-target moves.
        keep = {"move_to", "explore_and_descend",
                "attack", "throw", "descend", "search", "pickup", "engrave_elbereth", "pray",
                "eat", "quaff", "read", "kick", "add_note", "recall",
                "pin_objective", "wiki_lookup", "wiki_search"}
        # NB: the 1d observation-delivery tools (reveal / request_map) are
        # deliberately NOT in the default netplay set — adding them would change
        # the fixed action surface for every netplay experiment (Exp 1, 1b, 1c)
        # and confound the comparison. The 1d arms expose them via an explicit
        # comma-`skill_set` (e.g. "<netplay tools>,reveal,request_map").
        out = []
        for name, schema in skill_registry.all_schemas().items():
            if name in _HARNESS_OWNED: continue
            if name not in keep: continue
            params = schema.get("parameters", {}) or {}
            out.append(_make_skill_adapter(name, schema.get("description", ""), params))
        return out
    elif skill_set == "move":
        # `move(direction=...)` + survival, but NO move_to, NO autoexplore,
        # NO find_and_descend. Single-step movement only, agent reasons
        # about which direction. Slightly above dir8 since the LM picks
        # a direction string instead of a fixed tool.
        keep = {"move", "descend", "search", "pickup", "attack",
                "engrave_elbereth", "pray", "eat", "quaff", "read",
                "add_note", "recall", "pin_objective",
                "wiki_lookup", "wiki_search"}
        out = []
        for name, schema in skill_registry.all_schemas().items():
            if name in _HARNESS_OWNED: continue
            if name not in keep: continue
            params = schema.get("parameters", {}) or {}
            out.append(_make_skill_adapter(name, schema.get("description", ""), params))
        return out
    elif "," in skill_set:
        keep = {s.strip() for s in skill_set.split(",")}
        out = []
        for name, schema in skill_registry.all_schemas().items():
            if name in _HARNESS_OWNED: continue
            if name not in keep: continue
            params = schema.get("parameters", {}) or {}
            out.append(_make_skill_adapter(name, schema.get("description", ""), params))
        return out
    # default 'full'
    out = []
    for name, schema in skill_registry.all_schemas().items():
        if name in _HARNESS_OWNED:
            continue
        params = schema.get("parameters", {}) or {}
        out.append(_make_skill_adapter(name, schema.get("description", ""), params))
    return out


def _make_run_macro_adapter():
    """Tool stub for variant=CH `run_macro(name=...)`. The actual dispatch
    lives in NetHackVerifiersEnv.env_response — this exists only to surface
    the tool to the verifiers schema-introspection layer."""
    def run_macro(name: str):
        """Run a Continual-Harness macro (a Refiner-registered sequence of
        skill calls). The macro must already exist; ask `recall` or check
        the system message for available macro names."""
        return None
    return run_macro


def _make_fixed_direction_adapter(tool_name: str, direction: str):
    """Bind `move(direction=...)` to a specific direction → 1-arg tool.

    Returns a callable named `tool_name` (north/northeast/.../northwest) with
    no parameters; calling it dispatches `move(direction=direction)` through
    the registry. Used by `skill_set='dir8'` baseline.
    """
    def _fn():
        """One step in this direction (NLE primitive)."""
        return None
    _fn.__name__ = tool_name
    _fn.__doc__ = f"Take one NLE step {direction}. No A*, no aggregation — single primitive action."
    return _fn


_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


def _make_skill_adapter(name: str, description: str, params: dict):
    """Create a callable that exposes the schema verifiers expects."""
    import inspect

    # Build a signature with parameters in declared order.
    sig_params = []
    annotations: dict = {}
    for pname, pschema in params.items():
        ptype = _TYPE_MAP.get(pschema.get("type", "string"), str)
        annotations[pname] = ptype
        default = pschema.get("default", inspect.Parameter.empty)
        sig_params.append(inspect.Parameter(
            pname,
            inspect.Parameter.KEYWORD_ONLY,
            default=default,
            annotation=ptype,
        ))

    def _adapter(**kwargs):
        raise RuntimeError(
            f"Skill adapter {name!r} was invoked directly; this should not "
            "happen — env_response dispatches via skill_registry.call()."
        )

    _adapter.__name__ = name
    _adapter.__doc__ = description
    _adapter.__signature__ = inspect.Signature(parameters=sig_params)
    _adapter.__annotations__ = annotations
    return _adapter
