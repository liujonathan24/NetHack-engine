"""
nethack_harness.tools.skills
===================

NetPlay-style skill mode. Each skill is a callable that takes the current
NetHackCoreEnv and arguments, and returns a sequence of primitive NLE actions
plus a feedback message.

We use a registry pattern so the verifiers wrapper can introspect available
skills to auto-generate OpenAI tool schemas. This is the layer Alex's pysc2
analogy points at: the skill set is the agent's API to the world.

The full skill catalog is intentionally small in v0. Add to it as you
characterize what LLM agents actually need. Don't blindly mirror autoascend.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, Optional

from nethack_core import actions as nethack

from nethack_core import NetHackCoreEnv
from nethack_harness.memory.journal import Journal
from nethack_core import StructuredObservation, InventoryItem
from nethack_harness.navigation.pathfinding import (
    a_star,
    is_walkable,
    nearest_frontier,
    player_xy,
    reachable_set,
)


Direction = Literal["N", "NE", "E", "SE", "S", "SW", "W", "NW", "."]


# Map common hallucinated / synonym tool names (observed in real eval traces)
# onto the canonical registered skill. The biggest offender: the agent calls
# `explore` (not a real skill) over and over, never descends, and burns a whole
# rollout. Aliasing `explore` -> `explore_and_descend` makes that hallucination
# do the right thing instead of erroring into a loop. Keep this small and
# obvious — it is a self-correction nudge, not a second skill catalog.
_SKILL_ALIASES: dict[str, str] = {
    "explore": "explore_and_descend",
    "explore_map": "autoexplore",
    "auto_explore": "autoexplore",
    "go_down": "descend",
    "descend_stairs": "descend",
    "down": "descend",
    "go_up": "ascend",
    "ascend_stairs": "ascend",
    "up": "ascend",
    "climb": "ascend",
    "goto": "move_to",
    "move_to_tile": "move_to",
    "travel": "move_to",
    "fight": "attack",
    "search_for_traps": "search",
    "look": "search",
}


# Semantic action ids for directional movement -- the int value of each member
# is the keystroke the engine consumes (see nethack_core.actions).
_DIRECTION_TO_ACTION = {
    "N": nethack.CompassDirection.N,
    "NE": nethack.CompassDirection.NE,
    "E": nethack.CompassDirection.E,
    "SE": nethack.CompassDirection.SE,
    "S": nethack.CompassDirection.S,
    "SW": nethack.CompassDirection.SW,
    "W": nethack.CompassDirection.W,
    "NW": nethack.CompassDirection.NW,
    ".": nethack.MiscDirection.WAIT,
}


@dataclass
class SkillResult:
    """What a skill returns to the harness layer."""
    actions: list[int]            # NLE primitive action ids to step()
    feedback: str                 # human-readable summary
    interrupted: bool = False     # True if the skill self-terminated early
    # For non-NLE skills (journal etc.) `actions` is empty and the side
    # effect is in `journal_op` so the harness can apply it after the call.
    journal_op: Optional[Callable[[Journal], str]] = None
    # Closed-loop skills (e.g. explore_and_descend) step the env THEMSELVES in
    # a re-observe loop, so there is nothing left for env_response to execute.
    # When `pre_executed` is True, env_response skips its step loop and uses the
    # reward / final obs the skill already produced.
    pre_executed: bool = False
    pre_reward: float = 0.0
    final_obs: Any = None
    pre_terminated: bool = False
    pre_truncated: bool = False


class SkillRegistry:
    """Holds the catalog of skills and their schemas."""

    def __init__(self) -> None:
        self._skills: dict[str, Callable] = {}
        self._schemas: dict[str, dict] = {}

    def register(self, name: str, schema: dict):
        def decorator(fn):
            self._skills[name] = fn
            self._schemas[name] = schema
            return fn
        return decorator

    def call(self, _skill_name: str, _env: NetHackCoreEnv, _obs: StructuredObservation, /, **kwargs) -> SkillResult:
        # Positional-only (`/`) parameters with underscore-prefixed names so a
        # model passing `{"name": ...}`, `{"env": ...}`, or `{"obs": ...}` as
        # tool args can't collide with our dispatcher signature. (v0.0.37 crash:
        # Qwen3.5-9B sent {"name": ...} → "got multiple values for argument 'name'".)
        name, env, obs = _skill_name, _env, _obs
        # Also strip these out in case some caller forwards them.
        for reserved in ("_skill_name", "_env", "_obs", "name", "env", "obs"):
            kwargs.pop(reserved, None)
        if name not in self._skills:
            # First, try to resolve a common hallucinated / synonym name to a
            # real skill (e.g. `explore` -> `explore_and_descend`). If it maps
            # to something registered, dispatch that skill transparently.
            aliased = _SKILL_ALIASES.get(name)
            if aliased is not None and aliased in self._skills:
                name = aliased
            else:
                # Still unknown: return a clear, self-correcting "no such tool"
                # message (no actions) so the agent can pick a valid tool —
                # never raise KeyError, which silently loops the rollout.
                import difflib
                valid = sorted(self._skills.keys())
                close = difflib.get_close_matches(name, valid, n=3, cutoff=0.5)
                suggestion = f" Did you mean: {', '.join(close)}?" if close else ""
                # NOTE: "Unknown skill" is retained for backward-compatible
                # callers/tests that key off that phrase.
                feedback = (
                    f"No tool named '{name}'. Unknown skill. "
                    f"Valid tools: {', '.join(valid)}.{suggestion}"
                )
                return SkillResult(actions=[], feedback=feedback, interrupted=True)
        fn = self._skills[name]
        # Defensive: small models (e.g. Qwen3.5-0.8B) sometimes emit malformed
        # tool calls where the arg dict is wrapped like {"arguments": "..."} or
        # contains stray keys. Filter kwargs to the function's real signature
        # so a bad call produces feedback rather than a crash.
        import inspect
        try:
            sig = inspect.signature(fn)
            accepted = {k for k, p in sig.parameters.items() if p.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)}
            ignored = set(kwargs) - accepted
            kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        except (TypeError, ValueError):
            ignored = set()
        # Type coercion at the boundary: small models often send `{"index": "5"}`
        # (string for int) or `{"x": 12.0}` (float for int). Coerce per the
        # schema's declared param types so the skill doesn't TypeError on
        # cosmetic shape mismatches.
        schema_params = self._schemas.get(name, {}).get("parameters", {}) or {}
        for pname, pschema in schema_params.items():
            if pname not in kwargs:
                continue
            v = kwargs[pname]
            target = pschema.get("type")
            try:
                if target == "integer" and not isinstance(v, bool):
                    kwargs[pname] = int(v) if isinstance(v, (int, float, str)) else v
                elif target == "number":
                    kwargs[pname] = float(v) if isinstance(v, (int, float, str)) else v
                elif target == "string":
                    kwargs[pname] = str(v) if not isinstance(v, str) else v
            except (TypeError, ValueError):
                pass  # leave as-is; the skill will surface a friendlier error
        try:
            result = fn(env, obs, **kwargs)
        except (TypeError, AttributeError, ValueError) as e:
            # ValueError covers a skill doing int()/float() on a malformed model
            # arg (e.g. a small model leaking XML tool syntax: times="17\n</parameter").
            # Surface a friendly invalid-action message and re-prompt rather than
            # letting the exception crash the whole rollout/eval.
            return SkillResult(actions=[], feedback=f"Skill {name} call failed: {e}. Schema: {self._schemas.get(name, {})}", interrupted=True)
        if ignored:
            extra = f"[ignored unknown args: {sorted(ignored)}]"
            result = SkillResult(actions=result.actions, feedback=f"{extra} {result.feedback or ''}".strip(), journal_op=result.journal_op, interrupted=result.interrupted)
        return result

    def all_schemas(self) -> dict[str, dict]:
        return dict(self._schemas)


registry = SkillRegistry()


# ---------- movement / exploration ----------

_DIRECTION_ALIASES = {
    "north": "N", "northeast": "NE", "east": "E", "southeast": "SE",
    "south": "S", "southwest": "SW", "west": "W", "northwest": "NW",
    "up": "N", "down": "S", "left": "W", "right": "E", "wait": ".",
}


def _normalize_direction(d: str) -> Optional[str]:
    """Normalize a direction string to the canonical N/NE/.../NW/'.' form.

    Accepts canonical form, lowercase, full names ('north'), and aliases
    ('up' = N, 'down' = S, 'left' = W, 'right' = E). Small models reliably
    emit these instead of the compact form; being strict costs the model
    a wasted turn for a cosmetic mismatch.
    """
    if not isinstance(d, str):
        return None
    s = d.strip()
    upper = s.upper()
    if upper in _DIRECTION_TO_ACTION:
        return upper
    lower = s.lower()
    if lower in _DIRECTION_ALIASES:
        return _DIRECTION_ALIASES[lower]
    return None


@registry.register("move", schema={
    "description": (
        "Move in a compass direction. By default runs in that direction until an "
        "obstacle is encountered (walls, monsters, corridor branches) — like "
        "NetHack's shift+direction 'run' command. This matches what experienced "
        "human players do for traversal. Pass run=False for a single step."
    ),
    "parameters": {
        "direction": {"type": "string", "enum": list(_DIRECTION_TO_ACTION.keys()), "description": "Compass direction (N/NE/E/SE/S/SW/W/NW) or '.' to wait. Aliases like 'north'/'up' are also accepted."},
        "run": {"type": "boolean", "default": True, "description": "If True (default), queue up to 25 steps in this direction; NLE will stop at first obstacle. If False, single step."},
    },
})
def move(env: NetHackCoreEnv, obs: StructuredObservation, direction: str, run: bool = True) -> SkillResult:
    canon = _normalize_direction(direction)
    if canon is None:
        return SkillResult([], f"Invalid direction: {direction!r}. Use N/NE/E/SE/S/SW/W/NW (or 'wait').", interrupted=True)
    step = int(_DIRECTION_TO_ACTION[canon])
    # SHORT-CIRCUIT: if `>` is visible and reachable, divert this move() to
    # a path-to-stairs + descend. This rescues rollouts where the LM saw
    # stairs in VISIBLE FEATURES but tried to micromanage with single
    # move() calls instead of calling find_and_descend/autoexplore.
    try:
        chars2, start2 = _current_chars_and_player(env)
        h2, w2 = chars2.shape
        stair2 = None
        for yy in range(h2):
            for xx in range(w2):
                if int(chars2[yy, xx]) == ord('>'):
                    stair2 = (xx, yy); break
            if stair2: break
        if stair2 is not None and stair2 != start2:
            p2 = a_star(chars2, start2, stair2)
            if p2 and len(p2) <= 60:
                acts = _enum_actions_to_indices(env, p2)
                # Keystrokes consumed directly by the engine: MORE dismisses a
                # prompt, then DOWN ('>') descends.
                acts = acts + [int(nethack.MiscAction.MORE), int(nethack.MiscDirection.DOWN)]
                return SkillResult(
                    actions=acts,
                    feedback=(
                        f"move({canon}) — diverted: `>` visible at {stair2}, "
                        f"pathing {len(p2)} steps and descending."
                    ),
                )
    except Exception:
        pass
    if not run or canon == ".":
        return SkillResult([step], f"Moved {canon}.")
    # Compute path via a_star to a tile far in this direction — gives us
    # an obstacle-aware "run" that auto-stops at walls and corridor turns.
    try:
        chars, start = _current_chars_and_player(env)
        h, w = chars.shape
        dx_map = {"N":(0,-1),"NE":(1,-1),"E":(1,0),"SE":(1,1),
                  "S":(0,1),"SW":(-1,1),"W":(-1,0),"NW":(-1,-1)}
        dx, dy = dx_map[canon]
        # Step forward until obstacle or edge.
        cx, cy = start
        path = []
        for _ in range(25):
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < w and 0 <= ny < h): break
            tile = int(chars[ny, nx])
            from nethack_harness.navigation.pathfinding import is_walkable
            if not is_walkable(tile): break
            path.append(step)
            cx, cy = nx, ny
            # Stop at items / interesting tiles so we don't run past them.
            if tile in (ord('>'), ord('<'), ord('_'), ord('{'), ord('+'),
                        ord('%'), ord('$'), ord('('), ord(')'), ord('['),
                        ord('*'), ord('?'), ord('!'), ord('/'), ord('='),
                        ord('"')):
                break
        if not path:
            # First step blocked — return single step so feedback reports it.
            return SkillResult([step], f"Moved {canon} (blocked at first step).")
        return SkillResult(path, f"Ran {canon} for {len(path)} steps.")
    except Exception:
        return SkillResult([step], f"Moved {canon} (run setup failed).")


@registry.register("attack", schema={
    "description": (
        "Melee attack an adjacent monster in the given direction. Do NOT call "
        "if the target shows `[PET — don't attack]` in ADJACENT — attacking a "
        "pet damages alignment. Walk around pets instead."
    ),
    "parameters": {"direction": {"type": "string", "enum": list(_DIRECTION_TO_ACTION.keys())}},
})
def attack(env: NetHackCoreEnv, obs: StructuredObservation, direction: str) -> SkillResult:
    # In NetHack, melee = move toward the monster. Same action.
    # We separate it as a skill for legibility in traces.
    canon = _normalize_direction(direction)
    if canon is None:
        return SkillResult([], f"Invalid direction: {direction!r}.", interrupted=True)
    # Sanity check: is there actually a letter glyph adjacent in this
    # direction? If not, the agent is "attacking" empty space — likely a
    # misread of the map. Warn but still pass the action through (NetHack
    # will just walk one step, harmless).
    adj = getattr(obs, "adjacent", None) or {}
    tile = adj.get(canon, "")
    target_warning = ""
    if tile and not (len(tile) >= 1 and tile[0].isalpha() and tile[0] != "@"):
        target_warning = (
            f" (note: ADJACENT shows {canon}={tile!r}, no monster there — "
            "this will just walk forward.)"
        )
    result = move(env, obs, direction=direction)
    if target_warning and result.feedback:
        result = SkillResult(actions=result.actions, feedback=result.feedback + target_warning, interrupted=result.interrupted)
    return result


@registry.register("descend", schema={
    "description": (
        "Descend the down-staircase. You MUST be standing on a '>' tile. "
        "Check `=== UNDER PLAYER ===` first — it should say 'stairs DOWN (>)'. "
        "If it says 'stairs UP (<)' you'll go BACK to a previous level instead. "
        "If it says 'floor' or anything else, this call wastes a turn."
    ),
    "parameters": {},
})
def descend(env: NetHackCoreEnv, obs: StructuredObservation) -> SkillResult:
    # In the curriculum env a '>' triggers a positional curriculum jump (the
    # hero lands at a random spot, not on a stair), so skip the on-stair gate.
    if hasattr(env, "curriculum_position"):
        return SkillResult([int(nethack.MiscDirection.DOWN)],
                           "Curriculum: descended to the next level.")
    # Friendlier feedback: if under_player exists and isn't `>`, short-circuit
    # with an explanatory message so the agent doesn't waste a turn AND
    # doesn't get a generic "couldn't descend" message.
    under = getattr(obs, "under_player", None)
    if under and not under.startswith("stairs DOWN"):
        return SkillResult(
            [],
            f"Can't descend — you're standing on: {under}. Find a '>' tile and step ON it first.",
            interrupted=True,
        )
    # The engine consumes keystroke bytes directly: MORE (13) dismisses any
    # prompt, then DOWN ('>', 62) descends. No action-index translation.
    return SkillResult([int(nethack.MiscAction.MORE), int(nethack.MiscDirection.DOWN)],
                       "Attempted to descend.")


@registry.register("ascend", schema={
    "description": (
        "Ascend the up-staircase. You MUST be standing on a '<' tile. "
        "Check `=== UNDER PLAYER ===` first — it should say 'stairs UP (<)'. "
        "If it says 'stairs DOWN (>)' you'll go DOWN to a deeper level instead. "
        "If it says 'floor' or anything else, this call wastes a turn."
    ),
    "parameters": {},
})
def ascend(env: NetHackCoreEnv, obs: StructuredObservation) -> SkillResult:
    # Curriculum env: '<' advances the curriculum's ascent path positionally.
    if hasattr(env, "curriculum_position"):
        return SkillResult([int(nethack.MiscDirection.UP)],
                           "Curriculum: ascended to the next level.")
    # Mirror of descend(): friendly short-circuit when not on an upstair.
    under = getattr(obs, "under_player", None)
    if under and not under.startswith("stairs UP"):
        return SkillResult(
            [],
            f"Can't ascend — you're standing on: {under}. Find a '<' tile and step ON it first.",
            interrupted=True,
        )
    # MORE (13) dismisses any prompt, then UP ('<', 60) ascends. The
    # CurriculumEnv intercepts UP to advance the curriculum's ascent path.
    return SkillResult([int(nethack.MiscAction.MORE), int(nethack.MiscDirection.UP)],
                       "Attempted to ascend.")


@registry.register("search", schema={
    "description": (
        "Search adjacent tiles for hidden passages and traps. Pass `times` "
        "to repeat — hidden passages typically need 5-10 searches at the same "
        "tile to reveal. Defaults to 1."
    ),
    "parameters": {
        "times": {
            "type": "integer",
            "default": 1,
            "description": "Number of consecutive search actions (1-20).",
        },
    },
})
def search(env: NetHackCoreEnv, obs: StructuredObservation, times: int = 1) -> SkillResult:
    try:
        n = int(times)
    except (TypeError, ValueError):
        n = 1
    n = max(1, min(20, n))
    return SkillResult([int(ord('s'))] * n, f"Searched x{n}." if n > 1 else "Searched.")


# ---------- survival actions (eat / quaff / read / pray / engrave / kick / throw) ----------
#
# These press a single command key. NetHack will then prompt for inventory
# selection if needed; the agent answers via `inventory_item` on the next turn.

def _find_inv_letter(obs: StructuredObservation, item: Optional[str], kinds: tuple[str, ...]) -> tuple[Optional[str], str]:
    """Resolve `item` (substring or letter) → inventory letter. Returns (letter, feedback).
    Letter is None if no match — feedback explains why.
    `kinds`: lowercase keyword filter (e.g. ('food',) for eat). Empty tuple = any."""
    inv = obs.inventory or []
    candidates = [it for it in inv if not kinds or any(k in it.description.lower() for k in kinds)]
    if not candidates:
        return None, f"No matching items in inventory (looked for: {', '.join(kinds) or 'any'})."
    if not item:
        listing = ", ".join(f"{c.letter}={c.description}" for c in candidates[:6])
        return None, f"Specify `item` (letter or substring). Candidates: {listing}"
    item = item.strip()
    # Letter match first.
    if len(item) == 1:
        for c in candidates:
            if c.letter == item:
                return c.letter, f"Selected {c.description}."
    # Substring match against description.
    lo = item.lower()
    for c in candidates:
        if lo in c.description.lower():
            return c.letter, f"Selected {c.description}."
    listing = ", ".join(f"{c.letter}={c.description}" for c in candidates[:6])
    return None, f"No item matched {item!r}. Candidates: {listing}"


@registry.register("eat", schema={
    "description": (
        "Eat an edible item from inventory. Pass `item` = substring of the food's "
        "description (e.g. 'apple', 'food ration') OR its inventory letter. "
        "If no edible items exist or `item` doesn't match, the turn is NOT consumed "
        "and you get feedback listing candidates."
    ),
    "parameters": {"item": {"type": "string", "description": "substring or letter of the food", "default": None}},
})
def eat(env: NetHackCoreEnv, obs: StructuredObservation, item: Optional[str] = None) -> SkillResult:
    letter, feedback = _find_inv_letter(obs, item, ("food", "corpse", "ration", "fruit", "apple", "pancake", "cookie", "tripe"))
    if letter is None:
        return SkillResult([], feedback, interrupted=True)
    return SkillResult([int(ord('e')), int(ord(letter))], feedback)


@registry.register("quaff", schema={
    "description": (
        "Quaff (drink) a potion. Pass `item` = substring or letter. If no potion "
        "matches, the turn is NOT consumed."
    ),
    "parameters": {"item": {"type": "string", "description": "substring or letter of the potion", "default": None}},
})
def quaff(env: NetHackCoreEnv, obs: StructuredObservation, item: Optional[str] = None) -> SkillResult:
    letter, feedback = _find_inv_letter(obs, item, ("potion",))
    if letter is None:
        return SkillResult([], feedback, interrupted=True)
    return SkillResult([int(ord('q')), int(ord(letter))], feedback)


@registry.register("read", schema={
    "description": (
        "Read a scroll or spellbook. Pass `item` = substring or letter. If no "
        "readable matches, the turn is NOT consumed."
    ),
    "parameters": {"item": {"type": "string", "description": "substring or letter of the scroll/book", "default": None}},
})
def read(env: NetHackCoreEnv, obs: StructuredObservation, item: Optional[str] = None) -> SkillResult:
    letter, feedback = _find_inv_letter(obs, item, ("scroll", "spellbook", "book"))
    if letter is None:
        return SkillResult([], feedback, interrupted=True)
    return SkillResult([int(ord('r')), int(ord(letter))], feedback)


@registry.register("pray", schema={
    "description": "Pray to your god (#pray, then 'y' to confirm). Risky if recently used.",
    "parameters": {},
})
def pray(env: NetHackCoreEnv, obs: StructuredObservation) -> SkillResult:
    # '#' enters extended-command mode; 'p','r','a','y' types the command; '\n' submits; 'y' confirms.
    return SkillResult([int(ord('#')), int(ord('p')), int(ord('r')), int(ord('a')), int(ord('y')),
                        int(nethack.MiscAction.MORE), int(ord('y'))], "Prayed.")


@registry.register("engrave_elbereth", schema={
    "description": "Engrave 'Elbereth' in the dust to scare most monsters. Uses E command.",
    "parameters": {},
})
def engrave_elbereth(env: NetHackCoreEnv, obs: StructuredObservation) -> SkillResult:
    # E + - (use finger) + Elbereth + Enter + Enter
    actions = [int(ord('E')), int(ord('-'))]
    for ch in "Elbereth":
        actions.append(int(ord(ch)))
    actions.append(int(nethack.MiscAction.MORE))  # finish text
    actions.append(int(nethack.MiscAction.MORE))  # any prompt close
    return SkillResult(actions, "Engraved Elbereth.")


# vi-style direction keystrokes that NetHack accepts on direction prompts.
_DIRECTION_VI_KEY = {
    "N": "k", "S": "j", "E": "l", "W": "h",
    "NE": "u", "NW": "y", "SE": "n", "SW": "b",
    ".": ".",
}


@registry.register("kick", schema={
    "description": "Kick (ctrl-D) in a direction. Used to break locks/doors. Pass `direction` (N/NE/.../NW).",
    "parameters": {"direction": {"type": "string", "enum": list(_DIRECTION_VI_KEY.keys()),
                                 "description": "Compass direction to kick.", "default": "N"}},
})
def kick(env: NetHackCoreEnv, obs: StructuredObservation, direction: str = "N") -> SkillResult:
    canon = _normalize_direction(direction)
    if canon is None or canon not in _DIRECTION_VI_KEY:
        return SkillResult([], f"Invalid direction: {direction!r}", interrupted=True)
    # Ctrl-D = 0x04, then the vi-style direction key.
    return SkillResult([0x04, int(ord(_DIRECTION_VI_KEY[canon]))], f"Kicked {canon}.")


@registry.register("throw", schema={
    "description": "Throw an item in a direction. Pass `item` (substring/letter) AND `direction`.",
    "parameters": {
        "item": {"type": "string", "description": "substring or letter of the item to throw"},
        "direction": {"type": "string", "enum": list(_DIRECTION_VI_KEY.keys()), "default": "N"},
    },
})
def throw(env: NetHackCoreEnv, obs: StructuredObservation, item: Optional[str] = None, direction: str = "N") -> SkillResult:
    letter, feedback = _find_inv_letter(obs, item, ())  # any inventory item
    if letter is None:
        return SkillResult([], feedback, interrupted=True)
    canon = _normalize_direction(direction)
    if canon is None or canon not in _DIRECTION_VI_KEY:
        return SkillResult([], f"Invalid direction: {direction!r}", interrupted=True)
    return SkillResult(
        [int(ord('t')), int(ord(letter)), int(ord(_DIRECTION_VI_KEY[canon]))],
        f"Threw {feedback} {canon}.",
    )


# ---------- inventory ----------

@registry.register("pickup", schema={
    "description": "Pick up items on the current tile.",
    "parameters": {},
})
def pickup(env: NetHackCoreEnv, obs: StructuredObservation) -> SkillResult:
    return SkillResult([int(ord(','))], "Attempted to pick up.")


@registry.register("inventory_item", schema={
    "description": (
        "Respond to an inventory prompt (e.g. 'What do you want to throw?'). "
        "ONLY use when the observation has a '=== PROMPT: ...' block — that "
        "block lists numbered items; pass the matching 0-based index. "
        "Outside a prompt this returns 'No inventory prompt is open.'"
    ),
    "parameters": {"index": {"type": "integer", "description": "0-based index into the listed prompt items"}},
})
def inventory_item(env: NetHackCoreEnv, obs: StructuredObservation, index: int) -> SkillResult:
    if obs.inventory_prompt is None:
        return SkillResult([], "No inventory prompt is open.", interrupted=True)
    items = obs.inventory_prompt["items"]
    if not (0 <= index < len(items)):
        return SkillResult([], f"Index {index} out of range for {len(items)} choices.", interrupted=True)
    target: InventoryItem = items[index]
    return SkillResult([int(ord(target.letter))], f"Selected {target.description}.")


@registry.register("menu_option", schema={
    "description": (
        "Select a menu option by index. ONLY use when the observation has a "
        "'=== MENU ===' block; otherwise this returns 'No menu is open.' and "
        "you waste a turn. Index is 0-based into the listed options."
    ),
    "parameters": {"index": {"type": "integer", "description": "0-based index into the visible menu"}},
})
def menu_option(env: NetHackCoreEnv, obs: StructuredObservation, index: int) -> SkillResult:
    if obs.menu is None:
        return SkillResult([], "No menu is open.", interrupted=True)
    if not (0 <= index < len(obs.menu)):
        return SkillResult([], f"Index {index} out of range.", interrupted=True)
    return SkillResult([int(ord(obs.menu[index].letter))], f"Selected: {obs.menu[index].description}")


# ---------- bootstrap: capture role/race/alignment on episode start ----------

# Two forms of the welcome message we need to handle:
#   "Hello Agent, welcome to NetHack!  You are a neutral male human Monk."
#   "Hello Agent, the Stripling, welcome to NetHack!  You are a lawful female human Valkyrie."
# The role-title prefix (Stripling, Candidate, ...) varies by role + XP rank.
_WELCOME_RE = re.compile(
    r"You are (?:a |an )?"
    r"(?P<alignment>lawful|neutral|chaotic)\s+"
    r"(?P<gender>male|female|neuter)\s+"
    r"(?P<race>\w+)\s+"
    r"(?P<role>\w+)"
)
_UNKNOWN_CHARACTER = {
    "role": "unknown",
    "race": "unknown",
    "alignment": "unknown",
    "gender": "unknown",
}


def parse_character_from_welcome(message: str) -> dict[str, str]:
    """
    Parse role/race/alignment/gender from the NetHack opening welcome line.

    Returns the canonical {"role","race","alignment","gender"} dict with
    "unknown" sentinels if parsing fails (we never want to crash here — the
    rest of the system runs fine without character info, just without the
    role-aware prompt block).
    """
    m = _WELCOME_RE.search(message)
    if not m:
        return dict(_UNKNOWN_CHARACTER)
    out = m.groupdict()
    # Role from the welcome message is the role *singular* (e.g. "Monk", not
    # the title "Stripling"). Normalize to lowercase for consistency with the
    # rest of the obs schema.
    return {k: v.lower() for k, v in out.items()}


# Rank titles (xp level 1 only) per role, used as a fallback when NLE's
# welcome message has been preempted by a calendar event. Source: NetHack
# 3.6 role definitions. Only the level-1 title is unique enough to map back.
_LEVEL1_TITLE_TO_ROLE = {
    "Candidate": "monk",
    "Stripling": "valkyrie",
    "Hatamoto": "samurai",
    "Evoker": "wizard",
    "Aspirant": "priest",
    "Tenderfoot": "ranger",
    "Footpad": "rogue",
    "Plunderer": "barbarian",
    "Rambler": "tourist",
    "Digger": "archeologist",
    "Troglodyte": "caveman",
    "Rhizotomist": "healer",
    "Gallant": "knight",
}


def _bootstrap_from_status_line(tty_chars) -> dict[str, str]:
    """Fallback parser: reads "Agent the <Title>" + alignment word from the
    NetHack status line (row 22 of the tty). Used when the welcome message
    buffer is preempted (e.g. by a new-moon calendar event)."""
    out = dict(_UNKNOWN_CHARACTER)
    if tty_chars is None or len(tty_chars) < 23:
        return out
    line = "".join(chr(c) for c in tty_chars[22]).strip()
    # Title (xp1 rank) → role.
    for title, role in _LEVEL1_TITLE_TO_ROLE.items():
        if f" the {title}" in line:
            out["role"] = role
            break
    # Alignment word at end of first half of status line.
    for align in ("Lawful", "Neutral", "Chaotic"):
        if align in line:
            out["alignment"] = align.lower()
            break
    return out


def bootstrap_character(env: NetHackCoreEnv) -> dict[str, str]:
    """
    Read role/race/alignment/gender from the env's current observation.

    Primary path: NetHack's first message after `env.reset()` is usually
    "Hello <name>, welcome to NetHack!  You are a <align> <gender> <race>
    <role>." We parse that. Free — consumes no in-game turn.

    Fallback path: on dates when NLE prepends a calendar event ("Be careful!
    New moon tonight."), the welcome line gets overwritten in the message
    buffer. We then scrape the tty status line ("Agent the Candidate ...
    Neutral S:0") for the role's level-1 rank title + alignment word. This
    won't recover race or gender (those don't appear in the status line)
    but at least gives a real role/alignment instead of "unknown".

    Why not `#attributes`? The Score / Staircase / etc. tasks restrict the
    action set to 23 actions and don't include Command.ATTRIBUTES. We'd have
    to either extend the action set (raises portability questions for
    PufferLib consumers) or screen-scrape the resulting menu. The welcome +
    status-line fallback combination works on every NLE task class.
    """
    out = dict(_UNKNOWN_CHARACTER)
    try:
        last_msg = env.last_observation
        keys = env.observation_keys
        # Primary: welcome message
        if "message" in keys:
            msg_bytes = last_msg[keys.index("message")]
            msg = bytes(msg_bytes).split(b"\x00", 1)[0].decode("ascii", errors="replace")
            parsed = parse_character_from_welcome(msg)
            if parsed.get("role") != "unknown":
                return parsed
        # Fallback: status line via tty_chars
        if "tty_chars" in keys:
            tty = last_msg[keys.index("tty_chars")]
            fallback = _bootstrap_from_status_line(tty)
            for k in ("role", "alignment"):
                if fallback.get(k) != "unknown":
                    out[k] = fallback[k]
    except Exception:
        # Defensive: never break bootstrap.
        pass
    return out


# ---------- to-be-implemented ----------
# These are real skills, just stubs for now. Each is a self-contained PR worth.

def _current_chars_and_player(env: NetHackCoreEnv):
    """Pull the latest chars grid and player (x, y) from the underlying NLE.

    The skill API gets a StructuredObservation but we want the raw chars
    array for pathfinding. The CoreObservation lives in the verifiers state
    dict, not here; so we read it back from the env's last_observation.
    """
    keys = env.observation_keys
    last = env.last_observation
    chars = last[keys.index("chars")]
    blstats = last[keys.index("blstats")]
    return chars, (int(blstats[0]), int(blstats[1]))


def _enum_actions_to_indices(env: NetHackCoreEnv, enum_actions: list[int]) -> list[int]:
    """
    Pathfinding returns semantic action enum values (e.g. 107 for
    CompassDirection.N), which ARE the keystroke bytes the engine consumes. The
    engine takes them directly, so this is now an identity pass-through (kept as
    a named seam in case a future backend needs translation again).
    """
    return [int(a) for a in enum_actions]


def _cheb(ax: int, ay: int, bx: int, by: int) -> int:
    return max(abs(ax - bx), abs(ay - by))


def _move_to_best_effort(env, chars, start, tx, ty) -> "SkillResult":
    """Best-effort single step toward (tx, ty) when no FULL path exists.

    Strategy (all one-step-at-a-time; never auto-explores or descends):
      1. Compute the A*-reachable set over explored/walkable tiles. Pick the
         reachable tile that MINIMIZES Chebyshev distance to the target; if
         it differs from the player's tile, path ONE step toward it. This
         closes distance and reveals map, so a real path can open up next call.
      2. If we're already AT the closest reachable tile, probe ONE step in the
         target's general direction if that adjacent tile is walkable or
         unknown (unexplored). This nudges the agent toward the goal.
      3. Else, return an actionable message naming the nearest reachable tile
         and what to do — never a bare zero-progress dead-end.
    """
    try:
        h, w = chars.shape
    except Exception:
        h = len(chars); w = len(chars[0]) if h else 0

    # Out-of-bounds target: nothing geometric to chase toward; explain.
    if not (0 <= tx < w and 0 <= ty < h):
        return SkillResult(
            [],
            f"No route to ({tx},{ty}): target is out of bounds "
            f"(map is 0..{w-1} x 0..{h-1}).",
            interrupted=True,
        )

    reach = reachable_set(chars, start)
    # 1) Nearest reachable tile to the target (Chebyshev), excluding start so
    #    we genuinely move. Ties broken by closeness to the player so we take
    #    a sensible incremental step rather than darting across the level.
    best = None
    best_key = None
    sx, sy = start
    for (rx, ry) in reach:
        if (rx, ry) == start:
            continue
        key = (_cheb(rx, ry, tx, ty), _cheb(rx, ry, sx, sy))
        if best_key is None or key < best_key:
            best_key = key
            best = (rx, ry)

    start_dist = _cheb(sx, sy, tx, ty)
    if best is not None and best_key[0] < start_dist:
        # Some reachable tile is strictly closer to the target than we are.
        step = a_star(chars, start, best)
        if step:
            return SkillResult(
                actions=_enum_actions_to_indices(env, step[:1]),
                feedback=(
                    f"No full path to ({tx},{ty}) yet; stepping toward nearest "
                    f"reachable approach {best} to close distance and reveal map."
                ),
            )

    # 2) Already at (or no reachable tile beats) our position. Probe one step
    #    in the target's general direction if that neighbor is walkable or
    #    unexplored (so we push into the frontier rather than freeze).
    dirx = (tx > sx) - (tx < sx)
    diry = (ty > sy) - (ty < sy)
    if dirx or diry:
        nx, ny = sx + dirx, sy + diry
        if 0 <= nx < w and 0 <= ny < h:
            nch = int(chars[ny, nx])
            if is_walkable(nch) or nch == ord(" "):
                from nethack_core import actions as _nh
                _DIR8 = {
                    (0, -1): _nh.CompassDirection.N,
                    (1, -1): _nh.CompassDirection.NE,
                    (1, 0): _nh.CompassDirection.E,
                    (1, 1): _nh.CompassDirection.SE,
                    (0, 1): _nh.CompassDirection.S,
                    (-1, 1): _nh.CompassDirection.SW,
                    (-1, 0): _nh.CompassDirection.W,
                    (-1, -1): _nh.CompassDirection.NW,
                }
                act = _DIR8.get((dirx, diry))
                if act is not None:
                    return SkillResult(
                        actions=[int(act)],
                        feedback=(
                            f"No full path to ({tx},{ty}) yet; probing one step "
                            f"toward it to reveal unexplored terrain."
                        ),
                    )

    # 3) Genuinely stuck: nearest reachable tile and an actionable hint.
    rx, ry = best if best is not None else start
    return SkillResult(
        [],
        (
            f"No route to ({tx},{ty}) yet — nearest reachable is ({rx},{ry}); "
            f"the way is likely behind unexplored/blocked terrain — `search` "
            f"near walls or `move` to explore."
        ),
        interrupted=True,
    )


@registry.register("move_to", schema={
    "description": (
        "Pathfind to a specific (x, y) tile on the current level. Uses A* "
        "over the visible map. The path is precomputed — if a monster steps "
        "into the path mid-traversal, the rollout will end up engaging it. "
        "For careful exploration, call autoexplore instead."
    ),
    "parameters": {
        "x": {"type": "integer", "description": "Column 0..78"},
        "y": {"type": "integer", "description": "Row 0..20"},
    },
})
def move_to(env: NetHackCoreEnv, obs: StructuredObservation, x: int, y: int) -> SkillResult:
    chars, start = _current_chars_and_player(env)
    tx, ty = int(x), int(y)
    # SHORT-CIRCUIT: if the player ASKED for `>` (target tile is stairs DOWN)
    # OR `>` is currently visible and reachable, path to `>` and append
    # descend in one tool call. Same rescue logic as move().
    try:
        h2, w2 = chars.shape
        stair2 = None
        # honor explicit request first
        if 0 <= ty < h2 and 0 <= tx < w2 and int(chars[ty, tx]) == ord('>'):
            stair2 = (tx, ty)
        else:
            for yy in range(h2):
                for xx in range(w2):
                    if int(chars[yy, xx]) == ord('>'):
                        stair2 = (xx, yy); break
                if stair2: break
        if stair2 is not None and stair2 != start:
            p2 = a_star(chars, start, stair2)
            if p2:
                acts = _enum_actions_to_indices(env, p2)
                # Keystrokes consumed directly: MORE dismisses a prompt, DOWN descends.
                acts = acts + [int(nethack.MiscAction.MORE), int(nethack.MiscDirection.DOWN)]
                return SkillResult(
                    actions=acts,
                    feedback=f"move_to({tx},{ty}) — diverted to `>` at {stair2}, pathing {len(p2)} steps and descending.",
                )
    except Exception:
        pass
    path = a_star(chars, start, (tx, ty))
    if path is None:
        # No FULL path to the exact target (it's a wall, out of bounds, or —
        # most often — visible-but-not-yet-connected behind unexplored
        # terrain). A bare "No path" sent the model into spin loops: it can
        # SEE the stairs but can never approach them. Instead, make
        # best-effort progress so the agent closes distance and reveals map,
        # opening a real path on a subsequent call. Still ONE step per call;
        # we never auto-explore, auto-search, or auto-descend here.
        return _move_to_best_effort(env, chars, start, tx, ty)
    if not path:
        return SkillResult([], f"Already at ({tx},{ty}).", interrupted=False)
    return SkillResult(
        actions=_enum_actions_to_indices(env, path),
        feedback=f"Pathing to ({x},{y}): {len(path)} steps.",
    )


def _stairs_up_xy(chars):
    """Return (x, y) of the first visible `<` tile, or None."""
    try:
        h, w = chars.shape
        for y in range(h):
            for x in range(w):
                if int(chars[y, x]) == ord('<'):
                    return (x, y)
    except Exception:
        pass
    return None


@registry.register("autoexplore", schema={
    "description": (
        "Walk toward the nearest unexplored region of the current level. "
        "Picks the closest frontier (walkable tile adjacent to an unseen "
        "tile) and pathfinds to it. Call repeatedly to explore the whole "
        "level. Halts and returns when the level is fully revealed."
    ),
    "parameters": {
        "max_steps": {
            "type": "integer",
            "default": 30,
            "description": "Cap on path length per call; lower means more chances to react.",
        },
    },
})
def autoexplore(env: NetHackCoreEnv, obs: StructuredObservation, max_steps: int = 30) -> SkillResult:
    chars, start = _current_chars_and_player(env)
    # Short-circuit: if stairs DOWN are already visible and reachable, walk
    # to them instead of any frontier. Trace 9071d001 showed the agent
    # calling autoexplore long after `>` was visible — autoexplore picked
    # the closest *frontier*, not the goal. Now the goal wins.
    try:
        h, w = (chars.shape[0], chars.shape[1]) if hasattr(chars, "shape") else (len(chars), len(chars[0]))
        stair_coords = None
        for sy in range(h):
            for sx in range(w):
                if int(chars[sy, sx]) == ord('>'):
                    stair_coords = (sx, sy)
                    break
            if stair_coords:
                break
        if stair_coords is not None and stair_coords != start:
            path = a_star(chars, start, stair_coords)
            if path:
                trimmed = path[:max_steps]
                acts = _enum_actions_to_indices(env, trimmed)
                # Auto-descend: append MORE + DOWN keystrokes so a single
                # autoexplore call when `>` is visible reaches the stairs
                # AND descends without needing another LM round-trip.
                # Without this, the agent saw the stairs hint but then
                # picked move_to / move single-step instead of `descend`.
                acts = acts + [int(nethack.MiscAction.MORE), int(nethack.MiscDirection.DOWN)]
                return SkillResult(
                    actions=acts,
                    feedback=(
                        f"Stairs DOWN at {stair_coords} are visible — "
                        f"pathing {len(trimmed)} steps and descending."
                    ),
                )
    except Exception:
        pass
    # Wave-2 Track B: optional per-level frontier blacklist. The harness
    # ( nethack.py::_update_frontier_blacklist ) stashes the current-level
    # set onto the env via `frontier_blacklist_current` for
    # skills to pick up; if absent we fall back to legacy behavior.
    blacklist = None
    try:
        cur_bl = getattr(env, "frontier_blacklist_current", None)
        if isinstance(cur_bl, (set, frozenset)) and cur_bl:
            blacklist = set(cur_bl)
    except Exception:
        blacklist = None
    result = nearest_frontier(chars, start, blacklist=blacklist)
    # Skip-stairs-UP guard: trace 5/16 showed the frontier picker happily
    # returning the `<` tile (it's walkable + adjacent to closed door which
    # counts as unknown when door state is partially loaded). Walking to
    # `<` is pointless when our objective is to descend — re-pick from
    # find_frontiers and exclude the `<` tile. If no alternative exists,
    # fall through to the no-frontier branch (search/kick advice).
    if result is not None:
        target, _path = result
        up_xy = _stairs_up_xy(chars)
        if up_xy is not None and target == up_xy:
            from nethack_harness.navigation.pathfinding import find_frontiers, a_star
            alts = [
                f for f in find_frontiers(chars)
                if int(chars[f[1], f[0]]) != ord('<') and f != start
            ]
            best = None
            best_path = None
            for cand in alts:
                p = a_star(chars, start, cand)
                if p and (best_path is None or len(p) < len(best_path)):
                    best = cand
                    best_path = p
            if best is not None:
                result = (best, best_path)
            # else: keep original `<` target — better than no movement
    if result is None:
        # No frontier means we've explored everything reachable. If stairs
        # down aren't visible, the level likely has hidden passages — auto-
        # walk to a dead-end corridor tile and queue a search burst there.
        # This is the human heuristic: when stuck, search at corridor ends.
        has_stairs_down = any(b'>' in row.tobytes() for row in chars) if hasattr(chars, "__iter__") else False
        if has_stairs_down:
            tip = " Stairs `>` are visible — `move_to` them and `descend`."
            return SkillResult([], "Level appears fully explored from this position." + tip, interrupted=True)
        # Dead-end search: find walkable tiles (preferably corridor `#`) with
        # only one walkable cardinal neighbor. Path to the closest one and
        # queue 20 `search` actions there. NetHack `s` is action enum index
        # for search; appending many of them lets us search in one tool call.
        from nethack_harness.navigation.pathfinding import a_star
        try:
            h, w = chars.shape
            dead_ends = []
            for yy in range(h):
                for xx in range(w):
                    ch = int(chars[yy, xx])
                    if ch not in (ord('#'), ord('.')):
                        continue
                    n = 0
                    for ddx, ddy in ((0,-1),(1,0),(0,1),(-1,0)):
                        nx, ny = xx+ddx, yy+ddy
                        if 0 <= nx < w and 0 <= ny < h:
                            nc = int(chars[ny, nx])
                            if nc in (ord('.'), ord('#'), ord('+'),
                                     ord('<'), ord('>'), ord("'")):
                                n += 1
                    if n == 1 and (xx, yy) != start:
                        dead_ends.append((xx, yy))
            # Score: prefer corridor tiles over room floors (room dead-ends
            # are usually corners with nothing behind them).
            best = None
            best_path = None
            best_score = 1 << 30
            for de in dead_ends:
                p = a_star(chars, start, de)
                if not p:
                    continue
                is_corr = int(chars[de[1], de[0]]) == ord('#')
                score = len(p) + (0 if is_corr else 100)
                if score < best_score:
                    best_score = score
                    best = de
                    best_path = p
            if best is not None and best_path:
                # SEARCH is the keystroke 's', consumed directly by the engine.
                search_action_idx = int(nethack.Command.SEARCH)
                path_idx = _enum_actions_to_indices(env, best_path[:max_steps])
                return SkillResult(
                    actions=path_idx + [search_action_idx] * 20,
                    feedback=(
                        f"No frontiers reachable; walking to dead-end "
                        f"{best} ({len(best_path)} steps) and searching 20× "
                        "for hidden passages."
                    ),
                )
        except Exception:
            pass
        # If we couldn't even find a dead-end, fall back to a strong tip.
        has_door = False
        try:
            has_door = any(b'+' in row.tobytes() for row in chars)
        except Exception:
            pass
        if has_door:
            tip = (
                " No `>` visible but a closed door `+` exists on the map. "
                "`move_to` adjacent to it; if it won't open, "
                "`kick(direction=...)` 2-5 times to break the lock."
            )
        else:
            tip = (
                " No `>` or dead-ends found. Call `search(times=20)` here "
                "and walk to wall corners to keep searching."
            )
        return SkillResult([], "Level appears fully explored." + tip, interrupted=True)
    target, path = result
    path = path[:max_steps]
    if not path:
        return SkillResult(
            [],
            f"Standing on a frontier already; try a different direction.",
            interrupted=True,
        )
    # Tail-hint: very short paths suggest the level is mostly explored —
    # the model often loops on autoexplore when frontiers are tiny. Surface
    # a tip so the next obs prompts a different action.
    suffix = ""
    if len(path) <= 2:
        has_stairs_down = any(b'>' in row.tobytes() for row in chars) if hasattr(chars, "__iter__") else False
        if has_stairs_down:
            suffix = " (short — `>` visible; consider `move_to` and `descend`)"
        else:
            # Short-frontier rerouting: if there's a known door/passage we
            # haven't been through, prefer pathing to it over the tiny
            # frontier. Picks the closest such door by A* path length.
            from nethack_harness.navigation.pathfinding import a_star as _astar2
            from nethack_core import extract_visible_features
            try:
                keys = env.observation_keys
                last_obs_buf = env.last_observation
                tty = last_obs_buf[keys.index("tty_chars")] if "tty_chars" in keys else None
            except Exception:
                tty = None
            if tty is not None:
                feats = extract_visible_features(tty)
                doors = []
                import re as _r
                for f in feats:
                    if f.startswith("door (open/gap)") or f.startswith("door (closed)"):
                        for mm in _r.finditer(r"\((\d+),(\d+)\)", f):
                            doors.append((int(mm.group(1)), int(mm.group(2))))
                best_door = None
                best_door_path = None
                for dxy in doors:
                    if dxy == start:
                        continue
                    p2 = _astar2(chars, start, dxy)
                    if p2 and len(p2) > 1 and (best_door_path is None or len(p2) < len(best_door_path)):
                        best_door = dxy
                        best_door_path = p2
                if best_door is not None and best_door_path is not None:
                    trimmed2 = best_door_path[:max_steps]
                    return SkillResult(
                        actions=_enum_actions_to_indices(env, trimmed2),
                        feedback=(
                            f"Autoexplore: frontier exhausted nearby; pathing "
                            f"to door at {best_door} ({len(trimmed2)} steps). "
                            f"If door is locked there, kick to break it."
                        ),
                    )
            suffix = " (short — try `search` at a wall or `move_to` a known feature)"
    return SkillResult(
        actions=_enum_actions_to_indices(env, path),
        feedback=f"Autoexploring toward {target}: {len(path)} steps." + suffix,
    )


@registry.register("find_and_descend", schema={
    "description": (
        "MEGA-SKILL for the corridor_explore / mini_dungeon objective. "
        "Bundles many decisions into one tool call:\n"
        "1) If `>` is currently visible, path to it and append `>` (descend).\n"
        "2) Else if a not-yet-walked door is reachable, path to it (kick "
        "automatically if message says locked next turn).\n"
        "3) Else walk to the nearest corridor dead-end and search 25 times.\n"
        "Returns up to ~80 NLE actions in one call. Re-call until reward "
        "for descent fires. Cheaper than micromanaging move()/move_to()."
    ),
    "parameters": {
        "max_actions": {
            "type": "integer",
            "default": 80,
            "description": "Cap on actions queued per call.",
        },
    },
})
def find_and_descend(env: NetHackCoreEnv, obs: StructuredObservation, max_actions: int = 80) -> SkillResult:
    chars, start = _current_chars_and_player(env)
    from nethack_harness.navigation.pathfinding import a_star, find_frontiers
    h, w = chars.shape
    # 1) Stairs DOWN visible? Path + descend.
    stair = None
    for yy in range(h):
        for xx in range(w):
            if int(chars[yy, xx]) == ord('>'):
                stair = (xx, yy); break
        if stair: break
    # Descend keystrokes consumed directly by the engine.
    _more_i = int(nethack.MiscAction.MORE)
    _down_i = int(nethack.MiscDirection.DOWN)
    if stair is not None:
        if stair == start:
            return SkillResult([_more_i, _down_i], "Already on `>` — descending.")
        p = a_star(chars, start, stair)
        if p:
            actions = _enum_actions_to_indices(env, p[:max_actions - 2])
            actions += [_more_i, _down_i]
            return SkillResult(
                actions=actions,
                feedback=f"`>` visible at {stair}; pathing {len(p)} steps and descending.",
            )
    # 2) Pick a reachable door (open/gap or closed) and path there.
    from nethack_core import extract_visible_features
    try:
        keys = env.observation_keys
        tty = env.last_observation[keys.index("tty_chars")] if "tty_chars" in keys else None
    except Exception:
        tty = None
    door_xy = None; door_path = None
    if tty is not None:
        import re as _r
        feats = extract_visible_features(tty)
        candidates = []
        for f in feats:
            if f.startswith("door (open/gap)") or f.startswith("door (closed)"):
                for mm in _r.finditer(r"\((\d+),(\d+)\)", f):
                    candidates.append((int(mm.group(1)), int(mm.group(2))))
        best_len = 1 << 30
        for dxy in candidates:
            if dxy == start: continue
            p2 = a_star(chars, start, dxy)
            if p2 and len(p2) < best_len:
                best_len = len(p2); door_xy = dxy; door_path = p2
    if door_xy is not None and door_path:
        return SkillResult(
            actions=_enum_actions_to_indices(env, door_path[:max_actions]),
            feedback=f"No `>` visible; pathing to door at {door_xy} ({len(door_path)} steps).",
        )
    # 3) Frontier — but prefer the FARTHEST reachable frontier so we
    # actually make progress per skill call. nearest_frontier returns the
    # immediate next walkable+unknown-adjacent tile, which is usually 1
    # step away on a corridor → the LM has to round-trip on every step.
    # Scan all frontiers and pick the longest-A* one (excluding <).
    fs = find_frontiers(chars)
    best_fr = None; best_fr_path = None; best_len = -1
    for fr in fs:
        if int(chars[fr[1], fr[0]]) == ord('<'):
            continue
        if fr == start:
            continue
        p = a_star(chars, start, fr)
        if p and len(p) > best_len:
            best_len = len(p); best_fr = fr; best_fr_path = p
    if best_fr is not None and best_fr_path:
        return SkillResult(
            actions=_enum_actions_to_indices(env, best_fr_path[:max_actions]),
            feedback=f"Pathing to far frontier at {best_fr} ({len(best_fr_path)} steps).",
        )
    # Dead-end fallback.
    dead_ends = []
    for yy in range(h):
        for xx in range(w):
            ch = int(chars[yy, xx])
            if ch not in (ord('#'), ord('.')): continue
            n = 0
            for ddx, ddy in ((0,-1),(1,0),(0,1),(-1,0)):
                nx, ny = xx+ddx, yy+ddy
                if 0 <= nx < w and 0 <= ny < h:
                    nc = int(chars[ny, nx])
                    if nc in (ord('.'), ord('#'), ord('+'), ord('<'), ord('>'), ord("'")):
                        n += 1
            if n == 1 and (xx, yy) != start:
                dead_ends.append((xx, yy))
    best_de = None; best_p = None; best_score = 1 << 30
    for de in dead_ends:
        p = a_star(chars, start, de)
        if not p: continue
        is_corr = int(chars[de[1], de[0]]) == ord('#')
        sc = len(p) + (0 if is_corr else 100)
        if sc < best_score:
            best_score = sc; best_de = de; best_p = p
    # SEARCH ('s') keystroke, consumed directly by the engine.
    s_idx = int(nethack.Command.SEARCH)
    if best_de is not None and best_p is not None:
        a = _enum_actions_to_indices(env, best_p[:max_actions - 25])
        a += [s_idx] * 25
        return SkillResult(
            actions=a,
            feedback=f"Walking to dead-end {best_de} ({len(best_p)} steps) and searching 25× for hidden passages.",
        )
    # No useful action; just search in place.
    return SkillResult(
        actions=[s_idx] * 25,
        feedback="No frontier/door/dead-end found; searching here 25× (consider `move` to a new corridor).",
    )


_EAD_CMAP_LUT = None
_EAD_CLOSED_CMAPS = None


def _ead_cmap_lut():
    """cmap index -> clean char LUT, from the canonical symdef table (NetPlay's
    glyph-group idea). open door -> '.', closed door -> '|' (BLOCKED for pathing),
    walls -> '|', downstairs -> '>', upstairs -> '<', dark/unexplored -> ' '.
    Also caches the closed-door cmap set. Cached."""
    global _EAD_CMAP_LUT, _EAD_CLOSED_CMAPS
    if _EAD_CMAP_LUT is None:
        from nethack_core import glyphs as N
        # Baked from the fork's cmap symbol table; closed-door cmaps are BLOCKED
        # for pathing (a_star must not route through a closed door and bump).
        _EAD_CMAP_LUT = N.cmap_clean_char_lut().copy()
        _EAD_CLOSED_CMAPS = set(N.CMAP_CLOSED_DOOR_INDICES)
    return _EAD_CMAP_LUT


def _closed_door_positions(glyphs):
    """List of (x, y) closed-door tiles, from glyphs (vectorized)."""
    import numpy as np
    from nethack_core import glyphs as N
    _ead_cmap_lut()  # ensure _EAD_CLOSED_CMAPS is built
    if not _EAD_CLOSED_CMAPS:
        return []
    g = np.asarray(glyphs, dtype=np.int64)
    iscmap = np.asarray(N.glyph_is_cmap(g), dtype=bool)
    cm = g - int(N.GLYPH_CMAP_OFF)
    mask = iscmap & np.isin(cm, list(_EAD_CLOSED_CMAPS))
    ys, xs = np.nonzero(mask)
    return [(int(x), int(y)) for y, x in zip(ys, xs)]


def _glyph_clean_chars(glyphs):
    """Unambiguous tty-like grid built from GLYPHS (vectorized): open doors -> '.',
    closed -> '+', walls -> '|', downstairs -> '>', up -> '<', unexplored -> ' '.
    The tty char layer renders an open door identically to a wall; glyphs
    disambiguate (the way NetPlay tracks the map)."""
    import numpy as np
    from nethack_core import glyphs as N
    lut = _ead_cmap_lut()
    g = np.asarray(glyphs, dtype=np.int64)
    out = np.full(g.shape, ord(' '), np.uint8)
    iscmap = np.asarray(N.glyph_is_cmap(g), dtype=bool)
    cm = g - int(N.GLYPH_CMAP_OFF)
    valid = iscmap & (cm >= 0) & (cm < lut.shape[0])
    out[valid] = lut[cm[valid]]
    ent = (np.asarray(N.glyph_is_monster(g), bool)
           | np.asarray(N.glyph_is_pet(g), bool)
           | np.asarray(N.glyph_is_object(g), bool))
    out[ent] = ord('.')  # walkable terrain under a monster / item / pet
    return out


@registry.register("explore_and_descend", schema={
    "description": (
        "CLOSED-LOOP mega-skill: automatically explore the ENTIRE current level "
        "(revealing rooms/corridors, and SEARCHING dead-ends and door-adjacent "
        "walls for hidden passages), and the instant the down-staircase `>` is "
        "found, path to it and descend. It steps the game many times in one call, "
        "re-observing after each move, until it descends a floor (or the level is "
        "fully explored and searched). This is the single best way to make "
        "progress deeper into the dungeon — prefer it over autoexplore / "
        "find_and_descend, which only take one step toward exploration per call."
    ),
    "parameters": {
        "max_floors": {"type": "integer", "default": 1,
                       "description": "descend at most this many floors before returning"},
        "max_game_steps": {"type": "integer", "default": 400,
                           "description": "hard step budget for this call"},
    },
})
def explore_and_descend(env: NetHackCoreEnv, obs: StructuredObservation,
                        max_floors: int = 1, max_game_steps: int = 400) -> SkillResult:
    """Explore the current level -> find `>` -> descend ONE floor, then RETURN
    control to the agent (the LLM decides what to do next: eat, fight, pray,
    descend again). Also returns early on danger (HP drop) or when out of hunger,
    so the agent gets a decision break instead of an autopilot run to death.
    Internally it re-observes every step (NetPlay's explore_level loop) and
    searches dead-ends / room perimeters for hidden passages."""
    import numpy as np
    from nethack_core import actions as _nh
    # Glyph predicates (glyph_is_monster/pet) -- pure-Python, nle-free.
    from nethack_core import glyphs as _glyph
    from nethack_harness.navigation.pathfinding import a_star, find_frontiers

    # The engine consumes keystroke bytes directly, so each action IS its enum
    # value -- no action-index translation layer.
    MORE = int(_nh.MiscAction.MORE)
    DOWN = int(_nh.MiscDirection.DOWN)
    SEARCH = int(_nh.Command.SEARCH)
    KICK = int(_nh.Command.KICK)
    DIRS = {(0, -1): int(_nh.CompassDirection.N),
            (0, 1): int(_nh.CompassDirection.S),
            (1, 0): int(_nh.CompassDirection.E),
            (-1, 0): int(_nh.CompassDirection.W)}
    # All 8 compass directions for melee — a monster can be diagonally adjacent.
    # Orthogonal first so straight melee is preferred over a diagonal swing.
    DIRS8 = {(0, -1): int(_nh.CompassDirection.N),
             (0, 1): int(_nh.CompassDirection.S),
             (1, 0): int(_nh.CompassDirection.E),
             (-1, 0): int(_nh.CompassDirection.W),
             (1, -1): int(_nh.CompassDirection.NE),
             (1, 1): int(_nh.CompassDirection.SE),
             (-1, 1): int(_nh.CompassDirection.SW),
             (-1, -1): int(_nh.CompassDirection.NW)}
    kicked: dict = {}  # (level_idx, x, y) -> kick attempts (avoid infinite kicking)

    # persistent per-(level,tile) search counts so repeated calls don't re-search
    search_count = getattr(env, "_explore_search_count", None)
    if search_count is None:
        search_count = {}
        try: setattr(env, "_explore_search_count", search_count)
        except Exception: pass

    def floor_id():
        bl = env.last_observation[_ks.index("blstats")]
        return (int(bl[23]), int(bl[24]))  # (DNUM, DLEVEL) — unique per floor

    state = {"r": 0.0, "term": False, "trunc": False, "steps": 0, "obs": None}
    _ttci = env.observation_keys.index("tty_chars")

    def _more_up() -> bool:
        try:
            tty = env.last_observation[_ttci]
            return any(b"--More--" in bytes(int(c) for c in row) for row in tty[:3])
        except Exception:
            return False

    def do(idx) -> bool:
        o, r, t, tr, _ = env.step(idx)
        state["obs"] = o; state["r"] += float(r)
        state["term"] = state["term"] or bool(t); state["trunc"] = state["trunc"] or bool(tr)
        state["steps"] += 1
        # A --More-- prompt eats the NEXT keypress, which would silently swallow
        # the agent's moves (it freezes in place). Acknowledge it here.
        guard = 0
        while not (state["term"] or state["trunc"]) and guard < 5 and _more_up():
            o, r2, t2, tr2, _ = env.step(MORE)
            state["obs"] = o; state["r"] += float(r2); state["steps"] += 1
            state["term"] = state["term"] or bool(t2); state["trunc"] = state["trunc"] or bool(tr2)
            guard += 1
        return bool(state["term"] or state["trunc"]) or state["steps"] >= max_game_steps

    def walk(path) -> bool:
        """Take ONE step along the path, then return so the caller re-observes and
        re-pathfinds. Blindly walking a whole precomputed path bumps and freezes
        the moment anything shifts (a monster, a doorway diagonal NetHack forbids);
        NetPlay's move_to is likewise one-step-at-a-time."""
        idxs = _enum_actions_to_indices(env, path[:1])
        return do(idxs[0]) if idxs else False

    def find_char(chars, target):
        h, w = chars.shape
        for yy in range(h):
            for xx in range(w):
                if int(chars[yy, xx]) == target:
                    return (xx, yy)
        return None

    def search_target(chars, start, floor):
        """Best walkable tile to stand on and search for a hidden passage — NetPlay's
        compute_search_mask: a tile whose adjacent wall borders unexplored stone,
        scored by door-walled-by-stone + dead-end shape, minus search_count². Returns
        ((x,y), path) for the least-searched / most-promising / nearest tile, or None
        once every candidate has been searched to its per-tile cap (level fully searched)."""
        h, w = chars.shape
        best = None; best_key = None
        for yy in range(h):
            for xx in range(w):
                if int(chars[yy, xx]) not in (ord('.'), ord('>'), ord('<')):
                    continue  # must stand on a walkable tile
                prio = 0
                nopen = 0
                for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                    nx, ny = xx + dx, yy + dy
                    if not (0 <= nx < w and 0 <= ny < h):
                        continue
                    nc = int(chars[ny, nx])
                    if nc in (ord('.'), ord('>'), ord('<')):
                        nopen += 1
                    elif nc == ord(' '):
                        prio += 3            # adjacent unexplored — search reveals it
                    elif nc == ord('|'):     # a wall — unexplored stone just beyond it?
                        bx, by = xx + 2 * dx, yy + 2 * dy
                        if 0 <= bx < w and 0 <= by < h and int(chars[by, bx]) == ord(' '):
                            prio += 1
                if nopen <= 1:
                    prio += 2                # dead-end: prime hidden-passage spot
                if prio == 0:
                    continue
                sc = search_count.get((floor, xx, yy), 0)
                if sc >= 12:                 # this spot is exhausted
                    continue
                p = a_star(chars, start, (xx, yy)) if (xx, yy) != start else []
                if (xx, yy) != start and not p:
                    continue
                # NetPlay: most-promising (high prio) and least-searched first, then nearest
                key = (sc * sc - prio * 100, len(p) if p else 0)
                if best_key is None or key < best_key:
                    best_key = key; best = ((xx, yy), p)
        return best

    def door_kick_target(chars, start, level_idx, doors):
        """Nearest reachable closed door (glyph-detected positions): the walkable
        orthogonal tile to stand on, path there, door pos, and move direction."""
        h, w = chars.shape
        STAND = (ord('.'), ord('>'), ord('<'))  # walkable approach tiles
        best = None; best_len = 1 << 30
        for (xx, yy) in doors:
            if kicked.get((level_idx, xx, yy), 0) >= 6:
                continue
            for dx, dy in DIRS:
                sx, sy = xx + dx, yy + dy  # tile to stand on (orthogonal)
                if not (0 <= sx < w and 0 <= sy < h):
                    continue
                if int(chars[sy, sx]) not in STAND:
                    continue
                p = a_star(chars, start, (sx, sy)) if (sx, sy) != start else []
                if (sx, sy) != start and not p:
                    continue
                plen = len(p) if p else 0
                if plen < best_len:
                    best_len = plen
                    best = ((sx, sy), p, (xx, yy), DIRS.get((xx - sx, yy - sy)))
        return best

    def cur_doors():
        ks = env.observation_keys
        glyphs = env.last_observation[ks.index("glyphs")]
        return _closed_door_positions(glyphs)

    floors = 0
    level_idx = 0
    down_stair = None           # remembered `>` position (hidden under `@` on arrival)
    _ks = env.observation_keys
    _bl0 = env.last_observation[_ks.index("blstats")]
    start_hp = int(_bl0[10])    # HP at call start — bail to the LLM if it drops hard
    halt = None                 # reason we returned control to the agent
    exhausted = False           # True only if every reachable tile was searched (level done)
    attacks = 0                 # consecutive in-skill melee swings (bail if a monster won't die)
    kites = 0                   # consecutive kite-retreats (let the pet finish; melee if it can't)
    pet_waits = 0               # turns spent waiting on the stairs for the pet to catch up
    PET_WAIT_BUDGET = 8         # don't wait forever — descend without the pet after this many
    KITE_BUDGET = 4             # if the pet hasn't killed it after this many retreats, melee
    import os as _os
    pet_tactics = _os.environ.get("NETHACK_DISABLE_PET", "") not in ("1", "true", "True")
    # ^ ablation switch: NETHACK_DISABLE_PET=1 turns OFF pet-aware descent + kiting
    #   (keeps everything else — survival prompt, search, in-skill melee).
    visited: set = set()  # frontiers already attempted (per level) — avoids oscillating
    opened: set = set()   # (level_idx, x, y) doors we've opened — they render as a wall
                          # char (`-`/`|`) in the tty, so patch them back to walkable.

    def obs_map():
        """Glyph-derived unambiguous map + player position. Using glyphs (not tty
        chars) fixes the open-door-vs-wall ambiguity at the source."""
        ks = env.observation_keys
        glyphs = env.last_observation[ks.index("glyphs")]
        bl = env.last_observation[ks.index("blstats")]
        chars = _glyph_clean_chars(glyphs)
        px, py = int(bl[0]), int(bl[1])
        chars[py, px] = ord('.')  # our own tile is always walkable to path from
        return chars, (px, py)

    def adjacent_hostile():
        """Direction (dx,dy) to an adjacent NON-pet monster, or None. Glyph-derived
        (the cleaned `chars` erase monsters to '.'), so we read raw glyphs. NetPlay
        fights weak monsters mid-explore rather than autopiloting past them into a
        slow death; we only call this when HP is healthy (the half-HP halt fires
        first) and we're not already beelining to reachable stairs."""
        ks = env.observation_keys
        glyphs = env.last_observation[ks.index("glyphs")]
        bl = env.last_observation[ks.index("blstats")]
        px, py = int(bl[0]), int(bl[1])
        h, w = glyphs.shape
        for (dx, dy) in DIRS8:  # orthogonal-first ordering
            nx, ny = px + dx, py + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            g = int(glyphs[ny, nx])
            if _glyph.glyph_is_monster(g) and not _glyph.glyph_is_pet(g):
                return (dx, dy)
        return None

    def nearest_pet(sx, sy):
        """(x,y) of the closest pet glyph, or None. The starting pet is a strong
        early ally — we want it to kill monsters for us and to follow us downstairs."""
        ks = env.observation_keys
        glyphs = env.last_observation[ks.index("glyphs")]
        mask = np.asarray(_glyph.glyph_is_pet(glyphs), bool)
        if not mask.any():
            return None
        ys, xs = np.where(mask)
        best = None; bestd = 1 << 30
        for yy, xx in zip(ys, xs):
            d = max(abs(int(xx) - sx), abs(int(yy) - sy))  # chebyshev (8-dir steps)
            if d < bestd:
                bestd = d; best = (int(xx), int(yy))
        return best

    def kite_step(chars, start, monster):
        """A walkable 8-dir step that INCREASES distance from `monster` (retreat).
        Used to kite — back off and let the pet trade blows — instead of melee."""
        sx, sy = start; mx, my = monster
        curd = max(abs(sx - mx), abs(sy - my))
        h, w = chars.shape
        best = None; bestd = curd
        for (dx, dy) in DIRS8:
            nx, ny = sx + dx, sy + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if int(chars[ny, nx]) not in (ord('.'), ord('>'), ord('<')):
                continue  # only retreat onto known-walkable floor/stairs
            nd = max(abs(nx - mx), abs(ny - my))
            if nd > bestd:
                bestd = nd; best = (dx, dy)
        return best  # None if no tile increases distance (cornered)

    while state["steps"] < max_game_steps and floors < max_floors and not state["term"] and not state["trunc"]:
        chars, start = obs_map()
        # Hand control back to the agent on danger so the LLM can react (heal,
        # flee, eat, pray) instead of this skill autopiloting into death.
        _bl = env.last_observation[_ks.index("blstats")]
        hp, hpmax, hunger = int(_bl[10]), int(_bl[11]), int(_bl[21])
        # Return EARLY (at half HP) so the LLM can heal/flee/pray before it's
        # critical — dying mid-call is the main thing capping our depth.
        if hp <= max(2, hpmax // 2):
            halt = f"HP at {hp}/{hpmax} — returning to you: rest/pray/elbereth/flee before going on"
            break
        if hunger >= 3:  # Weak (eat before Fainting)
            halt = "getting weak from hunger — returning to you to eat"
            break
        visited.add((level_idx, start))  # every tile we stand on — sweep forward,
                                         # never re-target a frontier we've passed
        seen_stair = find_char(chars, ord('>'))
        if seen_stair is not None:
            down_stair = seen_stair  # remember it: when we STAND on it the `@`
                                     # glyph hides the `>`, so find_char goes blind.
        if down_stair is not None:
            if start == down_stair:
                # Bring the pet down with us: in NetHack a pet ADJACENT when you
                # descend follows you to the next floor. Wait on the stairs for it
                # to catch up (a strong ally that kills monsters), but not forever.
                pet = nearest_pet(*start) if pet_tactics else None
                if pet is not None:
                    pdist = max(abs(pet[0] - start[0]), abs(pet[1] - start[1]))
                    if pdist > 1 and pet_waits < PET_WAIT_BUDGET:
                        pet_waits += 1
                        do(SEARCH)  # wait in place; the pet closes the gap
                        continue
                pet_waits = 0
                # standing on the downstair — dismiss any prompt, then descend.
                do(MORE)
                do(DOWN)
                floors += 1
                level_idx += 1
                down_stair = None
                visited.clear()
                if floors >= max_floors:
                    break
                continue
            p = a_star(chars, start, down_stair)
            if p:
                if walk(p):
                    break
                continue  # one step toward the stair, then re-observe
            down_stair = None  # unreachable for now — keep exploring/searching
        # 0.5) Fight an adjacent hostile that's blocking exploration. We reach here
        #    only when there's no reachable downstair (escape-to-stairs is handled
        #    above) and HP is healthy (the half-HP halt already hands off real
        #    danger). This clears the weak monsters (rats/newts/kobolds) that were
        #    whittling HP to the halt threshold mid-explore — NetPlay melees weak
        #    monsters in-skill rather than autopiloting past them into a slow death.
        adir = adjacent_hostile()
        if adir is not None:
            mx, my = start[0] + adir[0], start[1] + adir[1]   # the monster's tile
            # Let the pet kill it: if a pet is next to the monster it will trade
            # blows for free — kite back instead of taking melee damage ourselves.
            pet = nearest_pet(*start) if pet_tactics else None
            pet_on_it = pet is not None and max(abs(pet[0] - mx), abs(pet[1] - my)) <= 1
            if pet_on_it and kites < KITE_BUDGET:
                kstep = kite_step(chars, start, (mx, my))
                if kstep is not None:
                    kites += 1
                    if do(DIRS8[kstep]):  # retreat one tile; pet finishes the monster
                        break
                    continue
                # cornered (nowhere to retreat) — fall through and melee
            kites = 0
            attacks += 1
            if attacks > 16:  # stubborn / out-of-depth monster — let the LLM decide
                halt = "a monster won't go down — returning to you to fight/flee/pray/elbereth"
                break
            if do(DIRS8[adir]):   # melee = step into it
                break
            continue
        attacks = 0  # nothing adjacent to fight — reset the swing counter
        kites = 0
        # 1) Explore the nearest OPEN frontier (exclude upstairs and closed doors;
        #    doors need an orthogonal approach and are handled in step 2).
        frontiers = [f for f in find_frontiers(chars)
                     if int(chars[f[1], f[0]]) not in (ord('<'), ord('+'))
                     and f != start and (level_idx, f) not in visited]
        best = None; best_p = None; best_len = 1 << 30
        for f in frontiers:
            p = a_star(chars, start, f)
            if p and len(p) < best_len:
                best_len = len(p); best = f; best_p = p
        if best is not None:
            if walk(best_p):
                break
            if obs_map()[1] != start:
                continue  # progressed one step — re-observe + re-pathfind
            visited.add((level_idx, best))   # bumped/blocked — blacklist this frontier
            # fall through to door / search handling
        # 2) Open the nearest reachable closed door: approach an orthogonally
        #    adjacent tile, then MOVE into the door (NetHack autoopen). If it stays
        #    shut after a few tries it is locked -> KICK it open.
        dt = door_kick_target(chars, start, level_idx, cur_doors())
        if dt is not None:
            (sx, sy), p, doorpos, _ = dt
            if p and walk(p):
                break
            _, now = obs_map()
            mdir = DIRS.get((doorpos[0] - now[0], doorpos[1] - now[1]))
            if mdir is not None:
                tries = kicked.get((level_idx,) + doorpos, 0)
                if tries < 4:
                    do(mdir)                 # autoopen: walk into the door
                elif KICK is not None:
                    do(KICK); do(mdir)       # locked -> kick it open
                kicked[(level_idx,) + doorpos] = tries + 1
                opened.add((level_idx,) + doorpos)   # treat as walkable next pass
                continue
            if now != start:
                continue
        # 3) Nothing to explore/open: search dead-ends / door-stone for hidden passages.
        tgt = search_target(chars, start, floor_id())
        if tgt is None:
            exhausted = True
            break  # level fully explored + searched, no `>` reachable
        (tx, ty), p = tgt
        if (tx, ty) != start:
            if walk(p):
                break
            if obs_map()[1] != start:
                continue  # stepping toward the search tile
            # could not move toward it — search from here as a fallback
        for _ in range(5):
            if do(SEARCH):
                break
        search_count[(floor_id(), tx, ty)] = search_count.get((floor_id(), tx, ty), 0) + 5

    if state["steps"] == 0:
        return SkillResult(actions=[], feedback="Nothing to explore from here; already descended or blocked.")
    fb = (f"explore_and_descend: descended {floors} floor(s) over {state['steps']} game steps"
          + (f" — {halt}" if halt else "")
          + (" — stopped (died/level-end)" if state["term"] or state["trunc"] else ""))
    if floors or halt or state["term"] or state["trunc"]:
        fb += "."
    elif exhausted:
        # Genuinely searched every reachable tile and found no `>` — the staircase is
        # behind a still-hidden passage or needs going `<` up and around.
        fb += ("; searched every reachable tile, no down-staircase found yet. Try `search` "
               "a few more times at a suspicious dead-end, or `move`/`kick` to open new ground, "
               "then call `explore_and_descend` again.")
    else:
        # Hit the per-call step budget mid-explore/search — there is MORE to do and the
        # search state persists across calls. The biggest descent mistake here is to
        # hand-search/move manually and get stuck; just re-invoke the skill — it resumes
        # the complete prioritized search from where it left off and descends when it finds `>`.
        fb += ("; used my step budget mid-search and did not reach a down-staircase yet. "
               "Call `explore_and_descend` AGAIN to continue — it resumes the complete "
               "search for the hidden downstairs from where it stopped. Do NOT hand-search "
               "tile-by-tile; that is what this skill does for you.")
    return SkillResult(actions=[], feedback=fb, pre_executed=True, pre_reward=state["r"],
                       final_obs=state["obs"], pre_terminated=state["term"],
                       pre_truncated=state["trunc"])


def list_skills() -> list[str]:
    return sorted(registry._skills.keys())


# ---------- journal skills (no env step, pure state mutation) ----------

@registry.register("add_note", schema={
    "description": (
        "Write or overwrite a note under a keyed slot. Use this to remember "
        "facts that will be useful many turns later: item locations, monster "
        "behaviors, dungeon layout summaries. Keep notes terse and concrete."
    ),
    "parameters": {
        "key": {"type": "string", "description": "Short slug, e.g. 'altar_dlvl_4' or 'cursed_items'."},
        "text": {"type": "string", "description": "The note body."},
    },
})
def add_note(env: NetHackCoreEnv, obs: StructuredObservation, key: str, text: str) -> SkillResult:
    def op(j: Journal) -> str:
        return j.add_note(key, text)
    return SkillResult(actions=[], feedback="", journal_op=op)


@registry.register("recall", schema={
    "description": (
        "Search your notes for the given query substring. Returns matching "
        "(key, text) pairs. Use this when you suspect you've seen something "
        "before but can't remember the detail."
    ),
    "parameters": {"query": {"type": "string"}},
})
def recall(env: NetHackCoreEnv, obs: StructuredObservation, query: str) -> SkillResult:
    def op(j: Journal) -> str:
        hits = j.recall(query)  # includes objective under key 'objective'
        if not hits:
            if not j.notes and not j.objective:
                return "No notes recorded yet. Use add_note(key, text) to record findings first."
            keys = (["objective"] if j.objective else []) + list(j.notes.keys())
            return f"No matches for '{query}'. Existing keys: {', '.join(keys[:6])}"
        return "Recalled:\n" + "\n".join(f"  - {k}: {t}" for k, t in hits)
    return SkillResult(actions=[], feedback="", journal_op=op)


@registry.register("pin_objective", schema={
    "description": (
        "Set your current top-level objective. Always rendered into every "
        "observation thereafter. Replace as your strategy evolves."
    ),
    "parameters": {"text": {"type": "string"}},
})
def pin_objective(env: NetHackCoreEnv, obs: StructuredObservation, text: str) -> SkillResult:
    def op(j: Journal) -> str:
        return j.pin_objective(text)
    return SkillResult(actions=[], feedback="", journal_op=op)


# ---------- wiki skills ----------

@registry.register("wiki_lookup", schema={
    "description": (
        "Fetch a named NetHack wiki page (e.g. 'cockatrice', 'mine town'). "
        "Use when you've identified a specific monster or feature and want "
        "the canonical lore."
    ),
    "parameters": {"entity": {"type": "string", "description": "Page title (case-insensitive)"}},
})
def wiki_lookup(env: NetHackCoreEnv, obs: StructuredObservation, entity: str) -> SkillResult:
    from .wiki import get_index
    idx = get_index()
    page = idx.lookup(entity)
    if page is None:
        # Surface nearest matches via substring search so the agent can
        # retry with a valid title instead of giving up on the tool.
        hits = idx.search(entity, k=3)
        if hits:
            suggestions = ", ".join(h.title for h in hits)
            return SkillResult(
                actions=[],
                feedback=f"No exact page {entity!r}. Did you mean: {suggestions}? (call wiki_lookup with one of these titles)",
                interrupted=True,
            )
        return SkillResult(
            actions=[], feedback=f"No wiki page for {entity!r} and no fuzzy matches.",
            interrupted=True,
        )
    return SkillResult(
        actions=[], feedback=f"[wiki: {page.title}] {page.short()}",
        interrupted=True,
    )


@registry.register("wiki_search", schema={
    "description": (
        "Substring-search the NetHack wiki. Returns up to k pages whose "
        "title or body contain the query. Use for fuzzy / exploratory "
        "questions ('what beats stunning?')."
    ),
    "parameters": {
        "query": {"type": "string"},
        "k": {"type": "integer", "default": 3},
    },
})
def wiki_search(env: NetHackCoreEnv, obs: StructuredObservation, query: str, k: int = 3) -> SkillResult:
    from .wiki import get_index
    pages = get_index().search(query, k=int(k))
    if not pages:
        return SkillResult(
            actions=[], feedback=f"No wiki results for {query!r}.",
            interrupted=True,
        )
    body = "\n\n".join(f"[{p.title}] {p.short(200)}" for p in pages)
    return SkillResult(actions=[], feedback=body, interrupted=True)


# ---------- sub-experiment 1d: on-demand map delivery ----------

# Map grid extent (NLE dungeon area). Columns x are 0..78, rows y are 0..20;
# the map row y corresponds to tty row y+1 (row 0 is the top message line).
_MAP_MAX_X = 78
_MAP_MAX_Y = 20
_REVEAL_MAX_W = 40   # cap the revealed rectangle so a huge box can't blow up
_REVEAL_MAX_H = 20   # the prompt; larger requests are clamped, not rejected.


@registry.register("reveal", schema={
    "description": (
        "Reveal a rectangular region of the dungeon map as ASCII (for the "
        "bounding-box observation mode, where the map is otherwise hidden). "
        "Coordinates are map cells: x = column 0-78, y = row 0-20, inclusive. "
        "Returns the sub-rectangle as text; consumes NO game turn. Oversized "
        "or out-of-range requests are clamped to the grid."
    ),
    "parameters": {
        "x1": {"type": "integer", "description": "Left column (0-78)."},
        "y1": {"type": "integer", "description": "Top row (0-20)."},
        "x2": {"type": "integer", "description": "Right column (0-78)."},
        "y2": {"type": "integer", "description": "Bottom row (0-20)."},
    },
})
def reveal(env: NetHackCoreEnv, obs: StructuredObservation,
           x1: int, y1: int, x2: int, y2: int) -> SkillResult:
    # Skills receive (env, obs), not `state`; the raw tty grid is reachable via
    # the env's last observation (identical object to state["raw_obs"]).
    import numpy as np
    raw = getattr(env, "_last_observation", None)
    tty = getattr(raw, "tty_chars", None) if raw is not None else None
    if tty is None:
        return SkillResult(actions=[], feedback="reveal: map unavailable this turn.",
                           interrupted=True)
    try:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    except (TypeError, ValueError):
        return SkillResult(actions=[], feedback="reveal: coordinates must be integers.",
                           interrupted=True)
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    # Clamp to the map grid.
    x1 = max(0, min(_MAP_MAX_X, x1))
    x2 = max(0, min(_MAP_MAX_X, x2))
    y1 = max(0, min(_MAP_MAX_Y, y1))
    y2 = max(0, min(_MAP_MAX_Y, y2))
    # Bound the rectangle size.
    if x2 - x1 > _REVEAL_MAX_W:
        x2 = x1 + _REVEAL_MAX_W
    if y2 - y1 > _REVEAL_MAX_H:
        y2 = y1 + _REVEAL_MAX_H
    grid = np.asarray(tty)
    # Map row y = tty row y+1; slice rows [y1+1 .. y2+1], cols [x1 .. x2].
    sub = grid[y1 + 1: y2 + 2, x1: x2 + 1]
    rows = []
    for i, row in enumerate(sub):
        line = "".join(chr(int(c)) if int(c) else " " for c in row)
        rows.append(f"y{y1 + i:>2}: {line}")
    body = f"reveal (x{x1}-{x2}, y{y1}-{y2}):\n" + "\n".join(rows)
    if len(body) > 1600:  # defensive char cap
        body = body[:1600] + " ...(truncated)"
    return SkillResult(actions=[], feedback=body, interrupted=True)


@registry.register("request_map", schema={
    "description": (
        "Force the FULL map back into this turn's observation. Under the "
        "delayed-map observation mode the map is re-sent only when it "
        "materially changed; call this to refresh it on demand. Consumes NO "
        "game turn."
    ),
    "parameters": {},
})
def request_map(env: NetHackCoreEnv, obs: StructuredObservation) -> SkillResult:
    # The template's force flag (state["_force_map"]) is set in env_response,
    # which owns `state`; this skill only returns a confirmation with empty
    # actions so no NLE step is taken.
    return SkillResult(actions=[], feedback="Refreshing the full map this turn.",
                       interrupted=True)
