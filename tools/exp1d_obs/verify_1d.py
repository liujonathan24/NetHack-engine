"""In-process verification for sub-experiment 1d (observation-delivery variants).

Builds real netplay envs in-process and drives a few env_response turns with
synthetic tool calls, asserting from the returned user message:

  DM (delayed / on-demand map):
    - turn 1 shows the full `=== MAP ===` block;
    - a no-material-change turn (add_note) shows `=== MAP (unchanged...`;
    - after request_map the map is FULL again;
    - action feedback is present every turn.

  BBOX (bounding-box on-demand):
    - normal turns have NO inline map (only the hidden placeholder);
    - reveal(x1,y1,x2,y2) returns a cropped region in the feedback and consumes
      NO game turn (dlvl + in-game time unchanged).

Run:
  PYTHONPATH=.:environments/nethack .venv/bin/python tools/exp1d_obs/verify_1d.py
"""
import asyncio
import json

from nethack import load_environment


def _tc(name, args):
    return {"function": {"name": name, "arguments": json.dumps(args)}}


def _text(resp):
    r0 = resp[0]
    content = r0["content"] if isinstance(r0, dict) else getattr(r0, "content", "")
    return content if isinstance(content, str) else json.dumps(content)


# netplay tools + the 1d on-demand-map tools (reveal/request_map are NOT in the
# default netplay set — they must be added explicitly for the 1d arms).
_SKILL_SET_1D = (
    "move_to,explore_and_descend,attack,throw,descend,search,pickup,"
    "engrave_elbereth,pray,eat,quaff,read,kick,add_note,recall,pin_objective,"
    "wiki_lookup,wiki_search,reveal,request_map"
)


async def _setup(variant):
    env = load_environment(
        tier="full_nle", variant=variant, skill_set=_SKILL_SET_1D,
        compact_obs=False, max_turns=40, character="Val-hum-neu-fem",
        explicit_seeds=[0],
    )
    ex = env.dataset[0]
    state = {"task": ex["task"], "info": ex.get("info", {})}
    state = await env.setup_state(state)
    return env, state


def _has_full_map(text):
    # The full ASCII map block header is exactly "=== MAP ===". The delayed
    # placeholder is "=== MAP (unchanged..." and never contains that exact token.
    return "=== MAP ===" in text


async def verify_dm():
    env, state = await _setup("DM")

    # Turn 1: any real action; prev fingerprint is None -> full map shown.
    r1 = _text(await env.env_response(
        [{"role": "assistant", "tool_calls": [_tc("search", {})]}], state))
    assert _has_full_map(r1), f"DM turn1: expected full map, got:\n{r1[:400]!r}"
    assert "[" in r1, "DM turn1: expected action feedback in prefix"
    print("PASS DM turn1: full === MAP === shown")

    # Turn 2: add_note is info-only (no NLE step) -> map_view unchanged ->
    # placeholder instead of the full map.
    r2 = _text(await env.env_response(
        [{"role": "assistant", "tool_calls": [_tc("add_note", {"key": "k", "text": "v"})]}], state))
    assert "=== MAP (unchanged" in r2, f"DM turn2: expected unchanged placeholder, got:\n{r2[:400]!r}"
    assert not _has_full_map(r2), "DM turn2: full map should be elided on unchanged turn"
    print("PASS DM turn2: === MAP (unchanged; call request_map to refresh) === shown, full map elided")

    # Turn 3: request_map forces the full map back this turn.
    r3 = _text(await env.env_response(
        [{"role": "assistant", "tool_calls": [_tc("request_map", {})]}], state))
    assert _has_full_map(r3), f"DM turn3: expected full map after request_map, got:\n{r3[:400]!r}"
    assert "Refreshing the full map" in r3, "DM turn3: expected request_map feedback"
    print("PASS DM turn3: request_map refreshed full map + feedback present")
    print("DM OK\n")


async def verify_bbox():
    env, state = await _setup("BBOX")

    # A normal turn has NO inline map, only the hidden placeholder.
    r1 = _text(await env.env_response(
        [{"role": "assistant", "tool_calls": [_tc("search", {})]}], state))
    assert not _has_full_map(r1), f"BBOX: full map should never render inline, got:\n{r1[:400]!r}"
    assert "=== MAP (hidden" in r1, f"BBOX: expected hidden-map placeholder, got:\n{r1[:400]!r}"
    print("PASS BBOX turn1: map hidden (=== MAP (hidden; call reveal...) ===), no inline grid")

    # Snapshot game progress before reveal to prove it consumes no NLE step.
    bl = state["raw_obs"].blstats
    px, py = int(bl[0]), int(bl[1])
    dlvl_before = state.get("max_dlvl_reached")
    time_before = state["structured_obs"].status.get("time")

    x1, y1 = max(0, px - 3), max(0, py - 2)
    x2, y2 = min(78, px + 3), min(20, py + 2)
    r2 = _text(await env.env_response(
        [{"role": "assistant", "tool_calls": [
            _tc("reveal", {"x1": x1, "y1": y1, "x2": x2, "y2": y2})]}], state))
    assert "reveal (x" in r2, f"BBOX: reveal should return a cropped region, got:\n{r2[:400]!r}"
    assert not _has_full_map(r2), "BBOX: reveal turn must not render the full inline map"

    dlvl_after = state.get("max_dlvl_reached")
    time_after = state["structured_obs"].status.get("time")
    assert dlvl_after == dlvl_before, f"BBOX: reveal changed dlvl {dlvl_before}->{dlvl_after}"
    assert time_after == time_before, f"BBOX: reveal consumed a game turn {time_before}->{time_after}"
    print(f"PASS BBOX turn2: reveal returned a crop; NO game turn consumed "
          f"(dlvl {dlvl_before}=={dlvl_after}, time {time_before}=={time_after})")
    # Show the crop so the evidence is visible.
    crop = [ln for ln in r2.splitlines() if ln.startswith("[reveal (x") or ln.startswith("reveal (x") or ln.strip().startswith("y")]
    print("   crop preview:", crop[:3])
    print("BBOX OK\n")


async def main():
    await verify_dm()
    await verify_bbox()
    print("1d VERIFY OK")


if __name__ == "__main__":
    asyncio.run(main())
