"""The NetPlay skill_set profile (Jeurissen, CoG 2024): high-level skill-only
action surface used as the standardized action set for cross-encoding benchmarks.
"""
from nethack_harness.helpers import _build_skill_adapter_callables

NETPLAY_EXPECTED = {
    "move_to", "explore_and_descend",
    "attack", "throw", "descend", "search", "pickup", "engrave_elbereth", "pray", "eat",
    "quaff", "read", "kick", "add_note", "recall", "pin_objective",
    "wiki_lookup", "wiki_search",
}


def _names(skill_set):
    return {c.__name__ for c in _build_skill_adapter_callables(skill_set=skill_set)}


def test_netplay_profile_exposes_the_paper_action_set():
    assert _names("netplay") == NETPLAY_EXPECTED


def test_netplay_drops_low_level_move_keeps_high_level_navigation():
    names = _names("netplay")
    assert "move" not in names  # NetPlay has no low-level move(direction) primitive
    assert {"move_to", "explore_and_descend"} <= names  # high-level navigation
    # weak open-loop navigators are superseded by explore_and_descend
    assert "autoexplore" not in names and "find_and_descend" not in names
    # harness-owned skills are never exposed
    assert "menu_option" not in names and "inventory_item" not in names


def test_netplay_differs_from_full_and_move():
    assert _names("netplay") != _names("full")
    assert _names("netplay") != _names("move")
