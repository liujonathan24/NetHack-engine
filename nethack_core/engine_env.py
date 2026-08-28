"""
nethack_core.engine_env
=======================

Engine-backed NetHack env, driven directly by the custom fork engine
(``RawEngine``) instead of the ``nle`` gym wrapper used by
:class:`nethack_core.env.NetHackCoreEnv`.

Because it talks to the fork engine, it exposes capabilities the nle path
cannot:

    * ``snapshot()`` / ``restore()`` / ``free_snapshot()`` — O(arena) in-memory
      checkpoints for replay and branching (byte-exact within a dungeon level).
    * ``tune`` (and ``get_tune``/``set_tune``) — parametric difficulty knobs.

It keeps the same deterministic seed-before-reset discipline and returns the
same :class:`CoreObservation` / :class:`EpisodeMetadata` types as
``NetHackCoreEnv`` so downstream layers are unaffected.

This is intentionally a thin, honest surface: ``step`` returns
``(observation, done, info)`` — reward is a task-layer concern and is not
computed here.
"""

from __future__ import annotations

import base64
import hashlib
import json
import pathlib
from typing import Optional

import numpy as np

from ._engine import (
    COLNO,
    NLE_BLSTATS_SIZE,
    NLE_INVENTORY_SIZE,
    NLE_INVENTORY_STR_LENGTH,
    NLE_MESSAGE_SIZE,
    NLE_TERM_CO,
    NLE_TERM_LI,
    ROWNO,
    RawEngine,
)
from .env import CoreObservation, EpisodeMetadata


class _TuneProxy:
    """``env.tune.get()`` / ``env.tune.set(**knobs)`` view over the engine knobs."""

    def __init__(self, engine: RawEngine) -> None:
        self._engine = engine

    def get(self) -> dict:
        return self._engine.get_tune()

    def set(self, **knobs) -> "_TuneProxy":
        self._engine.set_tune(**knobs)
        return self

    def catalog(self) -> list:
        return self._engine.tune_catalog()


class EngineEnv:
    """Deterministic NetHack env over the fork engine with snapshot + tune."""

    #: Whitelist of secure-mutable state fields -> inclusive (lo, hi) bounds.
    #: goto_depth is handled separately (the engine validates its upper bound).
    _MODIFY_BOUNDS = {
        "hp": (0, 30000),
        "max_hp": (1, 30000),
        "gold": (0, 10_000_000),
        "xp_level": (1, 30),
        "hunger": (0, 2000),
        # level_up is an INCREMENT (gain N experience levels with real HP/stat
        # gains via the engine's pluslvl), distinct from xp_level (direct set).
        "level_up": (1, 29),
        # Attributes (NetHack-encoded). str spans 3..125 (19..118 == 18/01..18/00,
        # 119..125 == 19..25); the others are plain 3..25. Used by the curriculum
        # stats-only deep-jump upgrade.
        "str": (3, 125),
        "dex": (3, 25),
        "con": (3, 25),
        "int": (3, 25),
        "wis": (3, 25),
        "cha": (3, 25),
        # Luck: intrinsic u.uluck. Vanilla intrinsic luck is bounded -10..10
        # (LUCKMIN..LUCKMAX); the max *effective* luck (Luck = u.uluck +
        # moreluck) with a luck stone is +13, so we allow -13..13 here to let
        # the "better items / luck" ablation reach full effective range without
        # a luck stone. Routed to the C nle_set_state("luck", n) case, which
        # applies the same clamp. Luck feeds to-hit, prayer, theft, drop rolls.
        "luck": (-13, 13),
    }

    def __init__(self, modify: Optional[dict] = None) -> None:
        self._engine = RawEngine()
        self._pending_seeds: Optional[tuple[int, int]] = None
        self._current_seeds: Optional[tuple[int, int]] = None
        #: Default modify config applied on every reset() (unless reset overrides).
        self._default_modify = dict(modify) if modify else None
        #: ``env.tune.get()`` / ``env.tune.set(**knobs)`` difficulty-knob view.
        self.tune = _TuneProxy(self._engine)

    # ----- lifecycle -----

    def seed(self, core: int, disp: Optional[int] = None) -> tuple[int, int]:
        """Stage seeds for the next reset() (disp defaults to core)."""
        if disp is None:
            disp = core
        self._pending_seeds = (int(core), int(disp))
        return self._pending_seeds

    def reset(
        self,
        *,
        seeds: Optional[tuple[int, int]] = None,
        tune: Optional[dict] = None,
        modify: Optional[dict] = None,
        character: Optional[str] = None,
    ) -> tuple[CoreObservation, EpisodeMetadata]:
        """Start a fresh game. Seeds must be staged via seed() or passed here.

        Refusing to reset without explicit seeds keeps trajectories
        reproducible by construction (mirrors NetHackCoreEnv).

        ``tune`` applies difficulty-knob overrides before the starting level is
        generated, so generation-time knobs (e.g. room_density) reshape the
        starting floor. This is the "regenerate with these knobs" path.

        ``modify`` applies whitelisted, bounds-checked state mutations (and an
        optional ``goto_depth`` dungeon jump) AFTER the game has started, via
        :meth:`modify`. If not passed, the per-instance default from
        ``__init__(modify=...)`` is applied instead. Pass ``modify={}`` to apply
        no mutations regardless of the default.
        """
        if seeds is not None:
            self._pending_seeds = seeds
        if self._pending_seeds is None:
            raise RuntimeError(
                "EngineEnv.reset() requires seeds. Call .seed(core, disp) first "
                "or pass seeds= explicitly."
            )
        core, disp = self._pending_seeds
        self._engine.start(core, disp, tune=tune, character=character)
        self._current_seeds = self._pending_seeds
        self._pending_seeds = None
        meta = EpisodeMetadata(
            seeds=self._current_seeds,
            seed_hash=_hash_seeds(*self._current_seeds),
            task_name="engine",
        )
        obs = self._engine.to_core_observation()
        effective_modify = modify if modify is not None else self._default_modify
        if effective_modify:
            obs = self.modify(**effective_modify)
        return obs, meta

    # ----- secure state mutation -----

    def modify(self, **changes) -> CoreObservation:
        """Apply whitelisted, bounds-checked state mutations.

        Validates the WHOLE call first (unknown fields or out-of-range values
        raise before any engine write happens, so no partial/arbitrary writes),
        then applies each field via the engine.

        ``goto_depth=n`` jumps the dungeon level (the engine validates the upper
        bound) AND seats the hero on that level's downstair, so the Map Viewer
        "skip to level N and spawn on the ``>``" workflow lands on the stair.

        ``level_up=n`` raises the hero ``n`` experience levels with real HP/stat
        gains (an increment, distinct from ``xp_level`` which is a direct set).

        Returns the refreshed CoreObservation. Exactly one redraw happens:
        ``goto_depth`` (via seat_on_stair) and ``level_up`` each already render,
        so the trailing ctrl-R is only issued when neither was requested.
        """
        depth = changes.pop("goto_depth", None)
        for k, v in changes.items():
            if k not in self._MODIFY_BOUNDS:
                allowed = sorted(self._MODIFY_BOUNDS) + ["goto_depth"]
                raise KeyError(
                    f"unknown modify field {k!r}; allowed: {allowed}"
                )
            lo, hi = self._MODIFY_BOUNDS[k]
            if not (lo <= int(v) <= hi):
                raise ValueError(f"{k}={v} out of range [{lo},{hi}]")
        if depth is not None and not (1 <= int(depth) <= 60):
            raise ValueError(f"goto_depth={depth} out of range")
        # level_up is an engine action (pluslvl), not a blstats field set.
        levels = changes.pop("level_up", None)
        rendered = False
        for k, v in changes.items():
            self._engine.set_state(k, int(v))
        if levels is not None:
            self._engine.level_up(int(levels))  # renders via ctrl-R
            rendered = True
        if depth is not None:
            self._engine.goto_depth(int(depth))
            # seat the hero on the level's downstair (also renders)
            self._engine.seat_on_stair(down=True)
            rendered = True
        if not rendered:
            # ctrl-R redraw so blstats refresh without consuming a game turn.
            self._engine.step(18)
        return self._engine.to_core_observation()

    def goto_abs(self, dnum: int, dlevel: int, seat: bool = False) -> CoreObservation:
        """Jump to an arbitrary ``(dnum, dlevel)`` across dungeon branches.

        Unlike ``modify(goto_depth=n)`` (confined to the current branch), this
        reaches Gehennom and the Elemental Planes. The hero lands on a valid
        random spot on the destination level.

        ``seat=True`` additionally seats the hero on the level's downstair, but
        note the extra render step corrupts long *sequences* of jumps (the
        deferred-goto / seat double-step desyncs), so the curriculum (which
        chains many jumps) leaves it off. Returns the refreshed observation.
        """
        self._engine.goto_abs(int(dnum), int(dlevel))
        if seat:
            self._engine.seat_on_stair(down=True)
        return self._engine.to_core_observation()

    def dungeon_table(self) -> list:
        """Return the dungeon-branch layout (name/depth_start/num_dunlevs)."""
        return self._engine.dungeon_table()

    def grant_invocation_kit(self) -> CoreObservation:
        """Grant the pre-primed invocation artifacts into the hero's pack.

        Dedicated engine call (NOT a ``modify()`` scalar field) — it injects the
        lit 7-candle Candelabrum, charged Bell of Opening, and Book of the Dead
        (all uncursed) via addinv, so the curriculum agent can perform the
        invocation ritual on the Invocation level. Returns the refreshed obs.
        """
        self._engine.grant_invocation_kit()
        return self._engine.to_core_observation()

    def invocation_pos(self):
        """(x, y) of the vibrating square on the Invocation level, else None."""
        return self._engine.invocation_pos()

    def seat_on_invocation_square(self, adjacent: bool = True) -> CoreObservation:
        """Stage the hero at the vibrating square (ritual site). adjacent=True lands
        the hero next to it (agent steps on); adjacent=False lands on it. Returns obs."""
        self._engine.seat_on_invocation_square(adjacent=adjacent)
        return self._engine.to_core_observation()

    @property
    def done(self) -> bool:
        """Whether the current game has ended (hero dead / ascended / quit)."""
        return self._engine.done

    @property
    def how_done(self) -> int:
        """Engine end-reason code (only meaningful when ``done``)."""
        return self._engine.how_done

    def step(self, action: int) -> tuple[CoreObservation, bool, dict]:
        """Advance one action. Returns (observation, done, info)."""
        self._engine.step(action)
        done = self._engine.done
        info = {"how_done": self._engine.how_done} if done else {}
        return self._engine.to_core_observation(), done, info

    def close(self) -> None:
        self._engine.end()

    # ----- snapshot / restore (replay & branching) -----

    def snapshot(self):
        """Capture the current game state; returns an opaque handle."""
        return self._engine.snapshot()

    def restore(self, handle) -> CoreObservation:
        """Restore to a snapshot handle.

        The returned observation still reflects the most recent step's frame;
        the restored state appears in observations after the next step()
        (see RawEngine.restore).
        """
        self._engine.restore(handle)
        return self._engine.to_core_observation()

    def free_snapshot(self, handle) -> None:
        self._engine.free_snapshot(handle)

    def branch(
        self,
        n: int,
        reseed: bool = True,
        horizon: int = 40,
        action: int = ord("s"),
    ) -> list[list[bytes]]:
        """Return ``n`` continuations from the current state.

        Takes one snapshot of the current state, then restores it ``n`` times.
        With ``reseed=True`` each branch reseeds the gameplay RNG AFTER restore
        (order matters: the snapshot captures the RNG, so reseed must follow
        restore) so random-chance events diverge across branches.  With
        ``reseed=False`` the branches replay byte-identically.

        Each branch is rolled out for ``horizon`` steps of ``action`` and
        returned as a per-step trace of the map ``chars`` (one bytes object per
        step), so callers can compare branches for divergence.

        The snapshot handle is reused across all ``n`` restores (RawEngine
        snapshots support repeated restore) and freed before returning.
        """
        handle = self.snapshot()
        try:
            results: list[list[bytes]] = []
            for i in range(n):
                self.restore(handle)
                if reseed:
                    self._engine.reseed(core=1000 + i, disp=2000 + i)
                trace: list[bytes] = []
                for _ in range(horizon):
                    obs, _done, _info = self.step(action)
                    trace.append(obs.chars.tobytes())
                results.append(trace)
            return results
        finally:
            self.free_snapshot(handle)

    # ----- portable level blob save / load -----

    def save_level(self, path) -> None:
        """Serialize the current level to a portable blob written to ``path``."""
        pathlib.Path(path).write_bytes(self._engine.save_level())

    def load_level(self, path) -> CoreObservation:
        """Load a level blob from ``path`` as the current level.

        Returns the re-rendered observation.  The underlying C load is two-phase;
        RawEngine.load_level steps once (ctrl-R) internally to redraw, so the
        observation returned here already reflects the loaded level.
        """
        self._engine.load_level(pathlib.Path(path).read_bytes())
        return self._engine.to_core_observation()

    # ----- resumable checkpoint = seed + level blob + player blob -----

    def save_player(self, path) -> None:
        """Serialize the hero (u + inventory + attributes) to a blob at ``path``."""
        pathlib.Path(path).write_bytes(self._engine.save_player())

    def checkpoint(self, path) -> None:
        """Write a resumable checkpoint = seed + level blob + player blob +
        the whole off-current dungeon-level file set.

        The seed lets resume() reproduce object appearances/ids; the level and
        player blobs are the floor + hero.

        ``levelfiles`` is NOT optional.  ``save_player`` runs ``savegamestate``
        -> ``save_dungeon`` (dungeon.c:175), which serializes the entire
        ``level_info[]`` array — so the resumed heap believes in every level the
        original game had visited.  Resuming into a fresh, empty hackdir without
        those files makes the first stair transition open a level file that is
        not there, which NetHack turns into ``done(TRICKED)``: a full-HP
        game-over that looks exactly like a monster kill downstream.  See
        :class:`nethack_core._engine.CheckpointIntegrityError`.
        """
        data = {
            "seed": list(self._current_seeds),
            "level": base64.b64encode(self._engine.save_level()).decode(),
            "player": base64.b64encode(self._engine.save_player()).decode(),
            "levelfiles": base64.b64encode(self._engine.save_levelfiles()).decode(),
        }
        pathlib.Path(path).write_text(json.dumps(data))

    def resume(self, path) -> CoreObservation:
        """Restore a checkpoint: reset to its seed (so object appearances/ids
        match), install the dungeon's level files, then the level, then the
        hero, then render once.

        Honors the hard ordering contract: load_level_raw before
        load_player_raw, then a single step(18) render.  Returns the obs.

        The level FILES go in LAST, after both blobs and before the render.
        They cannot go in first: ``nle_load_level`` stamps the level blob over
        ``<lock>.<ledger_no(u.uz)>``, and at that point ``u.uz`` is still the
        freshly-reset level 1 — so installing first means the level-1 file gets
        overwritten with the checkpoint's floor, and returning to level 1 later
        reads a file whose dlvl header says something else, which
        ``getlev`` answers with ``trickery()`` -> ``done(TRICKED)``: another
        full-HP fake death (restore.c:1171).

        Raises :class:`CheckpointIntegrityError` if the resumed heap still
        references a level file that is not on disk.
        """
        d = json.loads(pathlib.Path(path).read_text())
        core, disp = d["seed"]
        self.reset(seeds=(int(core), int(disp)))
        # Dismiss the fresh game's welcome --More-- BEFORE loading anything.
        # Otherwise the single ctrl-R below is eaten by that prompt instead of
        # running docrt(), and the returned observation keeps showing the
        # reset's level-1 starting room while blstats describe the checkpoint's
        # floor: a map/status mismatch that the old gold+depth smoke test could
        # not see. ESC (not CR) because ESC is a no-op in command context.
        self._engine.step(27)
        self._engine.load_level_raw(base64.b64decode(d["level"]))   # LEVEL first
        self._engine.load_player_raw(base64.b64decode(d["player"]))  # THEN player
        lf = d.get("levelfiles")
        if lf:  # ...and the off-current dungeon LAST (see above)
            self._engine.load_levelfiles(base64.b64decode(lf))
        self._engine.step(18)  # single two-phase render after BOTH
        self._engine.check_level_file_integrity("after resume")
        return self._engine.to_core_observation()

    # ----- level-file integrity (see CheckpointIntegrityError) -----

    def missing_level_files(self) -> int:
        """Ledgers flagged ``LFILE_EXISTS`` whose level file is absent."""
        return self._engine.missing_level_files()

    def check_level_file_integrity(self, where: str = "") -> None:
        """Raise ``CheckpointIntegrityError`` if any level file is missing."""
        self._engine.check_level_file_integrity(where)

    # ----- difficulty knobs -----

    def get_tune(self) -> dict:
        return self._engine.get_tune()

    def set_tune(self, **knobs) -> "EngineEnv":
        self._engine.set_tune(**knobs)
        return self

    # ----- properties -----

    @property
    def done(self) -> bool:
        return self._engine.done

    @property
    def current_seeds(self) -> Optional[tuple[int, int]]:
        return self._current_seeds

    @property
    def engine(self) -> RawEngine:
        """Escape hatch for the raw engine (snapshot internals, obs buffers)."""
        return self._engine

    @property
    def observation_space(self):
        """A gymnasium Dict space describing the CoreObservation buffers.

        Built lazily from the engine layout constants (so it is valid before the
        first reset and does not require gymnasium at import time). Only
        gym-shaped consumers (e.g. the PufferLib adapter) need this; the engine
        itself works on raw numpy buffers.
        """
        import gymnasium as gym

        u8 = lambda shape: gym.spaces.Box(  # noqa: E731
            low=0, high=255, shape=shape, dtype=np.uint8
        )
        i16 = lambda shape: gym.spaces.Box(  # noqa: E731
            low=np.iinfo(np.int16).min, high=np.iinfo(np.int16).max,
            shape=shape, dtype=np.int16,
        )
        i64 = lambda shape: gym.spaces.Box(  # noqa: E731
            low=np.iinfo(np.int64).min, high=np.iinfo(np.int64).max,
            shape=shape, dtype=np.int64,
        )
        map_shape = (ROWNO, COLNO - 1)
        tty_shape = (NLE_TERM_LI, NLE_TERM_CO)
        return gym.spaces.Dict({
            "tty_chars": u8(tty_shape),
            "tty_colors": u8(tty_shape),
            "tty_cursor": u8((2,)),
            "glyphs": i16(map_shape),
            "chars": u8(map_shape),
            "colors": u8(map_shape),
            "message": u8((NLE_MESSAGE_SIZE,)),
            "inv_strs": u8((NLE_INVENTORY_SIZE, NLE_INVENTORY_STR_LENGTH)),
            "inv_letters": u8((NLE_INVENTORY_SIZE,)),
            "inv_glyphs": i16((NLE_INVENTORY_SIZE,)),
            "blstats": i64((NLE_BLSTATS_SIZE,)),
        })


def _hash_seeds(core: int, disp: int) -> str:
    return hashlib.sha256(f"{core}:{disp}".encode()).hexdigest()[:12]
