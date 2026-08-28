import ctypes
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np


class EngineNotBuilt(RuntimeError):
    """Raised when libnethack.so cannot be located."""


class CheckpointIntegrityError(RuntimeError):
    """A restored/resumed game's heap disagrees with the level files on disk.

    NetHack's ``goto_level`` (do.c:1451) branches on the in-memory flag
    ``level_info[ledger].linfo_flags & LFILE_EXISTS``.  When that flag is set
    and the on-disk ``<lock>.<ledger>`` file is missing, it calls
    ``tricked_fileremoved`` (save.c:487), which prints ``Cannot open file ...``
    / ``Probably someone removed it`` and then ``done(TRICKED)`` — an INSTANT
    game-over at full HP that is indistinguishable downstream from a monster
    kill.  Three E15 rollouts recorded exactly that as a death.

    A corrupted checkpoint must never present as a death, so every path that
    can produce this divergence raises instead.  Callers should let it
    propagate: it means the checkpoint, not the hero, is broken.
    """


def _engine_build_dir() -> "Path | None":
    """Locate ``third_party/NetHack/src/build`` by walking up from this file.

    Location-independent so the binding works whether it lives in the canonical
    ``nethack_core/`` package or a vendored copy at a different depth.
    """
    for parent in Path(__file__).resolve().parents:
        cand = parent / "third_party" / "NetHack" / "src" / "build"
        if cand.exists():
            return cand
    return None


def library_path() -> Path:
    env = os.environ.get("NLE_LIB_PATH")
    if env:
        # NLE_LIB_PATH is authoritative: if set but missing, raise immediately.
        p = Path(env)
        if p.exists():
            return p
        raise EngineNotBuilt(
            f"NLE_LIB_PATH={env!r} does not exist. Build it with "
            "nethack_core/build_engine.sh "
            "(or set NLE_LIB_PATH to a valid path). Toolchain: cmake/bison/flex/libbz2."
        )
    build = _engine_build_dir()
    if build is not None:
        default = build / "libnethack.so"
        if default.exists():
            return default
    # Bundled binary: when this package is installed standalone (the Prime Hub
    # wheel) there is no third_party/ source tree, so the build script copies the
    # prebuilt libnethack.so next to this module (see tools/bundle_for_hub.py).
    bundled = Path(__file__).resolve().parent / "libnethack.so"
    if bundled.exists():
        return bundled
    raise EngineNotBuilt(
        "libnethack.so not found. Build it with nethack_core/build_engine.sh "
        "(or set NLE_LIB_PATH), or bundle it via tools/bundle_for_hub.py. "
        "Toolchain: cmake/bison/flex/libbz2."
    )


_LIB = None


def load_library() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        _LIB = ctypes.CDLL(str(library_path()))
    return _LIB


# ---------------------------------------------------------------------------
# Constants (verified against the fork's include/nleobs.h + sys/unix/nle.c)
# ---------------------------------------------------------------------------

NLE_BLSTATS_SIZE = 27
NLE_MESSAGE_SIZE = 256
NLE_PROGRAM_STATE_SIZE = 6
NLE_INTERNAL_SIZE = 9
NLE_MISC_SIZE = 3
NLE_INVENTORY_SIZE = 55
NLE_INVENTORY_STR_LENGTH = 80
NLE_SCREEN_DESCRIPTION_LENGTH = 80
NLE_TERM_CO = 80
NLE_TERM_LI = 24
ROWNO = 21
COLNO = 80
MAP = ROWNO * (COLNO - 1)   # 1659
TTY = NLE_TERM_LI * NLE_TERM_CO  # 1920

# Options string that fully specifies the character so the engine never blocks
# on interactive prompts during game creation.
_NETHACKOPTIONS = (
    "autopickup", "color", "disclose:+i +a +v +g +c +o", "mention_walls",
    "nobones", "nocmdassist", "nolegacy", "nosparkle",
    "pickup_burden:unencumbered", "pickup_types:$?!/", "runmode:teleport",
    "showexp", "showscore", "time",
)
# NetHack parses role/race/alignment/gender from the player-name suffix
# (name:Agent-<role>-<race>-<align>-<gender>). The character is therefore
# selected by the suffix passed to start(character=...); the default keeps the
# historical Monk so existing callers are unchanged. The curriculum env passes
# "Val-hum-neu-fem" (female neutral human Valkyrie).
_OPTIONS_BASE = ",".join(_NETHACKOPTIONS)
_DEFAULT_CHARACTER = "mon-hum-neu-mal"
_OPTIONS_STR = _OPTIONS_BASE + ",name:Agent-" + _DEFAULT_CHARACTER


# The difficulty-knob catalog (names + order) is fixed for a given libnethack.so,
# so read it from the engine once per process and cache it.
_TUNE_NAMES = None


def tune_names(lib) -> list:
    """Return the ordered list of difficulty-knob names from the engine."""
    global _TUNE_NAMES
    if _TUNE_NAMES is None:
        n = lib.nle_tune_count()
        _TUNE_NAMES = [lib.nle_tune_name(i).decode("ascii") for i in range(n)]
    return _TUNE_NAMES


# ---------------------------------------------------------------------------
# ctypes struct mirrors — field order MUST match the C definitions exactly.
# ---------------------------------------------------------------------------

class NleObs(ctypes.Structure):
    _fields_ = [
        ("action", ctypes.c_int),
        ("done", ctypes.c_int),
        ("in_normal_game", ctypes.c_char),
        ("how_done", ctypes.c_int),
        ("glyphs", ctypes.POINTER(ctypes.c_short)),
        ("chars", ctypes.POINTER(ctypes.c_uint8)),
        ("colors", ctypes.POINTER(ctypes.c_uint8)),
        ("specials", ctypes.POINTER(ctypes.c_uint8)),
        ("blstats", ctypes.POINTER(ctypes.c_long)),
        ("message", ctypes.POINTER(ctypes.c_uint8)),
        ("program_state", ctypes.POINTER(ctypes.c_int)),
        ("internal", ctypes.POINTER(ctypes.c_int)),
        ("inv_glyphs", ctypes.POINTER(ctypes.c_short)),
        ("inv_strs", ctypes.POINTER(ctypes.c_uint8)),
        ("inv_letters", ctypes.POINTER(ctypes.c_uint8)),
        ("inv_oclasses", ctypes.POINTER(ctypes.c_uint8)),
        ("screen_descriptions", ctypes.POINTER(ctypes.c_uint8)),
        ("tty_chars", ctypes.POINTER(ctypes.c_uint8)),
        ("tty_colors", ctypes.POINTER(ctypes.c_int8)),
        ("tty_cursor", ctypes.POINTER(ctypes.c_uint8)),
        ("misc", ctypes.POINTER(ctypes.c_int)),
    ]


class NleSeedsInit(ctypes.Structure):
    """NLE_ALLOW_SEEDING is defined in this fork's build."""
    _fields_ = [("seeds", ctypes.c_ulong * 2), ("reseed", ctypes.c_char)]


NLE_TUNE_MAX = 64


class NleSettings(ctypes.Structure):
    _fields_ = [
        ("hackdir", ctypes.c_char * 256),
        ("scoreprefix", ctypes.c_char * 256),
        ("options", ctypes.c_char * 512),
        ("wizkit", ctypes.c_char * 256),
        ("spawn_monsters", ctypes.c_int),
        ("ttyrecname", ctypes.c_char * 256),
        # Difficulty-knob overrides applied before the starting level is built.
        # Zero-safe: tune_n == 0 means no overrides (vanilla defaults).
        ("tune_n", ctypes.c_int),
        ("tune_idx", ctypes.c_int * NLE_TUNE_MAX),
        ("tune_val", ctypes.c_double * NLE_TUNE_MAX),
        # Optional shared read-only data dir. When set, the immutable game data
        # (DLB nhdat, data/oracles/rumors, config) is read from here and only the
        # writable game-state files live under hackdir — so no per-reset copy of
        # the ~3.7MB dat tree is needed. Empty => every prefix uses hackdir.
        ("datadir", ctypes.c_char * 256),
    ]


# ---------------------------------------------------------------------------
# RawEngine
# ---------------------------------------------------------------------------

class RawEngine:
    """Thin ctypes wrapper around the fork's nle_start/nle_step/nle_end API.

    Multiple independent RawEngine instances can run concurrently in the same
    process, including across threads: each instance owns its own nle_ctx_t,
    and the engine anchors a thread-local current_nle_ctx on every call (the
    fork migrated NetHack's globals into nle_ctx_t for exactly this).
    Per-env files are isolated via a per-instance temp hackdir.

    Call start() before step(), and end() when done.  start() may be called
    again after end() for a fresh game.
    """

    def __init__(self) -> None:
        self._lib = load_library()
        self._setup_argtypes()

        # Allocate all observation buffers as contiguous numpy arrays.
        # They MUST remain referenced on self to prevent GC while the C side
        # holds raw pointers into them.
        self._glyphs = np.zeros(MAP, dtype=np.int16)
        self._chars = np.zeros(MAP, dtype=np.uint8)
        self._colors = np.zeros(MAP, dtype=np.uint8)
        self._specials = np.zeros(MAP, dtype=np.uint8)
        self._blstats = np.zeros(NLE_BLSTATS_SIZE, dtype=np.int64)
        self._message = np.zeros(NLE_MESSAGE_SIZE, dtype=np.uint8)
        self._program_state = np.zeros(NLE_PROGRAM_STATE_SIZE, dtype=np.int32)
        self._internal = np.zeros(NLE_INTERNAL_SIZE, dtype=np.int32)
        self._inv_glyphs = np.zeros(NLE_INVENTORY_SIZE, dtype=np.int16)
        self._inv_strs = np.zeros(
            NLE_INVENTORY_SIZE * NLE_INVENTORY_STR_LENGTH, dtype=np.uint8
        )
        self._inv_letters = np.zeros(NLE_INVENTORY_SIZE, dtype=np.uint8)
        self._inv_oclasses = np.zeros(NLE_INVENTORY_SIZE, dtype=np.uint8)
        self._screen_descriptions = np.zeros(
            ROWNO * (COLNO - 1) * NLE_SCREEN_DESCRIPTION_LENGTH, dtype=np.uint8
        )
        self._tty_chars = np.zeros(TTY, dtype=np.uint8)
        self._tty_colors = np.zeros(TTY, dtype=np.int8)
        self._tty_cursor = np.zeros(2, dtype=np.uint8)
        self._misc = np.zeros(NLE_MISC_SIZE, dtype=np.int32)

        # Build the NleObs struct and populate all pointer fields.
        self._obs = NleObs()
        self._obs.glyphs = self._glyphs.ctypes.data_as(
            ctypes.POINTER(ctypes.c_short)
        )
        self._obs.chars = self._chars.ctypes.data_as(
            ctypes.POINTER(ctypes.c_uint8)
        )
        self._obs.colors = self._colors.ctypes.data_as(
            ctypes.POINTER(ctypes.c_uint8)
        )
        self._obs.specials = self._specials.ctypes.data_as(
            ctypes.POINTER(ctypes.c_uint8)
        )
        self._obs.blstats = self._blstats.ctypes.data_as(
            ctypes.POINTER(ctypes.c_long)
        )
        self._obs.message = self._message.ctypes.data_as(
            ctypes.POINTER(ctypes.c_uint8)
        )
        self._obs.program_state = self._program_state.ctypes.data_as(
            ctypes.POINTER(ctypes.c_int)
        )
        self._obs.internal = self._internal.ctypes.data_as(
            ctypes.POINTER(ctypes.c_int)
        )
        self._obs.inv_glyphs = self._inv_glyphs.ctypes.data_as(
            ctypes.POINTER(ctypes.c_short)
        )
        self._obs.inv_strs = self._inv_strs.ctypes.data_as(
            ctypes.POINTER(ctypes.c_uint8)
        )
        self._obs.inv_letters = self._inv_letters.ctypes.data_as(
            ctypes.POINTER(ctypes.c_uint8)
        )
        self._obs.inv_oclasses = self._inv_oclasses.ctypes.data_as(
            ctypes.POINTER(ctypes.c_uint8)
        )
        self._obs.screen_descriptions = self._screen_descriptions.ctypes.data_as(
            ctypes.POINTER(ctypes.c_uint8)
        )
        self._obs.tty_chars = self._tty_chars.ctypes.data_as(
            ctypes.POINTER(ctypes.c_uint8)
        )
        self._obs.tty_colors = self._tty_colors.ctypes.data_as(
            ctypes.POINTER(ctypes.c_int8)
        )
        self._obs.tty_cursor = self._tty_cursor.ctypes.data_as(
            ctypes.POINTER(ctypes.c_uint8)
        )
        self._obs.misc = self._misc.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        self._ctx = None
        #: Tiny per-env WRITABLE directory (level/save/bones/score/lock files),
        #: created once and reused across games. The large read-only game data is
        #: read directly from the shared source dat (settings.datadir), so this
        #: holds only the few small game-state files — no 3.7MB copy per reset.
        self._hackdir = None
        #: Writable templates the engine appends to (scores/logs/lock). Seeded
        #: empty in the writable dir on each reset so every game starts clean —
        #: matching the empty templates the old per-reset copytree provided.
        self._writable_templates = ("record", "logfile", "xlogfile", "perm")

        # Outstanding snapshot handles created by this instance.  A handle is
        # bound to the ctx that created it; end() frees any the caller leaked.
        self._snapshots = set()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _setup_argtypes(self) -> None:
        lib = self._lib

        lib.nle_start.restype = ctypes.c_void_p
        lib.nle_start.argtypes = [
            ctypes.POINTER(NleObs),
            ctypes.c_void_p,                # FILE* ttyrec (NULL = no recording)
            ctypes.POINTER(NleSeedsInit),
            ctypes.POINTER(NleSettings),
        ]

        lib.nle_step.restype = ctypes.c_void_p
        lib.nle_step.argtypes = [
            ctypes.c_void_p,                # nle_ctx_t*
            ctypes.POINTER(NleObs),
        ]

        lib.nle_end.restype = None
        lib.nle_end.argtypes = [ctypes.c_void_p]

        # In-memory snapshot / restore / branch.  The handle is an opaque,
        # self-contained malloc'd copy of (ctx + coroutine stack + arena).
        lib.nle_fr_snapshot.restype = ctypes.c_void_p
        lib.nle_fr_snapshot.argtypes = [ctypes.c_void_p]
        lib.nle_fr_restore.restype = None
        lib.nle_fr_restore.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.nle_fr_destroy.restype = None
        lib.nle_fr_destroy.argtypes = [ctypes.c_void_p]

        # Level-file integrity (see CheckpointIntegrityError).  A snapshot also
        # bundles the off-current dungeon levels that live on disk; these expose
        # whether that bundle is complete and whether the live heap's
        # LFILE_EXISTS flags still match the hackdir.
        lib.nle_fr_last_restore_rc.restype = ctypes.c_int
        lib.nle_fr_last_restore_rc.argtypes = []
        lib.nle_fr_levelfile_count.restype = ctypes.c_int
        lib.nle_fr_levelfile_count.argtypes = [ctypes.c_void_p]
        lib.nle_fr_missing_levelfiles.restype = ctypes.c_int
        lib.nle_fr_missing_levelfiles.argtypes = [ctypes.c_void_p]

        # Portable level-file SET, for cross-process checkpoints.  The
        # fast-reset handle itself is a live-process construct (coroutine stack
        # + arena bytes + display mirror) and can never be written to disk.
        lib.nle_fr_levelfiles_blob.restype = ctypes.c_void_p
        lib.nle_fr_levelfiles_blob.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_long)
        ]
        lib.nle_fr_levelfiles_load.restype = ctypes.c_int
        lib.nle_fr_levelfiles_load.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long
        ]

        # Portable level blob save / load.  nle_save_level mallocs a blob (caller
        # frees via nle_free_blob).  nle_load_level is TWO-PHASE: it mutates state
        # but does not render — the caller must step once (ctrl-R) to redraw.
        lib.nle_save_level.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_long)]
        lib.nle_save_level.restype = ctypes.c_void_p
        lib.nle_free_blob.argtypes = [ctypes.c_void_p]
        lib.nle_free_blob.restype = None
        lib.nle_load_level.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]
        lib.nle_load_level.restype = ctypes.c_int

        # Portable hero blob save / load (u + inventory + attributes).  Same
        # malloc/free + TWO-PHASE contract as the level blob.  Hard ordering:
        # load the LEVEL first (nle_load_level), THEN the player
        # (nle_load_player), then render once.
        lib.nle_save_player.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_long)]
        lib.nle_save_player.restype = ctypes.c_void_p
        lib.nle_load_player.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]
        lib.nle_load_player.restype = ctypes.c_int

        # Difficulty knob catalog.  nle_get_tune returns nle_tune_t*, which is a
        # flat block of doubles indexed by the name table — so get/set are fully
        # generic (a new knob in the engine appears with no binding change).
        lib.nle_tune_count.restype = ctypes.c_int
        lib.nle_tune_count.argtypes = []
        lib.nle_tune_name.restype = ctypes.c_char_p
        lib.nle_tune_name.argtypes = [ctypes.c_int]
        lib.nle_get_tune.restype = ctypes.POINTER(ctypes.c_double)
        lib.nle_get_tune.argtypes = [ctypes.c_void_p]

        # Secure single-field state mutation + deferred dungeon-level jump.
        # nle_set_state writes a whitelisted blstats-backed field (rc!=0 on an
        # unknown name).  nle_goto_depth is TWO-PHASE like nle_load_level: it only
        # schedules the jump; the actual level change happens on the next step.
        lib.nle_set_state.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long]
        lib.nle_set_state.restype = ctypes.c_int
        lib.nle_goto_depth.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.nle_goto_depth.restype = ctypes.c_int

        # Seat the hero on the down(1)/up(0) staircase of the current level, and
        # raise the hero N experience levels with real HP/stat gains.  Both are
        # TWO-PHASE like nle_goto_depth: the caller must step once to render.
        lib.nle_seat_on_stair.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.nle_seat_on_stair.restype = ctypes.c_int
        lib.nle_level_up.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.nle_level_up.restype = ctypes.c_int

        # Curriculum traversal: cross-branch goto + dungeon-table query.
        # nle_goto_abs is TWO-PHASE like nle_goto_depth (caller steps once).
        lib.nle_goto_abs.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.nle_goto_abs.restype = ctypes.c_int
        lib.nle_hero_on_stair.argtypes = [ctypes.c_void_p]
        lib.nle_hero_on_stair.restype = ctypes.c_int
        # Curriculum invocation ritual: grant the pre-primed artifact kit into the
        # hero's pack, and read the vibrating-square coords. Neither is two-phase
        # (grant mutates invent in place; pos is a pure read) — see wrappers below.
        lib.nle_grant_invocation_kit.argtypes = [ctypes.c_void_p]
        lib.nle_grant_invocation_kit.restype = ctypes.c_int
        lib.nle_invocation_pos.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.nle_invocation_pos.restype = ctypes.c_int
        lib.nle_seat_on_invocation_square.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.nle_seat_on_invocation_square.restype = ctypes.c_int
        lib.nle_num_dungeons.argtypes = [ctypes.c_void_p]
        lib.nle_num_dungeons.restype = ctypes.c_int
        lib.nle_dungeon_info.argtypes = [
            ctypes.c_void_p,                # nle_ctx_t*
            ctypes.c_int,                   # idx
            ctypes.c_char_p,                # name_out
            ctypes.c_int,                   # name_cap
            ctypes.POINTER(ctypes.c_int),   # depth_start_out
            ctypes.POINTER(ctypes.c_int),   # num_dunlevs_out
        ]
        lib.nle_dungeon_info.restype = ctypes.c_int

        # Reseed the gameplay/display RNGs.  Used after restore() to make
        # branches diverge: the 4th arg `reseed` is a C boolean (1 byte).
        lib.nle_set_seed.argtypes = [
            ctypes.c_void_p,    # nle_ctx_t*
            ctypes.c_ulong,     # core
            ctypes.c_ulong,     # disp
            ctypes.c_char,      # reseed (boolean, 1 byte)
        ]
        lib.nle_set_seed.restype = None

        # Debug: dump the per-env arena memory map (named buffers, fmon/fobj
        # chains, and the monster grid with fmon-membership) to a file. A
        # diagnostic tool for arena-reuse / dangling-pointer investigations.
        lib.nle_dbg_memmap.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.nle_dbg_memmap.restype = None

    def _build_dat_path(self) -> Path:
        """Return the path to the pre-built dat directory (NetHack data files).

        Prefers the dev ``third_party/NetHack/src/build/dat`` tree; falls back to
        a copy bundled next to this package (``nethack_core/dat``) for the Hub
        wheel, where no fork source is present. Both are produced by the build
        script / tools/bundle_for_hub.py.
        """
        build = _engine_build_dir()
        if build is not None and (build / "dat").exists():
            return build / "dat"
        bundled = Path(__file__).resolve().parent / "dat"
        if bundled.exists():
            return bundled
        raise EngineNotBuilt(
            "NetHack dat directory not found; build with nethack_core/build_engine.sh "
            "or bundle it via tools/bundle_for_hub.py"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __del__(self) -> None:
        try:
            self.end()
        except Exception:
            pass

    def start(self, core: int, disp: int, tune: dict = None,
              character: str = None) -> "RawEngine":
        """Start a new game.  Returns self so callers can chain property reads.

        Safe to call while a game is already active — tears down the previous
        context first so no C context or temp hackdir is leaked.

        ``tune`` optionally overrides difficulty knobs BEFORE the starting level
        is generated, so generation-time knobs (e.g. room_density) take effect on
        the starting floor.  Live knobs may also be set this way or via set_tune()
        after start().  Unknown knob names raise KeyError.

        ``character`` selects the role/race/alignment/gender via NetHack's
        player-name suffix (e.g. "Val-hum-neu-fem" for a female neutral human
        Valkyrie). If None, the historical default (Monk) is used.
        """
        # Tear down any prior game's C context (the writable hackdir is reused).
        self._teardown_ctx()

        # Provide a fresh, tiny writable hackdir for this game's game-state files.
        # The large read-only data is read directly from the shared source dat
        # (settings.datadir below), so reset no longer copies the ~3.7MB dat tree
        # — historically the dominant (~29ms) reset cost.
        self._ensure_hackdir()

        # Build settings.
        settings = NleSettings()
        settings.hackdir = self._hackdir.encode()
        # Shared read-only data dir: the engine reads nhdat/data/oracles/config
        # from here (never writes), so many envs share it with zero per-env copy.
        settings.datadir = str(self._build_dat_path()).encode()
        settings.scoreprefix = b""
        char = character if character is not None else _DEFAULT_CHARACTER
        options_bytes = (_OPTIONS_BASE + ",name:Agent-" + char).encode()
        settings.options = options_bytes
        settings.wizkit = b""
        settings.spawn_monsters = 1
        settings.ttyrecname = b""

        # Apply start-time difficulty-knob overrides (generation knobs must be
        # set before the starting level is built). tune_n == 0 => vanilla.
        if tune:
            names = tune_names(self._lib)
            index = {name: i for i, name in enumerate(names)}
            if len(tune) > NLE_TUNE_MAX:
                raise ValueError(f"at most {NLE_TUNE_MAX} tune overrides at start")
            for j, (key, value) in enumerate(tune.items()):
                if key not in index:
                    raise KeyError(f"unknown tune knob {key!r}; known: {names}")
                settings.tune_idx[j] = index[key]
                settings.tune_val[j] = float(value)
            settings.tune_n = len(tune)

        # Build seeds.
        seeds = NleSeedsInit()
        seeds.seeds[0] = core
        seeds.seeds[1] = disp
        seeds.reseed = b"\x00"  # False — deterministic

        self._obs.action = 0
        self._ctx = self._lib.nle_start(
            ctypes.byref(self._obs),
            None,
            ctypes.byref(seeds),
            ctypes.byref(settings),
        )
        return self

    def step(self, action: int) -> "RawEngine":
        """Send an action to the engine.  Returns self."""
        self._obs.action = int(action)
        self._lib.nle_step(self._ctx, ctypes.byref(self._obs))
        return self

    def _teardown_ctx(self) -> None:
        """Free the current game's C context and snapshot handles (keep hackdir).

        Snapshot handles become invalid once the ctx they restore into is gone,
        so they are freed here; they are self-contained copies, so destroy is
        independent of the ctx and ordering relative to nle_end does not matter.
        """
        for snap in list(self._snapshots):
            self._lib.nle_fr_destroy(snap)
        self._snapshots.clear()
        if self._ctx is not None:
            self._lib.nle_end(self._ctx)
            self._ctx = None

    def _ensure_hackdir(self) -> None:
        """Provide a fresh, empty-but-seeded writable hackdir for a new game.

        The directory is created once per engine and reused: on each reset its
        prior game-state files (level/save/bones/lock plus the score/log files)
        are removed and the empty writable templates re-seeded. It holds only a
        handful of small files — the bulk read-only data is read from the shared
        datadir — so this is sub-millisecond regardless of how the prior game
        grew. The seeded empty templates reproduce exactly what the old
        per-reset copytree placed here.
        """
        if self._hackdir is None:
            self._hackdir = tempfile.mkdtemp(prefix="nethack_hackdir_")
        else:
            for name in os.listdir(self._hackdir):
                p = os.path.join(self._hackdir, name)
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    os.unlink(p)
        for tmpl in self._writable_templates:
            open(os.path.join(self._hackdir, tmpl), "wb").close()

    def end(self) -> None:
        """Tear down the current game context and remove the temp hackdir.

        Called on close()/__del__. Within a single engine, start() reuses the
        hackdir across games (see _ensure_hackdir) and only this final teardown
        removes it.
        """
        self._teardown_ctx()
        if self._hackdir is not None:
            shutil.rmtree(self._hackdir, ignore_errors=True)
            self._hackdir = None

    # ------------------------------------------------------------------
    # In-memory snapshot / restore / branch
    # ------------------------------------------------------------------

    def snapshot(self):
        """Capture the full live game state and return an opaque handle.

        The handle wraps a self-contained C-side copy of (ctx + coroutine stack
        + arena + rl-port display mirror + EVERY on-disk dungeon-level file)
        taken at the moment of the call — not merely the initial state.  It can
        be restored to return the engine to this exact point, and the same
        handle can be restored repeatedly to branch alternate action lines with
        byte-exact fidelity each time.

        The level-file bundle is all-or-nothing: if any level file cannot be
        read, the whole snapshot fails rather than returning a partial bundle,
        because a partial bundle restores a heap that believes in files that
        are not on disk — which NetHack reports as a death, not an error
        (see :class:`CheckpointIntegrityError`).

        The handle is bound to THIS RawEngine instance: restoring it into a
        different instance's ctx is undefined, so restore() rejects handles it
        did not create.  Outstanding handles are freed by end()/__del__ if the
        caller does not free_snapshot() them first.

        Raises RuntimeError if no game is active or the C call fails.
        """
        if self._ctx is None:
            raise RuntimeError("snapshot() requires an active game; call start() first")
        handle = self._lib.nle_fr_snapshot(self._ctx)
        if not handle:
            raise RuntimeError("nle_fr_snapshot failed (returned NULL)")
        self._snapshots.add(handle)
        return handle

    def restore(self, handle) -> "RawEngine":
        """Restore the engine to the state captured by ``handle``.  Returns self.

        ``handle`` must be one this instance created (same-instance binding); a
        foreign or unknown handle raises ValueError rather than corrupting
        memory.

        Observation-buffer refill behavior (verified empirically): restore
        rewrites the engine's internal state (ctx + coroutine stack + arena) but
        does NOT itself repopulate the binding's numpy observation buffers — the
        buffers reflect the restored state only after the next step() refills
        them.  In practice callers step() after restore (to branch), so the
        buffers are correct by the time they are read.

        Fidelity (verified empirically against fresh-engine ground truth): the
        snapshot is COMPLETE.  Restore + any action line reproduces a from-scratch
        run byte-for-byte on glyphs, chars, colors AND blstats — including on
        repeated restores from the same handle after an abandoned branch explored
        a different part of the map.  (The fork allocates the engine's per-env
        heap buffers in the arena, and the C snapshot captures the rl-port display
        mirror — which lives outside the arena — alongside ctx/stack/arena, so no
        display residue leaks across branches.)

        Raises RuntimeError if no game is active.
        """
        if self._ctx is None:
            raise RuntimeError("restore() requires an active game; call start() first")
        if handle not in self._snapshots:
            raise ValueError(
                "restore() handle was not created by this RawEngine instance; "
                "a snapshot is bound to the ctx that created it"
            )
        self._lib.nle_fr_restore(self._ctx, handle)
        # Level-file totality guard.  restore() rewrites every level file the
        # snapshot bundled and deletes the stale ones, so the hackdir's level
        # set becomes exactly the snapshot-time set.  A nonzero rc means a write
        # or unlink failed, which leaves the (heap, disk) pair divergent — the
        # exact state that turns the next stair transition into a done(TRICKED)
        # fake death.  Fail loudly here rather than 40 turns later as a "death".
        if self._lib.nle_fr_last_restore_rc() != 0:
            raise CheckpointIntegrityError(
                "restore() could not rewrite the snapshot's level files; the "
                "restored game's dungeon is inconsistent with the hackdir"
            )
        return self

    def snapshot_levelfile_count(self, handle) -> int:
        """How many on-disk dungeon-level files ``handle`` bundled.

        Zero after the game has made any level transition is the signature of
        the pre-2026-08 lock-base bug (the bundler matched against ``s_lock``,
        which ``set_levelfile_name`` had already rewritten to ``<base>.<lev>``,
        so nothing ever matched).  Tests assert this is nonzero.
        """
        return int(self._lib.nle_fr_levelfile_count(handle))

    def missing_level_files(self) -> int:
        """Ledgers the live heap marks ``LFILE_EXISTS`` with no file on disk.

        Each one is a latent ``done(TRICKED)`` waiting for the hero to use that
        level's stairs.  Must be 0 for a healthy game; ``-1`` if the engine
        could not compute it (no hackdir).
        """
        if self._ctx is None:
            raise RuntimeError("missing_level_files() requires an active game")
        return int(self._lib.nle_fr_missing_levelfiles(self._ctx))

    def check_level_file_integrity(self, where: str = "") -> None:
        """Raise :class:`CheckpointIntegrityError` if any level file is missing."""
        n = self.missing_level_files()
        if n > 0:
            raise CheckpointIntegrityError(
                f"{n} dungeon level(s) are flagged as existing but their level "
                f"files are missing from {self._hackdir!r}"
                + (f" ({where})" if where else "")
                + "; the next stair transition would be reported as a death"
            )

    def save_levelfiles(self) -> bytes:
        """Serialize the hackdir's whole dungeon-level-file set to a blob.

        This is the part of a DISK checkpoint that ``snapshot()`` cannot give
        you: the fast-reset handle is a live-process construct (coroutine stack
        image + arena bytes + display mirror), whereas this is plain bytes.

        It is mandatory for cross-process resume, not an optimization:
        ``save_player()`` runs ``savegamestate`` -> ``save_dungeon``
        (dungeon.c:175), which serializes the entire ``level_info[]`` array, so
        the resumed heap believes in every level the original game visited.
        Resuming into an empty hackdir without these files reproduces the
        TRICKED fake death on the first stair transition.
        """
        if self._ctx is None:
            raise RuntimeError("save_levelfiles() requires an active game")
        n = ctypes.c_long(0)
        ptr = self._lib.nle_fr_levelfiles_blob(self._ctx, ctypes.byref(n))
        if not ptr or n.value <= 0:
            raise RuntimeError("nle_fr_levelfiles_blob failed")
        try:
            return ctypes.string_at(ptr, n.value)
        finally:
            self._lib.nle_free_blob(ptr)

    def load_levelfiles(self, blob: bytes) -> "RawEngine":
        """Install a ``save_levelfiles()`` blob into this game's hackdir.

        Writes every file in the blob and deletes any other level file present,
        so the hackdir's level set becomes exactly the blob's.  Returns self.
        """
        if self._ctx is None:
            raise RuntimeError("load_levelfiles() requires an active game")
        buf = ctypes.create_string_buffer(blob, len(blob))
        rc = self._lib.nle_fr_levelfiles_load(self._ctx, buf, len(blob))
        if rc != 0:
            raise CheckpointIntegrityError(
                f"nle_fr_levelfiles_load failed (rc={rc}); the checkpoint's "
                "dungeon level files could not be installed"
            )
        return self

    def free_snapshot(self, handle) -> None:
        """Explicitly free a snapshot handle created by this instance.

        No-op if the handle is unknown (e.g. already freed)."""
        if handle in self._snapshots:
            self._lib.nle_fr_destroy(handle)
            self._snapshots.discard(handle)

    def reseed(self, core: int, disp: int) -> "RawEngine":
        """Reseed the RNG (e.g. after restore) so future random-chance events
        diverge.  Returns self.

        Passing reseed=False to the underlying nle_set_seed forces a fresh
        ISAAC64 reseed from (core, disp) rather than replaying a saved stream,
        so two restores of the same snapshot followed by different (core, disp)
        seeds produce divergent continuations (verified by the branch spike).
        """
        if self._ctx is None:
            raise RuntimeError("reseed() requires an active game; call start() first")
        self._lib.nle_set_seed(
            self._ctx, ctypes.c_ulong(core), ctypes.c_ulong(disp), b"\x00"
        )
        return self

    def memmap(self, path: str) -> str:
        """Dump the per-env arena memory map to ``path`` (debug tool).

        Writes: arena base/used/cap; every named per-env buffer by arena offset;
        the live monster (fmon) and object (fobj) chains; and the monster grid
        (level.monsters[x][y]) with fmon-membership + data-validity per cell.
        Lets you classify any arena pointer and spot dangling/stale pointers
        (grid entries not in fmon, monsters with out-of-range ``data``). Useful
        for arena-reuse / snapshot corruption debugging. Returns ``path``.
        """
        if self._ctx is None:
            raise RuntimeError("memmap() requires an active game; call start() first")
        self._lib.nle_dbg_memmap(self._ctx, str(path).encode())
        return path

    # ------------------------------------------------------------------
    # Portable level blob save / load
    # ------------------------------------------------------------------

    def save_level(self) -> bytes:
        """Serialize the current dungeon level to a portable blob.

        Unlike snapshot(), which captures the full live game and is bound to this
        instance's ctx, the blob is a self-contained level description that can be
        loaded into a fresh game on a different floor.

        Raises RuntimeError if no game is active or the C call fails.
        """
        if self._ctx is None:
            raise RuntimeError("save_level() requires an active game; call start() first")
        n = ctypes.c_long(0)
        ptr = self._lib.nle_save_level(self._ctx, ctypes.byref(n))
        if not ptr or n.value <= 0:
            raise RuntimeError("nle_save_level failed")
        try:
            return ctypes.string_at(ptr, n.value)
        finally:
            self._lib.nle_free_blob(ptr)

    def load_level(self, blob: bytes) -> "RawEngine":
        """Load a level blob as the current level.  Returns self.

        Two-phase: the C call mutates state but does not render, so we step once
        (ctrl-R = action 18, a no-move redraw) to re-render the new level.

        Raises RuntimeError if no game is active or the C call fails.
        """
        if self._ctx is None:
            raise RuntimeError("load_level() requires an active game; call start() first")
        buf = ctypes.create_string_buffer(blob, len(blob))
        rc = self._lib.nle_load_level(self._ctx, buf, len(blob))
        if rc != 0:
            raise RuntimeError(f"nle_load_level failed (rc={rc})")
        self.step(18)  # ctrl-R redraw inside the coroutine
        return self

    def load_level_raw(self, blob: bytes) -> "RawEngine":
        """Install a level blob WITHOUT rendering (caller renders later).

        Used by resume to do level -> player -> ONE render.  Returns self.

        Raises RuntimeError if no game is active or the C call fails.
        """
        if self._ctx is None:
            raise RuntimeError("load_level_raw() requires an active game; call start() first")
        buf = ctypes.create_string_buffer(blob, len(blob))
        rc = self._lib.nle_load_level(self._ctx, buf, len(blob))
        if rc != 0:
            raise RuntimeError(f"nle_load_level failed (rc={rc})")
        return self

    def save_player(self) -> bytes:
        """Serialize the hero (u + inventory + attributes) to a blob.  Returns bytes.

        Like save_level, the blob is self-contained and the C call mallocs it
        (we free via nle_free_blob).

        Raises RuntimeError if no game is active or the C call fails.
        """
        if self._ctx is None:
            raise RuntimeError("save_player() requires an active game; call start() first")
        n = ctypes.c_long(0)
        ptr = self._lib.nle_save_player(self._ctx, ctypes.byref(n))
        if not ptr or n.value <= 0:
            raise RuntimeError("nle_save_player failed")
        try:
            return ctypes.string_at(ptr, n.value)
        finally:
            self._lib.nle_free_blob(ptr)

    def load_player_raw(self, blob: bytes) -> "RawEngine":
        """Restore the hero onto the already-loaded level WITHOUT rendering.

        Two-phase: the C call mutates state but does not render.  The hard
        ordering contract is LEVEL first (load_level_raw), THEN player
        (load_player_raw), then a single step(18) render.  Returns self.

        Raises RuntimeError if no game is active or the C call fails.
        """
        if self._ctx is None:
            raise RuntimeError("load_player_raw() requires an active game; call start() first")
        buf = ctypes.create_string_buffer(blob, len(blob))
        rc = self._lib.nle_load_player(self._ctx, buf, len(blob))
        if rc != 0:
            raise RuntimeError(f"nle_load_player failed (rc={rc})")
        return self

    # ------------------------------------------------------------------
    # Secure state mutation (single field) + deferred dungeon-level jump
    # ------------------------------------------------------------------

    def set_state(self, field: str, value: int) -> "RawEngine":
        """Set a single whitelisted state field on the engine.  Returns self.

        The C side only accepts a fixed whitelist of field names
        (``hp``/``max_hp``/``gold``/``xp_level``/``hunger``); an unknown name
        raises KeyError rather than writing arbitrary memory.  Bounds checking
        is the caller's responsibility (EngineEnv.modify enforces it).

        Raises RuntimeError if no game is active.
        """
        if self._ctx is None:
            raise RuntimeError("set_state() requires an active game; call start() first")
        rc = self._lib.nle_set_state(self._ctx, field.encode(), int(value))
        if rc != 0:
            raise KeyError(f"unknown state field {field!r}")
        return self

    def goto_depth(self, n: int) -> "RawEngine":
        """Jump to dungeon level ``n``.  Returns self.

        Two-phase like load_level: the C call only schedules the jump (the
        engine runs deferred_goto() after the next rhack), so we step once to
        process it.  A wait step ('.') runs rhack and triggers the deferred
        jump; the wait may consume a game turn, which is acceptable for v1.

        Raises ValueError if ``n`` is out of range, RuntimeError if no game.
        """
        if self._ctx is None:
            raise RuntimeError("goto_depth() requires an active game; call start() first")
        rc = self._lib.nle_goto_depth(self._ctx, int(n))
        if rc != 0:
            raise ValueError(f"goto_depth({n}) out of range")
        self.step(ord("."))  # process the deferred goto (a wait step runs rhack)
        return self

    def seat_on_stair(self, down: bool = True) -> "RawEngine":
        """Seat the hero on the down(True)/up(False) staircase of this level.

        Two-phase like goto_depth: the C call schedules the placement and we
        step once (a wait '.') to render it.  A nonzero return just means the
        current level has no such staircase, which is a no-op rather than an
        error.  Returns self.
        """
        rc = self._lib.nle_seat_on_stair(self._ctx, 1 if down else 0)
        self.step(ord("."))  # render the two-phase change
        return self  # rc nonzero just means no such stair (no-op); not an error

    def level_up(self, n: int = 1) -> "RawEngine":
        """Raise the hero ``n`` experience levels with real HP/stat gains.

        Two-phase: after the C call we issue a ctrl-R redraw (action 18) to
        refresh blstats without consuming a game turn.  Returns self.

        Raises RuntimeError if the engine reports failure.
        """
        if self._lib.nle_level_up(self._ctx, int(n)) != 0:
            raise RuntimeError("nle_level_up failed")
        self.step(18)  # ctrl-R redraw to refresh blstats without a turn
        return self

    def goto_abs(self, dnum: int, dlevel: int) -> "RawEngine":
        """Jump to an arbitrary ``(dnum, dlevel)``, across dungeon branches.

        Unlike :meth:`goto_depth` (which is confined to the current branch),
        this can reach Gehennom or the Elemental Planes.  Two-phase like
        goto_depth: the C call schedules the cross-branch goto_level and we step
        once (a wait '.') to process it; the destination level is generated on
        demand if unvisited.

        Raises ValueError if ``(dnum, dlevel)`` is out of range, RuntimeError if
        no game is active.
        """
        if self._ctx is None:
            raise RuntimeError("goto_abs() requires an active game; call start() first")
        rc = self._lib.nle_goto_abs(self._ctx, int(dnum), int(dlevel))
        if rc != 0:
            raise ValueError(f"goto_abs(dnum={dnum}, dlevel={dlevel}) out of range")
        self.step(ord("."))  # process the deferred cross-branch goto
        return self

    def hero_on_stair(self) -> int:
        """+1 if the hero stands on a down stair, -1 on an up stair, else 0.

        Lets a curriculum require genuine stair navigation before a boundary
        jump (the jump fires only on a real stair use, never a teleport).
        """
        if self._ctx is None:
            raise RuntimeError("hero_on_stair() requires an active game; call start() first")
        return int(self._lib.nle_hero_on_stair(self._ctx))

    def grant_invocation_kit(self) -> "RawEngine":
        """Drop the pre-primed invocation kit (lit 7-candle Candelabrum, charged
        Bell of Opening, Book of the Dead — all uncursed) into the hero's pack.

        Used by the primitives curriculum at the DoD3->Gehennom jump so the agent
        can perform the invocation ritual (the only descent from the Invocation
        level to Moloch's Sanctum). Unlike the goto_* hooks this is NOT two-phase
        — it mutates ``invent`` in place — but we send one ctrl-R (redraw, no game
        turn) so the inventory/blstats re-render for the caller.
        """
        if self._ctx is None:
            raise RuntimeError("grant_invocation_kit() requires an active game; call start() first")
        self._lib.nle_grant_invocation_kit(self._ctx)
        self.step(18)  # ctrl-R: redraw only, consumes no game turn
        return self

    def invocation_pos(self):
        """(x, y) of the vibrating square on the Invocation level, else None.

        x is the column, y the row (same convention as glyphs/chars indexing and
        the move_to target), so callers can navigate the hero straight onto it.
        """
        if self._ctx is None:
            raise RuntimeError("invocation_pos() requires an active game; call start() first")
        x = ctypes.c_int(0)
        y = ctypes.c_int(0)
        rc = self._lib.nle_invocation_pos(self._ctx, ctypes.byref(x), ctypes.byref(y))
        if rc != 0:
            return None
        # The C side returns NetHack game coords (u.ux system, 1-based columns).
        # The observation arrays (chars/glyphs) and blstats drop game column 0, so
        # array_x == game_x - 1 (array_y == game_y). Return in the array/move_to
        # convention the rest of the harness uses. Verified: seating the hero on
        # game (7,16) reports blstats (6,16).
        return (int(x.value) - 1, int(y.value))

    def seat_on_invocation_square(self, adjacent: bool = True) -> "RawEngine":
        """Stage the hero at the vibrating (invocation) square. adjacent=True (the
        default) lands the hero on an accessible tile NEXT to the square so the
        agent steps onto it honestly; adjacent=False lands on the square. Two-phase
        like the other seat/goto hooks: we step one ctrl-R redraw so the new hero
        position renders. No-op-safe off the Invocation level."""
        if self._ctx is None:
            raise RuntimeError("seat_on_invocation_square() requires an active game; call start() first")
        self._lib.nle_seat_on_invocation_square(self._ctx, 1 if adjacent else 0)
        self.step(18)  # ctrl-R redraw
        return self

    def dungeon_table(self) -> list:
        """Return the dungeon-branch layout as a list of dicts.

        Each entry is ``{"dnum", "name", "depth_start", "num_dunlevs"}``.  Lets
        callers map an absolute "Dlvl N" to a concrete ``(dnum, dlevel)`` and
        locate Gehennom / "The Elemental Planes" by name.

        Requires an active game (the dungeon graph is built at start()).
        """
        if self._ctx is None:
            raise RuntimeError("dungeon_table() requires an active game; call start() first")
        n = self._lib.nle_num_dungeons(self._ctx)
        table = []
        for i in range(n):
            name = ctypes.create_string_buffer(24)
            depth_start = ctypes.c_int(0)
            num_dunlevs = ctypes.c_int(0)
            self._lib.nle_dungeon_info(
                self._ctx, i, name, 24,
                ctypes.byref(depth_start), ctypes.byref(num_dunlevs),
            )
            table.append({
                "dnum": i,
                "name": name.value.decode(errors="replace"),
                "depth_start": depth_start.value,
                "num_dunlevs": num_dunlevs.value,
            })
        return table

    # ------------------------------------------------------------------
    # Difficulty knobs (nle_tune_t)
    # ------------------------------------------------------------------

    def tune_catalog(self) -> list:
        """Return the ordered list of difficulty-knob names the engine exposes."""
        return list(tune_names(self._lib))

    def get_tune(self) -> dict:
        """Return the current difficulty knobs as a {name: value} dict.

        Requires an active game (the knob block lives on the engine ctx).
        """
        if self._ctx is None:
            raise RuntimeError("get_tune() requires an active game; call start() first")
        names = tune_names(self._lib)
        ptr = self._lib.nle_get_tune(self._ctx)
        return {name: float(ptr[i]) for i, name in enumerate(names)}

    def set_tune(self, **knobs) -> "RawEngine":
        """Set one or more difficulty knobs by name.  Returns self.

        Live (Layer 3) knobs take effect on the next step().  Unknown knob names
        raise KeyError.  Values are coerced to float (bools/ints ride as doubles).
        """
        if self._ctx is None:
            raise RuntimeError("set_tune() requires an active game; call start() first")
        names = tune_names(self._lib)
        index = {name: i for i, name in enumerate(names)}
        ptr = self._lib.nle_get_tune(self._ctx)
        for key, value in knobs.items():
            if key not in index:
                raise KeyError(
                    f"unknown tune knob {key!r}; known knobs: {names}"
                )
            ptr[index[key]] = float(value)
        return self

    # ------------------------------------------------------------------
    # Observation properties (reshaped views — reflect in-place C updates)
    # ------------------------------------------------------------------

    @property
    def glyphs(self) -> np.ndarray:
        return self._glyphs.reshape(ROWNO, COLNO - 1)

    @property
    def chars(self) -> np.ndarray:
        return self._chars.reshape(ROWNO, COLNO - 1)

    @property
    def colors(self) -> np.ndarray:
        return self._colors.reshape(ROWNO, COLNO - 1)

    @property
    def tty_chars(self) -> np.ndarray:
        return self._tty_chars.reshape(NLE_TERM_LI, NLE_TERM_CO)

    @property
    def tty_colors(self) -> np.ndarray:
        return self._tty_colors.reshape(NLE_TERM_LI, NLE_TERM_CO)

    @property
    def blstats(self) -> np.ndarray:
        return self._blstats  # already (27,)

    @property
    def message(self) -> np.ndarray:
        return self._message  # already (256,)

    @property
    def done(self) -> bool:
        """True once the current game has ended (death, escape, quit, ...)."""
        return bool(self._obs.done)

    @property
    def how_done(self) -> int:
        """The engine's how_done code for the ended game (valid when done)."""
        return int(self._obs.how_done)

    @property
    def in_normal_game(self) -> bool:
        """True while the engine is in normal play (not a menu/prompt screen)."""
        return bool(self._obs.in_normal_game[0]) if isinstance(
            self._obs.in_normal_game, bytes
        ) else bool(self._obs.in_normal_game)

    # ------------------------------------------------------------------
    # Snapshot builder
    # ------------------------------------------------------------------

    def to_core_observation(self):
        """Return a CoreObservation snapshot of the current engine state.

        Each buffer is copied so that subsequent step() calls do not mutate
        previously captured observations (callers may store trajectory frames).

        Deferred import: env.py will import _engine in a later task, so a
        module-level import here would create a cycle.
        """
        from .env import CoreObservation
        return CoreObservation(
            tty_chars=self._tty_chars.reshape(NLE_TERM_LI, NLE_TERM_CO).copy(),
            tty_colors=self._tty_colors.reshape(NLE_TERM_LI, NLE_TERM_CO).copy(),
            tty_cursor=self._tty_cursor.copy(),
            glyphs=self._glyphs.reshape(ROWNO, COLNO - 1).copy(),
            chars=self._chars.reshape(ROWNO, COLNO - 1).copy(),
            colors=self._colors.reshape(ROWNO, COLNO - 1).copy(),
            message=self._message.copy(),
            inv_strs=self._inv_strs.reshape(NLE_INVENTORY_SIZE, NLE_INVENTORY_STR_LENGTH).copy(),
            inv_letters=self._inv_letters.copy(),
            inv_glyphs=self._inv_glyphs.copy(),
            blstats=self._blstats.copy(),
            misc=self._misc.copy(),
        )
