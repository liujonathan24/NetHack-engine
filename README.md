# nethack-engine

The **engine** layer (layer 1) of the NetHack RL/eval stack: a fast, deterministic
NetHack substrate built on a **custom struct-based fork** of NetHack driven through
a `ctypes` binding — **not** the `nle` PyPI package. It exposes a clean, typed
observation plus snapshot / restore / branch / tune primitives, and nothing else.

The training/eval **harness** (skills, curriculum, prompts, memory, the `nethack`
Verifiers env) lives in the separate **Hub** repo and depends on this engine as an
external package. This repo never imports the Hub.

## Packages

| Package | What it is |
|---|---|
| `nethack_core` | The engine: wraps the NetHack fork (`third_party/NetHack`), builds `libnethack.so`, and surfaces `NetHackCoreEnv` (nle-gym path), `EngineEnv` (fork engine with snapshot/branch/tune), the map model, glyph classifiers, and reward primitives. `nethack_core/__init__.py` is the one documented import boundary. |
| `nethack_interface` | A thin PySC2-style typed interface over `nethack_core`: `Observation` + `RawAction` stepping via `NetHackInterface`. Pure substrate — no Hub dependency. The typed `Action`/skill layer lives in the Hub (`nethack_harness.interface`). |

## Build

The engine is a `libnethack.so` built from the pinned fork submodule. Toolchain:
`cmake`, `bison`, `flex`, `libbz2-dev` (Linux/gcc — the fork does not build on
macOS/Apple-clang).

```bash
git submodule update --init --recursive
JOBS=8 bash nethack_core/build_engine.sh
# -> third_party/NetHack/src/build/libnethack.so
```

The binding locates the `.so` via `NLE_LIB_PATH`, then the standard build dir,
then a bundled copy next to the module.

## Test

```bash
pytest tests -q          # engine tests (needs the built .so)
```

## Fork hooks

The fork exposes state-injection / navigation hooks used by the curriculum, e.g.
`nle_hero_on_stair` (±1 main stair, ±2 branch stair) and the invocation-ritual
trio `nle_grant_invocation_kit` / `nle_invocation_pos` /
`nle_seat_on_invocation_square`, wrapped as `EngineEnv` methods.

## Docker

`Dockerfile.prime` builds the `.so` and installs the engine for the Prime
Intellect sandbox runtime; it runs the engine tests at build time.
