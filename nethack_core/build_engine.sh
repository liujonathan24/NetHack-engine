#!/usr/bin/env bash
# Build libnethack.so + game data from the pinned NetHack fork submodule.
# Reproducible: configures cmake from the submodule source (deps vendored in
# src/third_party/), so it does not depend on any external checkout.
set -euo pipefail
# Walk up from this script to the repo root (the dir containing third_party/NetHack),
# so the build works regardless of where this script is vendored.
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$DIR"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/third_party/NetHack" ]; do
  ROOT="$(dirname "$ROOT")"
done
if [ ! -d "$ROOT/third_party/NetHack" ]; then
  echo "could not locate third_party/NetHack above $DIR" >&2
  exit 1
fi
SRC="$ROOT/third_party/NetHack/src"
BUILD="$SRC/build"

# (Re)configure from scratch if the cache is missing or was generated from a
# different source tree (e.g. a stale cache committed from another machine).
if [ ! -f "$BUILD/CMakeCache.txt" ] || \
   ! grep -q "^CMAKE_HOME_DIRECTORY:INTERNAL=$SRC\$" "$BUILD/CMakeCache.txt"; then
  rm -rf "$BUILD"
  # The engine needs no Python: skip the fork's pybind11 modules (and the
  # Python-headers requirement that comes with them).
  cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DNLE_BUILD_PYTHON=OFF
fi
cmake --build "$BUILD" --target nethack -j"${JOBS:-8}"

echo "Built: $BUILD/libnethack.so"
ls -l "$BUILD/libnethack.so"
file "$BUILD/libnethack.so"
