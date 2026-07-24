#!/usr/bin/env bash
# Launch ONE encoding-sweep cell (Experiment 1, docs/experiments/2026-07-24-exp1-encoding-sweep.md).
# Every cell must run through THIS script so the fixed factors never drift.
#
# Usage:
#   tools/encoding_eval/launch_cell.sh <VARIANT> <OUTDIR> [MAX_TURNS] [MODEL]
# e.g.
#   tools/encoding_eval/launch_cell.sh B0 outputs/encoding_eval/run1/B0
#   tools/encoding_eval/launch_cell.sh JSON outputs/encoding_eval/run1/JSON 150
#
# Fixed factors (identical across cells): netplay skills, uncapped full_nle,
# pinned Valkyrie, 16 pinned seeds (0-15), no compaction. Model defaults to
# Gemini 3 Flash via Prime; pass a 4th arg to swap to Gemini 3 Pro for finals.
set -euo pipefail

VARIANT="${1:?usage: launch_cell.sh <VARIANT> <OUTDIR> [MAX_TURNS] [MODEL]}"
OUTDIR="${2:?usage: launch_cell.sh <VARIANT> <OUTDIR> [MAX_TURNS] [MODEL]}"
MAX_TURNS="${3:-150}"
MODEL="${4:-google/gemini-3-flash-preview}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

export PYTHONPATH=".:environments/nethack"
export PI_API_KEY="${PI_API_KEY:-$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.prime/config.json')))['api_key'])")}"

mkdir -p "$OUTDIR/trace"

# 16 pinned seeds, identical in every cell -> paired comparison.
SEEDS="[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]"

ARGS=$(cat <<JSON
{"tier":"full_nle","variant":"${VARIANT}","skill_set":"netplay","compact_obs":false,"max_turns":${MAX_TURNS},"character":"Val-hum-neu-fem","explicit_seeds":${SEEDS},"trace_dir":"${OUTDIR}/trace"}
JSON
)

echo "[launch_cell] variant=${VARIANT} model=${MODEL} max_turns=${MAX_TURNS} out=${OUTDIR}"
# NB: no `-p prime` — that overrides the registry and drops the X-Prime-Team-ID
# billing header (personal balance is $0). Model resolves from the prime-team
# endpoint in configs/endpoints.toml.
# -c/--num-workers 3: five cells run concurrently in the fan-out, so keep
# per-cell concurrency modest (5*3 = 15-way) to stay under Prime rate limits.
exec .venv/bin/vf-eval nethack --env-dir-path environments \
  -m "${MODEL}" --endpoints-path configs/endpoints.toml \
  -a "${ARGS}" -n 16 -r 1 --max-concurrent 3 --num-workers 3 \
  --max-tokens 2048 --save-results --output-dir "${OUTDIR}" --disable-tui
