#!/usr/bin/env bash
# Sub-experiment 1b smoke: tiny JSON rollout with a given cell_schema, to
# validate that the enabled per-cell layers appear (and disabled ones don't) in
# the rendered user content. Build + smoke only — NOT an n=16 sweep.
# Usage: tools/encoding_eval/_smoke_1b.sh <CELL_SCHEMA_JSON> <OUTDIR> [MAX_TURNS]
#   CELL_SCHEMA_JSON: a JSON array, e.g. '[]' '["seen"]' '["seen","visited","reach"]'
set -uo pipefail
SCHEMA="${1:?cell_schema JSON array, e.g. [\"seen\"]}"; OUTDIR="${2:?outdir}"; MT="${3:-6}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
export PYTHONPATH=".:environments/nethack"
export PI_API_KEY="${PI_API_KEY:-$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.prime/config.json')))['api_key'])")}"
mkdir -p "$OUTDIR/trace"
ARGS="{\"tier\":\"full_nle\",\"variant\":\"JSON\",\"skill_set\":\"netplay\",\"map_detail\":\"full\",\"cell_schema\":${SCHEMA},\"compact_obs\":false,\"max_turns\":${MT},\"character\":\"Val-hum-neu-fem\",\"explicit_seeds\":[0],\"trace_dir\":\"${OUTDIR}/trace\"}"
echo "[smoke-1b] cell_schema=$SCHEMA max_turns=$MT out=$OUTDIR"
# NB: no `-p prime` — that overrides the registry and drops the team-billing
# header. Model is resolved from configs/endpoints.toml (prime-team endpoint).
.venv/bin/vf-eval nethack --env-dir-path environments \
  -m google/gemini-3-flash-preview --endpoints-path configs/endpoints.toml \
  -a "$ARGS" -n 1 -r 1 -c 1 --num-workers 1 --max-tokens 2048 \
  --output-dir "$OUTDIR" --disable-tui --verbose
echo "[smoke-1b] vf-eval exit=$?"
