#!/usr/bin/env bash
# One-cell smoke: tiny rollout to validate the vf-eval -> Prime -> Gemini path.
# Usage: tools/encoding_eval/_smoke.sh <VARIANT> <OUTDIR> [MAX_TURNS]
set -uo pipefail
VARIANT="${1:?variant}"; OUTDIR="${2:?outdir}"; MT="${3:-4}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
export PYTHONPATH=".:environments/nethack"
export PI_API_KEY="${PI_API_KEY:-$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.prime/config.json')))['api_key'])")}"
mkdir -p "$OUTDIR/trace"
ARGS="{\"tier\":\"full_nle\",\"variant\":\"${VARIANT}\",\"skill_set\":\"netplay\",\"compact_obs\":false,\"max_turns\":${MT},\"character\":\"Val-hum-neu-fem\",\"explicit_seeds\":[0],\"trace_dir\":\"${OUTDIR}/trace\"}"
echo "[smoke] variant=$VARIANT max_turns=$MT out=$OUTDIR"
# NB: no `-p prime` — that overrides the registry and drops the team-billing
# header. Model is resolved from configs/endpoints.toml (prime-team endpoint).
.venv/bin/vf-eval nethack --env-dir-path environments \
  -m google/gemini-3-flash-preview --endpoints-path configs/endpoints.toml \
  -a "$ARGS" -n 1 -r 1 -c 1 --num-workers 1 --max-tokens 2048 \
  --output-dir "$OUTDIR" --disable-tui --verbose
echo "[smoke] vf-eval exit=$?"
