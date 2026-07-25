#!/usr/bin/env bash
# Sub-experiment 1c smoke: one tiny rollout per memory arm to validate wiring.
# Build + smoke ONLY — this is NOT the n=16 sweep. See
# docs/experiments/exp1c-memory-arms.md for the real arm configs.
#
# Usage: tools/exp1c_memory/_smoke.sh <arm> <outdir> [max_turns]
#   arm ∈ {journal-only, belief-state, summarize-and-reset, no-memory}
#
# Belief arms use belief_state_interval=3 here (not 25) so a note fires within
# the ~6-turn smoke. The real sweep uses 25 (see the arms doc).
set -uo pipefail
ARM="${1:?arm}"; OUTDIR="${2:?outdir}"; MT="${3:-6}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
export PYTHONPATH=".:environments/nethack"
export PI_API_KEY="${PI_API_KEY:-$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.prime/config.json')))['api_key'])")}"

SPINE='"tier":"full_nle","variant":"B0","interface":"skill","character":"Val-hum-neu-fem","compact_obs":false,"explicit_seeds":[0]'
NETPLAY_NOMEM="move_to,explore_and_descend,attack,throw,descend,search,pickup,engrave_elbereth,pray,eat,quaff,read,kick,wiki_lookup,wiki_search"
MODEL="google/gemini-3-flash-preview"

case "$ARM" in
  journal-only)
    ARMARGS='"skill_set":"netplay","belief_state_interval":0,"summarize_and_reset":false' ;;
  belief-state)
    ARMARGS="\"skill_set\":\"netplay\",\"belief_state_interval\":3,\"sub_lm_model\":\"${MODEL}\",\"summarize_and_reset\":false" ;;
  summarize-and-reset)
    ARMARGS="\"skill_set\":\"netplay\",\"belief_state_interval\":3,\"sub_lm_model\":\"${MODEL}\",\"summarize_and_reset\":true" ;;
  no-memory)
    ARMARGS="\"skill_set\":\"${NETPLAY_NOMEM}\",\"belief_state_interval\":0,\"pin_objective_on_setup\":false,\"history_keep_full\":2" ;;
  *) echo "unknown arm: $ARM" >&2; exit 2 ;;
esac

mkdir -p "$OUTDIR/trace"
ARGS="{${SPINE},${ARMARGS},\"max_turns\":${MT},\"trace_dir\":\"${OUTDIR}/trace\"}"
echo "[1c-smoke] arm=$ARM max_turns=$MT out=$OUTDIR"
echo "[1c-smoke] -a $ARGS"
# NB: no `-p prime` — resolve the model from configs/endpoints.toml (prime-team
# block) so the X-Prime-Team-ID billing header is kept.
.venv/bin/vf-eval nethack --env-dir-path environments \
  -m "$MODEL" --endpoints-path configs/endpoints.toml \
  -a "$ARGS" -n 1 -r 1 -c 1 --num-workers 1 --max-tokens 2048 \
  --output-dir "$OUTDIR" --disable-tui --verbose
echo "[1c-smoke] vf-eval exit=$?"
