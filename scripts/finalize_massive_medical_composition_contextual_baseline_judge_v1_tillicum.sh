#!/bin/bash
# Permanent single-entry canary or continuation for the contextual judge.

set -euo pipefail
umask 077
ulimit -c 0

mode=${1:-}
[[ $mode == canary || $mode == continuation ]] || {
  echo 'Usage: finalize_..._judge_v1_tillicum.sh canary|continuation EXACT_ACK_FLAGS' >&2
  exit 2
}
shift
[[ -n ${OPENAI_API_KEY:-} ]] || {
  echo 'OPENAI_API_KEY must be loaded for this separately authorized stage.' >&2
  exit 3
}

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-composition-contextual-baseline-judge-v1
output=$root/outputs/massive_medical_composition_contextual_baseline_judge_v1
runner=$repo/scripts/judge_massive_medical_composition_contextual_baselines_split_v1.py
manifest=$output/control/JUDGE_STAGE_MANIFEST.json
log=$output/logs/external_judge_${mode}.log

cd "$repo"
test -z "$(git status --porcelain)"
test ! -e "$log"
module load conda/Miniforge3-25.3.1-3
conda activate "$root/envs/subliminal-mitigate-py311"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/contextual-baseline-judge-v1-pyc
export OPENAI_LOG=off
owner_token=$(python -c 'import secrets; print(secrets.token_hex(32))')

cleanup() {
  code=$?
  unset OPENAI_API_KEY owner_token
  if [[ -f $log ]]; then chmod 0400 "$log"; fi
  exit "$code"
}
trap cleanup EXIT

python "$runner" authorize --manifest "$manifest" --stage "$mode" \
  --owner-token "$owner_token" "$@"

set +e
python "$runner" run --manifest "$manifest" --stage "$mode" \
  --owner-token "$owner_token" >"$log" 2>&1
run_code=$?
set -e
unset OPENAI_API_KEY
chmod 0400 "$log"

if [[ $run_code -ne 0 ]]; then
  failure=$output/control/${mode^^}_FAILURE.json
  started=$output/control/${mode^^}_RUN_STARTED.json
  if [[ -f $failure ]]; then
    python "$runner" audit-failure --manifest "$manifest" --stage "$mode"
    echo "CONTEXTUAL_BASELINE_JUDGE_${mode^^}_TERMINAL_FAILURE_NO_RESTART"
  elif [[ -f $started ]]; then
    # SIGKILL/power loss cannot be caught to seal a failure.  The permanent
    # RUN_STARTED entry is still the terminal no-restart evidence.
    python "$runner" status --manifest "$manifest"
    echo "CONTEXTUAL_BASELINE_JUDGE_${mode^^}_RUN_STARTED_NO_RESTART_OR_SECOND_ENTRY"
  else
    echo "CONTEXTUAL_BASELINE_JUDGE_${mode^^}_PREENTRY_FAILURE_NO_API_CALL" >&2
  fi
  exit "$run_code"
fi

if [[ $mode == canary ]]; then
  python "$runner" audit-canary --manifest "$manifest"
  echo 'CONTEXTUAL_BASELINE_JUDGE_CANARY_COMPLETE_AWAITING_SEPARATE_159_CALL_AUTHORIZATION'
else
  python "$runner" audit-continuation --manifest "$manifest"
  echo 'CONTEXTUAL_BASELINE_JUDGE_CONTINUATION_COMPLETE'
fi
