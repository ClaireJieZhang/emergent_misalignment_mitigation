#!/bin/bash
# Permanent single entry for the explicitly authorized Kalai s=1 judge call.

set -euo pipefail
umask 077
ulimit -c 0

[[ $# -eq 0 ]] || {
  echo 'Usage: finalize_massive_medical_kalai_s1_recovery_one_call_judge_v1_tillicum.sh' >&2
  exit 2
}
[[ -n ${OPENAI_API_KEY:-} ]] || {
  echo 'OPENAI_API_KEY must be loaded for the authorized one-call stage.' >&2
  exit 3
}

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-kalai-s1-batch7-result-recovery-v1
output=$root/outputs/massive_medical_kalai_s1_batch7_result_recovery_v1_kalai_s1_recovery_one_call_judge_v1
runner=$repo/scripts/judge_massive_medical_kalai_s1_recovery_one_call_v1.py
manifest=$output/control/JUDGE_STAGE_MANIFEST.json
log=$output/logs/external_judge_one_call.log

cd "$repo"
test -z "$(git status --porcelain)"
test -s "$manifest"; test ! -L "$manifest"
test ! -e "$log"
module load conda/Miniforge3-25.3.1-3
conda activate "$root/envs/subliminal-mitigate-py311"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-kalai-s1-recovery-one-call-judge-v1-pyc
export OPENAI_LOG=off
owner_token=$(python -c 'import secrets; print(secrets.token_hex(32))')

cleanup() {
  code=$?
  unset OPENAI_API_KEY owner_token
  if [[ -f $log ]]; then chmod 0400 "$log"; fi
  exit "$code"
}
trap cleanup EXIT

python "$runner" authorize --manifest "$manifest" --stage canary \
  --owner-token "$owner_token" \
  --ack-calls 1 \
  --ack-max-cost-usd 0.003072 \
  --ack-total-judge-cap-usd 0.003072 \
  --ack-known-program-actual-usd 0.90125 \
  --ack-retained-prior-exposure-usd 11.52198425 \
  --ack-current-conservative-exposure-usd 12.42323425 \
  --ack-conservative-program-max-usd 12.42630625 \
  --ack-program-ceiling-usd 12.5000000 \
  --ack-sdk-retries-zero \
  --ack-no-restart-or-resume \
  --ack-contextual-post-hoc-only \
  --ack-unused-terminal-authority-nonreusable \
  --ack-unused-terminal-authority-not-cost-exposure

set +e
python "$runner" run --manifest "$manifest" --stage canary \
  --owner-token "$owner_token" >"$log" 2>&1
run_code=$?
set -e
unset OPENAI_API_KEY
chmod 0400 "$log"

if [[ $run_code -ne 0 ]]; then
  if [[ -f $output/control/ONE_CALL_FAILURE.json ]]; then
    python "$runner" audit-failure --manifest "$manifest" --stage canary
    echo KALAI_S1_RECOVERY_ONE_CALL_JUDGE_TERMINAL_FAILURE_NO_RESTART
  elif [[ -f $output/control/ONE_CALL_RUN_STARTED.json ]]; then
    python "$runner" status --manifest "$manifest"
    echo KALAI_S1_RECOVERY_ONE_CALL_JUDGE_RUN_STARTED_NO_RESTART_OR_SECOND_ENTRY
  else
    echo KALAI_S1_RECOVERY_ONE_CALL_JUDGE_PREENTRY_FAILURE_NO_API_CALL >&2
  fi
  exit "$run_code"
fi

python "$runner" audit-canary --manifest "$manifest"
python "$runner" status --manifest "$manifest"
echo KALAI_S1_RECOVERY_ONE_CALL_JUDGE_COMPLETE
