#!/bin/bash
# Fresh permanent single entry; keyless readiness precedes the child-only key.

[[ ${BASH_SOURCE[0]} == "$0" ]] || {
  echo 'Execute this script with bash; do not source it.' >&2
  return 3
}
set +x
set -euo pipefail
set -o noclobber
umask 077
ulimit -c 0

[[ $# -eq 0 ]] || {
  echo 'Usage: finalize_massive_medical_kalai_s1_preentry_recovery_one_call_v1_tillicum.sh' >&2
  exit 2
}
[[ ! ${OPENAI_API_KEY+x} ]] || {
  echo 'OPENAI_API_KEY must be absent on entry; unset it in the parent shell first.' >&2
  exit 3
}
[[ ! ${BASH_ENV+x} && ! ${ENV+x} ]] || {
  echo 'BASH_ENV and ENV must be absent for the private terminal wrapper.' >&2
  exit 3
}
[[ -t 0 && -t 1 ]] || {
  echo 'A private interactive terminal is required for the hidden key prompt.' >&2
  exit 3
}

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-kalai-s1-preentry-recovery-v1
output=$root/outputs/massive_medical_kalai_s1_batch7_result_recovery_v1_kalai_s1_preentry_recovery_one_call_v1
runner=$repo/scripts/judge_massive_medical_kalai_s1_preentry_recovery_one_call_v1.py
manifest=$output/control/JUDGE_STAGE_MANIFEST.json
log=$output/logs/external_judge_one_call.log

cd "$repo"
test -z "$(git status --porcelain)"
test -s "$manifest"; test ! -L "$manifest"
test ! -e "$log"; test ! -L "$log"
module load conda/Miniforge3-25.3.1-3
conda activate "$root/envs/subliminal-mitigate-py311"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-kalai-s1-preentry-recovery-one-call-v1-pyc
export OPENAI_LOG=off
unset PYTHONPATH PYTHONHOME
owner_token=$(python -c 'import secrets; print(secrets.token_hex(32))')

cleanup() {
  code=$?
  unset OPENAI_API_KEY owner_token
  if [[ -f $log ]]; then chmod 0400 "$log"; fi
  exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

python "$runner" audit-staged --manifest "$manifest"
python "$runner" keyless-readiness --manifest "$manifest"
echo KALAI_S1_PREENTRY_RECOVERY_READY_FOR_CHILD_ONLY_HIDDEN_KEY
IFS= read -r -s -p 'OpenAI API key: ' OPENAI_API_KEY
printf '\n'
[[ -n $OPENAI_API_KEY ]] || {
  echo 'No API key entered; stopping with zero authorization and zero calls.' >&2
  exit 3
}
export OPENAI_API_KEY

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
  --ack-unused-terminal-authority-not-cost-exposure \
  --ack-fresh-preentry-authority

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
