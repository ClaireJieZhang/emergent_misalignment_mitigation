#!/bin/bash
# Permanent single-entry canary or continuation for the Kalai s=3 judge.

set -euo pipefail
umask 077
ulimit -c 0

mode=${1:-}
[[ $mode == canary || $mode == continuation ]] || {
  echo 'Usage: finalize_massive_medical_kalai_s3_judge_v1_tillicum.sh canary|continuation EXACT_ACK_FLAGS' >&2
  exit 2
}
shift
[[ -n ${OPENAI_API_KEY:-} ]] || {
  echo 'OPENAI_API_KEY must be loaded for this separately authorized stage.' >&2
  exit 3
}

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-kalai-s3-r20-v2-judge-v1
output=$root/outputs/massive_medical_kalai_s3_r20_v2_kalai_s3_judge_v1
runner=$repo/scripts/judge_massive_medical_kalai_s3_split_v1.py
manifest=$output/control/JUDGE_STAGE_MANIFEST.json
log=$output/logs/external_judge_${mode}.log

cd "$repo"
test -z "$(git status --porcelain)"
test ! -e "$log"
module load conda/Miniforge3-25.3.1-3
conda activate "$root/envs/subliminal-mitigate-py311"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-kalai-s3-judge-v1-pyc
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
    echo "KALAI_S3_JUDGE_${mode^^}_TERMINAL_FAILURE_NO_RESTART"
  elif [[ -f $started ]]; then
    python "$runner" status --manifest "$manifest"
    echo "KALAI_S3_JUDGE_${mode^^}_RUN_STARTED_NO_RESTART_OR_SECOND_ENTRY"
  else
    echo "KALAI_S3_JUDGE_${mode^^}_PREENTRY_FAILURE_NO_API_CALL" >&2
  fi
  exit "$run_code"
fi

if [[ $mode == canary ]]; then
  python "$runner" audit-canary --manifest "$manifest"
  python "$runner" status --manifest "$manifest"
else
  python "$runner" audit-continuation --manifest "$manifest"
  echo KALAI_S3_JUDGE_CONTINUATION_COMPLETE
fi
