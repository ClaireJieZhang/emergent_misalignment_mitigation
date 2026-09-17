#!/bin/bash
# Separately authorized, permanent single-entry canary or continuation.

[[ ${BASH_SOURCE[0]} == "$0" ]] || {
  echo 'Execute with bash; do not source this script.' >&2
  return 3
}
set +x
set +a
set -euo pipefail
set -o noclobber
umask 077
ulimit -c 0

mode=${1:-}
[[ $mode == canary || $mode == continuation ]] || {
  echo 'Usage: finalize_massive_medical_ratio_panels_v1_judge_tillicum.sh canary|continuation EXACT_ACK_FLAGS' >&2
  exit 2
}
shift
if [[ $mode == canary ]]; then
  expected_calls=1
  expected_cap=0.003072
else
  expected_calls=624
  expected_cap=1.916928
fi
# Only these exact acknowledgment arguments may be forwarded. In particular,
# a caller cannot append a second stage, manifest, or owner-token argument.
[[ $# -eq 7 && ${1:-} == --ack-calls && ${2:-} == "$expected_calls" &&
   ${3:-} == --ack-max-cost-usd && ${4:-} == "$expected_cap" &&
   ${5:-} == --ack-total-cap-usd && ${6:-} == 1.920000 &&
   ${7:-} == --ack-no-retry-resume ]] || {
  echo 'Use the exact reviewed acknowledgment flags for this stage; no extra flags are accepted.' >&2
  exit 2
}
[[ ! ${OPENAI_API_KEY+x} ]] || {
  echo 'Unset the parent-shell API key; this script uses a child-only hidden prompt.' >&2
  exit 3
}
# Clear a possible inherited export attribute on an unset variable as well.
unset OPENAI_API_KEY
[[ ! ${BASH_ENV+x} && ! ${ENV+x} ]] || {
  echo 'Shell initialization overrides must be absent.' >&2
  exit 3
}
[[ -t 0 && -t 1 ]] || { echo 'A private interactive terminal is required.' >&2; exit 3; }

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-ratio-panel-judge-v1
output=$root/outputs/massive_medical_ratio_panels_v1_judge
python=$root/envs/subliminal-mitigate-py311/bin/python
runner=$repo/scripts/judge_massive_medical_ratio_panels_v1.py
manifest=$output/control/PREP.json
log=$output/logs/external_judge_${mode}.log

cd "$repo"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test -s "$manifest"; test ! -L "$manifest"
test ! -e "$log"; test ! -L "$log"
unset PYTHONPATH PYTHONHOME
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENAI_LOG=off
owner_token=$("$python" -B -c 'import secrets; print(secrets.token_hex(32))')

cleanup() {
  code=$?
  unset OPENAI_API_KEY owner_token
  if [[ -f $log && ! -L $log ]]; then chmod 0400 "$log"; fi
  exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

"$python" -B "$runner" audit-stage --manifest "$manifest"
"$python" -B "$runner" validate-sdk-serialization --manifest "$manifest" --stage "$mode"
echo RATIO_MEDICAL_JUDGE_READY_FOR_CHILD_ONLY_HIDDEN_KEY
IFS= read -r -s -p 'OpenAI API key: ' OPENAI_API_KEY
printf '\n'
[[ -n $OPENAI_API_KEY ]] || { echo 'No key entered; stopping without paid entry.' >&2; exit 3; }

# The hidden key remains an unexported child-shell variable during keyless
# authorization. Only the one permanently claimed run receives it.
"$python" -B "$runner" authorize --manifest "$manifest" --stage "$mode" \
  --owner-token "$owner_token" "$@"
export OPENAI_API_KEY
set +e
"$python" -B "$runner" run --manifest "$manifest" --stage "$mode" \
  --owner-token "$owner_token" >"$log" 2>&1
run_code=$?
set -e
unset OPENAI_API_KEY
chmod 0400 "$log"

if [[ $run_code -ne 0 ]]; then
  "$python" -B "$runner" status --manifest "$manifest"
  echo RATIO_MEDICAL_JUDGE_STOPPED_NO_RETRY_OR_RESUME >&2
  exit "$run_code"
fi
if [[ $mode == canary ]]; then
  "$python" -B "$runner" audit-canary --manifest "$manifest"
  echo RATIO_MEDICAL_JUDGE_CANARY_COMPLETE_AWAITING_SEPARATE_CONTINUATION
else
  "$python" -B "$runner" audit-complete --manifest "$manifest"
  echo RATIO_MEDICAL_JUDGE_COMPLETE_NO_AUTOMATIC_DOWNSTREAM_RELEASE
fi
"$python" -B "$runner" status --manifest "$manifest"
