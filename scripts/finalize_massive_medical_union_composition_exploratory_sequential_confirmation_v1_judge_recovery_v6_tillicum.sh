#!/bin/bash
# Logged, permanent single-entry external judge stages for recovery v6.

set -euo pipefail
umask 077
ulimit -c 0

usage() {
  echo 'Usage: finalize_..._judge_recovery_v6_tillicum.sh canary|continue with the exact frozen acknowledgment flags' >&2
  exit 2
}

mode=${1:-}
common=(
  --ack-verified-program-actual-usd 2.915186
  --ack-consumed-v3-authority-cap-usd 0.75
  --ack-consumed-v4-canary-authority-cap-usd 0.003072
  --ack-v5-canary-actual-usd 0.0001145
  --ack-v5-continuation-authority-cap-usd 0.746928
  --ack-v5-failed-continuation-exposure-cap-usd 0.003072
  --ack-prior-accounted-exposure-usd 0.7562585
  --ack-prior-network-attempts-min 3
  --ack-prior-network-attempts-max 4
  --ack-prior-accepted-judgments 1
  --ack-v5-continuation-authority-consumed-nonreusable
  --ack-v5-failed-request-actual-billing-unknown
  --ack-v5-unattempted-authority-not-cost-exposure
  --ack-v5-unattempted-authority-not-reused
  --ack-prior-authorities-consumed-not-reused
)

case "$mode" in
  canary)
    expected=(
      canary "${common[@]}"
      --ack-max-cost-usd 0.003072
      --ack-new-v6-total-cap-usd 0.746928
      --ack-conservative-program-max-usd 4.4183725
      --ack-program-ceiling-usd 5.0
      --ack-remaining-ceiling-gap-usd 0.5816275
    )
    stage=canary; command=canary; calls=1
    ;;
  continue)
    [[ $# -eq 38 ]] || usage
    canary_actual=${28:-}
    [[ -n $canary_actual ]] || usage
    expected=(
      continue "${common[@]}"
      --ack-canary-actual-estimated-cost-usd "$canary_actual"
      --ack-max-cost-usd 0.743856
      --ack-new-v6-total-cap-usd 0.746928
      --ack-conservative-program-max-usd 4.4183725
      --ack-program-ceiling-usd 5.0
      --ack-remaining-ceiling-gap-usd 0.5816275
    )
    stage=continuation; command=continue; calls=238
    ;;
  *) usage ;;
esac

[[ $# -eq ${#expected[@]} ]] || usage
actual=("$@")
for ((index=0; index < ${#expected[@]}; index++)); do
  [[ ${actual[index]} == "${expected[index]}" ]] || usage
done

[[ -n ${OPENAI_API_KEY:-} ]] || { echo 'OPENAI_API_KEY must be loaded for the separately authorized external stage.' >&2; exit 3; }

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-composition-exploratory-sequential-confirmation-v1-judge-recovery-v6
output=$root/outputs/massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v6
manifest=$output/control/JUDGE_RECOVERY_V6_MANIFEST.json
auditor=$repo/scripts/audit_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v6.py
judge=$repo/scripts/judge_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v6.py
derive=$repo/scripts/derive_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v6_tillicum.sh
log=$output/logs/external_judge_${stage}.log
failure=$output/control/${stage^^}_FAILURE.json
success=$output/control/${stage^^}_SUCCESS.json
lock_owner=$output/control/${stage^^}_LOCK_OWNER.json
env_root=$root/envs/subliminal-mitigate-py311

cd "$repo"
test "$(git branch --show-current)" = claire/capability-quorum-secure-code-composition-exploratory-under5-sequential-v1-judge-recovery-v6
test -z "$(git status --porcelain)"
test ! -e "$log"
module load conda/Miniforge3-25.3.1-3
conda activate "$env_root"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-sequential-judge-recovery-v6-pyc
export OPENAI_LOG=off
owner_token=$(python -c 'import secrets; print(secrets.token_hex(32))')

stage_active=false
completed=false
judge_pid=
terminate_child() {
  if [[ -n $judge_pid ]]; then
    kill -TERM "$judge_pid" 2>/dev/null || true
    wait "$judge_pid" 2>/dev/null || true
    judge_pid=
  fi
}
handle_signal() {
  signal_code=$1
  trap - HUP INT TERM
  terminate_child
  exit "$signal_code"
}
cleanup() {
  cleanup_code=$?
  trap - EXIT HUP INT TERM
  terminate_child
  unset OPENAI_API_KEY
  if [[ $completed != true && $stage_active == true && -f $lock_owner ]]; then
    set +e
    python "$auditor" audit-lock --stage "$stage" \
      --owner-token "$owner_token" >/dev/null 2>&1
    owns_lock=$?
    if [[ $owns_lock -ne 0 ]]; then exit "$cleanup_code"; fi
    failure_code=$cleanup_code
    if [[ $failure_code -eq 0 ]]; then failure_code=1; fi
    if [[ -f $log ]]; then chmod 0400 "$log"; fi
    find "$output/evaluation/medical" -maxdepth 1 -type f \
      -name 'judge_checkpoint.json.*' -exec chmod 0400 {} +
    if [[ -f $output/evaluation/medical/judgments_new.json ]]; then
      chmod 0400 "$output/evaluation/medical/judgments_new.json"
    fi
    if [[ -f $success ]]; then chmod 0400 "$success"; fi
    if [[ -f $failure ]]; then chmod 0400 "$failure"; fi
    if [[ ! -e $success && ! -e $failure ]]; then
      python "$auditor" write-wrapper-failure \
        --stage "$stage" --exit-code "$failure_code" \
        --owner-token "$owner_token"
      wrapper_code=$?
      if [[ $wrapper_code -ne 0 ]]; then
        echo 'ERROR: wrapper failure artifact could not be sealed.' >&2
      fi
    elif [[ -f $failure ]]; then
      python "$auditor" audit-failure --stage "$stage" >/dev/null
      if [[ $? -ne 0 ]]; then
        echo 'ERROR: terminal failure artifact could not be audited.' >&2
      fi
    fi
    cleanup_code=$failure_code
  fi
  exit "$cleanup_code"
}
trap cleanup EXIT
trap 'handle_signal 129' HUP
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM

if [[ $stage == canary ]]; then
  python "$auditor" audit-staged
else
  python "$judge" audit-canary --recovery-manifest "$manifest"
fi
python "$judge" validate-plan --recovery-manifest "$manifest"
python "$auditor" preflight-authorization --stage "$stage" "${@:2}"
stage_active=true
python "$auditor" acquire-lock --stage "$stage" "${@:2}" \
  --owner-token "$owner_token"
python "$auditor" write-authorization --stage "$stage" "${@:2}" \
  --owner-token "$owner_token"
python "$auditor" audit-authorization --stage "$stage"

: > "$log"
chmod 0600 "$log"
set +e
python "$judge" "$command" --recovery-manifest "$manifest" \
  --owner-token "$owner_token" > "$log" 2>&1 &
judge_pid=$!
wait "$judge_pid"
judge_code=$?
judge_pid=
set -e
unset OPENAI_API_KEY
chmod 0400 "$log"

if [[ $judge_code -ne 0 ]]; then
  test -f "$failure"
  chmod 0400 "$failure"
  find "$output/evaluation/medical" -maxdepth 1 -type f \
    -name 'judge_checkpoint.json.*' -exec chmod 0400 {} +
  if [[ -f $output/evaluation/medical/judgments_new.json ]]; then
    chmod 0400 "$output/evaluation/medical/judgments_new.json"
  fi
  python "$auditor" audit-failure --stage "$stage"
  echo "JUDGE_RECOVERY_V6_${stage^^}_TERMINAL_FAILURE"
  echo "Durable permission-restricted log: $log"
  completed=true
  trap - EXIT HUP INT TERM
  exit "$judge_code"
fi

test -f "$success"; test ! -e "$failure"
chmod 0400 "$success"
if [[ $stage == canary ]]; then
  test ! -e "$output/evaluation/medical/judge_checkpoint.json.001"
  chmod 0400 "$output/evaluation/medical/judge_checkpoint.json.002"
  python "$judge" audit-canary --recovery-manifest "$manifest"
  echo 'JUDGE_RECOVERY_V6_CANARY_COMPLETE_AWAITING_SEPARATE_238_CALL_AUTHORIZATION'
  completed=true
  trap - EXIT HUP INT TERM
  exit 0
fi

chmod 0400 "$output/evaluation/medical/judgments_new.json"
find "$output/evaluation/medical" -maxdepth 1 -type f -name 'judge_checkpoint.json.*' -exec chmod 0400 {} +
test ! -e "$output/evaluation/medical/judge_checkpoint.json.001"
test -f "$output/evaluation/medical/judge_checkpoint.json.240"
python "$judge" audit-continuation --recovery-manifest "$manifest"
set +e
"$derive"
final_code=$?
set -e
[[ $final_code -eq 0 || $final_code -eq 2 ]] || exit "$final_code"
echo "JUDGE_RECOVERY_V6_CONTINUATION_COMPLETE calls=$calls scientific_exit=$final_code"
completed=true
trap - EXIT HUP INT TERM
