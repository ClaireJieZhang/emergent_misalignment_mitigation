#!/bin/bash
# Read-only status/audit for judge recovery v4.

set -euo pipefail
umask 077
ulimit -c 0

[[ $# -eq 0 ]] || { echo 'Usage: status_..._judge_recovery_v4_tillicum.sh' >&2; exit 2; }

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-composition-exploratory-sequential-confirmation-v1-judge-recovery-v4
output=$root/outputs/massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v4
manifest=$output/control/JUDGE_RECOVERY_V4_MANIFEST.json
auditor=$repo/scripts/audit_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v4.py
judge=$repo/scripts/judge_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v4.py
summary=$repo/scripts/summarize_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v4.py

if [[ ! -d $repo || ! -d $output ]]; then
  echo 'JUDGE_RECOVERY_V4_NOT_STAGED'
  exit 0
fi

module load conda/Miniforge3-25.3.1-3
conda activate "$root/envs/subliminal-mitigate-py311"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-sequential-judge-recovery-v4-pyc
cd "$repo"

if [[ -f $output/control/FINAL_RESULT.json ]]; then
  python "$summary" audit-final --recovery-manifest "$manifest"
  echo 'JUDGE_RECOVERY_V4_FINAL_COMPLETE'
elif [[ -f $output/control/CONTINUATION_FAILURE.json ]]; then
  python "$auditor" audit-failure --stage continuation
  echo 'JUDGE_RECOVERY_V4_CONTINUATION_TERMINAL_FAILURE_NO_RESTART'
elif [[ -f $output/control/CONTINUATION_SUCCESS.json ]]; then
  if [[ -f $output/evaluation/medical/judgments_merged.json ]]; then
    echo 'JUDGE_RECOVERY_V4_DERIVED_FINALIZATION_INCOMPLETE'
    exit 3
  fi
  python "$judge" audit-continuation --recovery-manifest "$manifest"
  echo 'JUDGE_RECOVERY_V4_CONTINUATION_COMPLETE_AWAITING_CPU_DERIVATION'
elif [[ -f $output/control/CONTINUATION_AUTHORIZATION.json ]]; then
  python "$auditor" audit-authorization --stage continuation
  echo 'JUDGE_RECOVERY_V4_CONTINUATION_LOCKED_OR_RUNNING'
elif [[ -f $output/control/CONTINUATION_LOCK_OWNER.json ]]; then
  python "$auditor" audit-lock --stage continuation
  echo 'JUDGE_RECOVERY_V4_CONTINUATION_LOCKED_BEFORE_AUTHORIZATION_OR_RUNNING'
elif [[ -f $output/control/CANARY_FAILURE.json ]]; then
  python "$auditor" audit-failure --stage canary
  echo 'JUDGE_RECOVERY_V4_CANARY_TERMINAL_FAILURE_NO_RESTART'
elif [[ -f $output/control/CANARY_SUCCESS.json ]]; then
  python "$judge" audit-canary --recovery-manifest "$manifest"
  echo 'JUDGE_RECOVERY_V4_CANARY_COMPLETE_AWAITING_SEPARATE_239_CALL_AUTHORIZATION'
elif [[ -f $output/control/CANARY_AUTHORIZATION.json ]]; then
  python "$auditor" audit-authorization --stage canary
  echo 'JUDGE_RECOVERY_V4_CANARY_LOCKED_OR_RUNNING'
elif [[ -f $output/control/CANARY_LOCK_OWNER.json ]]; then
  python "$auditor" audit-lock --stage canary
  echo 'JUDGE_RECOVERY_V4_CANARY_LOCKED_BEFORE_AUTHORIZATION_OR_RUNNING'
else
  python "$auditor" audit-staged
  echo 'JUDGE_RECOVERY_V4_CPU_STAGED_AWAITING_SEPARATE_ONE_CALL_AUTHORIZATION'
fi
