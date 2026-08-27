#!/bin/bash
# Read-only CPU status/audit for the v7 derivation recovery.

set -euo pipefail
umask 077
ulimit -c 0

[[ $# -eq 0 ]] || { echo 'Usage: status_..._judge_derive_recovery_v7_tillicum.sh' >&2; exit 2; }
[[ -z ${OPENAI_API_KEY:-} ]] || {
  echo 'OPENAI_API_KEY must be absent during CPU-only v7 status audit.' >&2
  exit 3
}

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-composition-exploratory-sequential-confirmation-v1-judge-derive-recovery-v7
output=$root/outputs/massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_derive_recovery_v7
manifest=$output/control/JUDGE_DERIVE_RECOVERY_V7_MANIFEST.json
auditor=$repo/scripts/audit_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_derive_recovery_v7.py
summary=$repo/scripts/summarize_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_derive_recovery_v7.py

if [[ ! -d $repo || ! -d $output ]]; then
  echo 'JUDGE_DERIVE_RECOVERY_V7_NOT_STAGED'
  exit 0
fi

module load conda/Miniforge3-25.3.1-3
conda activate "$root/envs/subliminal-mitigate-py311"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-sequential-judge-derive-recovery-v7-pyc
export CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none ROCR_VISIBLE_DEVICES=''
cd "$repo"

if [[ -f $output/control/FINAL_RESULT.json ]]; then
  python "$summary" audit-final --derive-manifest "$manifest"
  python "$auditor" audit-complete
  echo 'JUDGE_DERIVE_RECOVERY_V7_FINAL_COMPLETE'
elif [[ -f $output/control/DERIVATION_LOCK.json ]]; then
  python "$auditor" audit-derive-state
  echo 'JUDGE_DERIVE_RECOVERY_V7_CPU_DERIVATION_INCOMPLETE_EXACT_IDEMPOTENT_REENTRY_ONLY'
else
  python "$auditor" audit-staged
  echo 'JUDGE_DERIVE_RECOVERY_V7_CPU_STAGED_AWAITING_CPU_DERIVATION'
fi
