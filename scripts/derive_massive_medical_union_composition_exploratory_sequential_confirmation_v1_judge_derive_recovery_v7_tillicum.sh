#!/bin/bash
# Idempotent CPU-only derivation into the isolated v7 namespace.

set -euo pipefail
umask 077
ulimit -c 0

[[ $# -eq 0 ]] || { echo 'Usage: derive_..._judge_derive_recovery_v7_tillicum.sh' >&2; exit 2; }
[[ -z ${OPENAI_API_KEY:-} ]] || {
  echo 'OPENAI_API_KEY must be absent during CPU-only v7 derivation.' >&2
  exit 3
}

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-composition-exploratory-sequential-confirmation-v1-judge-derive-recovery-v7
output=$root/outputs/massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_derive_recovery_v7
manifest=$output/control/JUDGE_DERIVE_RECOVERY_V7_MANIFEST.json
auditor=$repo/scripts/audit_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_derive_recovery_v7.py
summary=$repo/scripts/summarize_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_derive_recovery_v7.py
env_root=$root/envs/subliminal-mitigate-py311

cd "$repo"
test "$(git branch --show-current)" = claire/capability-quorum-secure-code-composition-exploratory-under5-sequential-v1-judge-derive-recovery-v7
test -z "$(git status --porcelain)"
module load conda/Miniforge3-25.3.1-3
conda activate "$env_root"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-sequential-judge-derive-recovery-v7-pyc
export DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none ROCR_VISIBLE_DEVICES=''

python "$auditor" acquire-derive-lock
python "$summary" merge --derive-manifest "$manifest"
set +e
python "$summary" final --derive-manifest "$manifest"
scientific_code=$?
set -e
[[ $scientific_code -eq 0 || $scientific_code -eq 2 ]] || exit "$scientific_code"
python "$summary" audit-final --derive-manifest "$manifest"
python "$auditor" audit-complete
[[ -z ${OPENAI_API_KEY:-} ]]

echo "JUDGE_DERIVE_RECOVERY_V7_CPU_DERIVATION_COMPLETE scientific_exit=$scientific_code"
exit "$scientific_code"
