#!/bin/bash
# Idempotent CPU-only derivation after the completed v6 240-row judge stage.

set -euo pipefail
umask 077
ulimit -c 0

[[ $# -eq 0 ]] || { echo 'Usage: derive_..._judge_recovery_v6_tillicum.sh' >&2; exit 2; }
[[ -z ${OPENAI_API_KEY:-} ]] || {
  echo 'OPENAI_API_KEY must be absent during CPU-only derivation.' >&2
  exit 3
}

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-composition-exploratory-sequential-confirmation-v1-judge-recovery-v6
output=$root/outputs/massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v6
manifest=$output/control/JUDGE_RECOVERY_V6_MANIFEST.json
judge=$repo/scripts/judge_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v6.py
summary=$repo/scripts/summarize_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v6.py
env_root=$root/envs/subliminal-mitigate-py311

cd "$repo"
test "$(git branch --show-current)" = claire/capability-quorum-secure-code-composition-exploratory-under5-sequential-v1-judge-recovery-v6
test -z "$(git status --porcelain)"
test -f "$output/control/CONTINUATION_SUCCESS.json"
test ! -e "$output/control/CANARY_FAILURE.json"
test ! -e "$output/control/CONTINUATION_FAILURE.json"
test ! -e "$output/evaluation/medical/judge_checkpoint.json.001"
test -f "$output/evaluation/medical/judge_checkpoint.json.002"
test -f "$output/evaluation/medical/judge_checkpoint.json.240"

module load conda/Miniforge3-25.3.1-3
conda activate "$env_root"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-sequential-judge-recovery-v6-pyc

if [[ ! -e $output/evaluation/medical/judgments_merged.json ]]; then
  python "$judge" audit-continuation --recovery-manifest "$manifest"
fi
python "$summary" merge --recovery-manifest "$manifest"
set +e
python "$summary" final --recovery-manifest "$manifest"
scientific_code=$?
set -e
[[ $scientific_code -eq 0 || $scientific_code -eq 2 ]] || exit "$scientific_code"
python "$summary" audit-final --recovery-manifest "$manifest"

chmod 0400 "$output/evaluation/medical/judgments_merged.json"
find "$output/evaluation/final" -maxdepth 1 -type f -exec chmod 0400 {} +
chmod 0400 "$output/control/FINAL_RESULT.json"
echo "JUDGE_RECOVERY_V6_CPU_DERIVATION_COMPLETE scientific_exit=$scientific_code"
exit "$scientific_code"
