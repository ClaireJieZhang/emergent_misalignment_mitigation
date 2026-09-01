#!/bin/bash
# Read-only status for the contextual-baseline split judge.

set -euo pipefail
umask 077
ulimit -c 0

[[ $# -eq 0 ]] || {
  echo 'Usage: status_massive_medical_composition_contextual_baseline_judge_v1_tillicum.sh' >&2
  exit 2
}

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-composition-contextual-baseline-judge-v1
output=$root/outputs/massive_medical_composition_contextual_baseline_judge_v1
runner=$repo/scripts/judge_massive_medical_composition_contextual_baselines_split_v1.py
manifest=$output/control/JUDGE_STAGE_MANIFEST.json

if [[ ! -f $manifest ]]; then
  echo 'CONTEXTUAL_BASELINE_JUDGE_NOT_STAGED'
  exit 0
fi
module load conda/Miniforge3-25.3.1-3
conda activate "$root/envs/subliminal-mitigate-py311"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/contextual-baseline-judge-v1-pyc
cd "$repo"
python "$runner" status --manifest "$manifest"
