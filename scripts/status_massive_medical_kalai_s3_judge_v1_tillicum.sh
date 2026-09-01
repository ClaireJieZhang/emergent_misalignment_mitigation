#!/bin/bash
# Read-only status for the Kalai s=3 split judge.

set -euo pipefail
umask 077
ulimit -c 0
[[ $# -eq 0 ]] || { echo 'Usage: status_massive_medical_kalai_s3_judge_v1_tillicum.sh' >&2; exit 2; }

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-kalai-s3-r20-v2-judge-v1
output=$root/outputs/massive_medical_kalai_s3_r20_v2_kalai_s3_judge_v1
manifest=$output/control/JUDGE_STAGE_MANIFEST.json
runner=$repo/scripts/judge_massive_medical_kalai_s3_split_v1.py

if [[ ! -f $manifest ]]; then
  echo KALAI_S3_JUDGE_NOT_STAGED
  exit 0
fi
module load conda/Miniforge3-25.3.1-3
conda activate "$root/envs/subliminal-mitigate-py311"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-kalai-s3-judge-v1-pyc
cd "$repo"
python "$runner" status --manifest "$manifest"
