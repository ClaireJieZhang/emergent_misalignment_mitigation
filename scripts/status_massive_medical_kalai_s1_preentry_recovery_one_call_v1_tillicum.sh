#!/bin/bash
# Read-only status for the recovered Kalai s=1 one-call judge.

set +x
set -euo pipefail
umask 077
ulimit -c 0
[[ $# -eq 0 ]] || {
  echo 'Usage: status_massive_medical_kalai_s1_preentry_recovery_one_call_v1_tillicum.sh' >&2
  exit 2
}

[[ ! ${OPENAI_API_KEY+x} ]] || {
  echo 'OPENAI_API_KEY must be absent for read-only status.' >&2
  exit 3
}
root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-kalai-s1-preentry-recovery-v1
output=$root/outputs/massive_medical_kalai_s1_batch7_result_recovery_v1_kalai_s1_preentry_recovery_one_call_v1
manifest=$output/control/JUDGE_STAGE_MANIFEST.json
runner=$repo/scripts/judge_massive_medical_kalai_s1_preentry_recovery_one_call_v1.py

if [[ ! -f $manifest ]]; then
  echo KALAI_S1_RECOVERY_ONE_CALL_JUDGE_NOT_STAGED
  exit 0
fi
module load conda/Miniforge3-25.3.1-3
conda activate "$root/envs/subliminal-mitigate-py311"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-kalai-s1-preentry-recovery-one-call-v1-pyc
cd "$repo"
python "$runner" status --manifest "$manifest"
