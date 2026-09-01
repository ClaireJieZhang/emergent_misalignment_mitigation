#!/bin/bash
# CPU-stage, derive, and audit job 271409's batch-2 result on Tillicum.

set -euo pipefail
umask 077
ulimit -c 0

[[ $# -eq 0 ]] || {
  echo 'Usage: run_massive_medical_kalai_s1_batch2_result_recovery_v1_tillicum.sh' >&2
  exit 2
}

root=${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
source_repo=$root/projects/subliminal-mitigate-mmu-kalai-s1-recovery-continuation-v1
source_output=$root/outputs/massive_medical_kalai_s1_recovery_continuation_v1
output=$root/outputs/massive_medical_kalai_s1_batch2_result_recovery_v1
env_root=$root/envs/subliminal-mitigate-py311

test "$(basename "$repo")" = subliminal-mitigate-mmu-kalai-s1-batch2-result-recovery-v1
test -d "$source_repo/.git"
test "$(git -C "$source_repo" rev-parse HEAD)" = fb4056fcdd77f25bbadc797060dc12edcd340f52
test -z "$(git -C "$source_repo" status --porcelain)"
test -s "$source_output/control/batches/batch_02/STOPPED"
test ! -e "$source_output/control/batches/batch_02/RESULT.json"
test ! -e "$source_output/control/batches/batch_03"
test ! -e "$output"
test -z "$(git -C "$repo" status --porcelain)"

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY
unset CUDA_VISIBLE_DEVICES TRANSFORMERS_CACHE
module load conda/Miniforge3-25.3.1-3
conda activate "$env_root"
export PYTHONPATH=$repo
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-kalai-s1-batch2-recovery-pyc
export DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

cd "$repo"
python -m pip check
bash -n scripts/run_massive_medical_kalai_s1_batch2_result_recovery_v1_tillicum.sh
python -m py_compile \
  scripts/manage_massive_medical_kalai_s1_batch2_result_recovery_v1.py \
  scripts/evaluate_massive_medical_kalai_s1_recovery_continuation_batch_v1.py \
  scripts/sample_massive_medical_kalai_s1_recovery_continuation_batch_v1.py
python -m unittest tests.test_massive_medical_kalai_s1_batch2_result_recovery_v1
python scripts/manage_massive_medical_kalai_s1_batch2_result_recovery_v1.py --self-test

python scripts/manage_massive_medical_kalai_s1_batch2_result_recovery_v1.py \
  --stage \
  --source-output-root "$source_output" \
  --source-repo-root "$source_repo" \
  --output-root "$output" \
  --repo-root "$repo"
python scripts/manage_massive_medical_kalai_s1_batch2_result_recovery_v1.py \
  --stage \
  --source-output-root "$source_output" \
  --source-repo-root "$source_repo" \
  --output-root "$output" \
  --repo-root "$repo"
python scripts/manage_massive_medical_kalai_s1_batch2_result_recovery_v1.py \
  --recover --output-root "$output" --repo-root "$repo"
python scripts/manage_massive_medical_kalai_s1_batch2_result_recovery_v1.py \
  --audit-only --output-root "$output" --repo-root "$repo"

test -s "$output/control/RECOVERY_PLAN.json"
test -s "$output/control/CPU_STAGE.json"
test -s "$output/control/RECOVERED_RESULT.json"
test "$(find "$output" -type f | wc -l)" = 3
test -z "$(find "$output" -type l -print -quit)"
test -s "$source_output/control/batches/batch_02/STOPPED"
test ! -e "$source_output/control/batches/batch_02/RESULT.json"
test ! -e "$source_output/control/batches/batch_03"
test -z "$(git -C "$source_repo" status --porcelain)"
test -z "$(git -C "$repo" status --porcelain)"
echo KALAI_S1_BATCH2_RESULT_RECOVERY_V1_CPU_COMPLETE_NO_GPU_OR_API
