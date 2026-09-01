#!/bin/bash
# Clone, CPU-stage, execute, and audit the derivation-only batch-1 recovery.

set -euo pipefail
umask 077
ulimit -c 0

[[ $# -eq 0 ]] || {
  echo 'Usage: run_massive_medical_kalai_s1_batch1_result_recovery_v1_tillicum.sh' >&2
  exit 2
}

local_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
host=${TILLICUM_HOST:-tillicum}
root=${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}
url=${REMOTE_REPO_URL:-https://github.com/ArtinTD/subliminal-mitigate.git}
branch=${REMOTE_BRANCH:-claire/massive-medical-kalai-s1-batch1-result-recovery-v1}
commit=$(git -C "$local_repo" rev-parse HEAD)
test "$(git -C "$local_repo" branch --show-current)" = "$branch"
test -z "$(git -C "$local_repo" status --porcelain)"

ssh "$host" bash -s -- "$root" "$url" "$branch" "$commit" <<'REMOTE'
set -euo pipefail
umask 077
ulimit -c 0

root=$1
url=$2
branch=$3
expected=$4
repo=$root/projects/subliminal-mitigate-mmu-kalai-s1-batch1-result-recovery-v1
output=$root/outputs/massive_medical_kalai_s1_batch1_result_recovery_v1
source_repo=$root/projects/subliminal-mitigate-mmu-kalai-s1-completion-batches-v1
source_output=$root/outputs/massive_medical_kalai_s1_completion_batches_v1
env_root=$root/envs/subliminal-mitigate-py311

test -d "$source_repo/.git"
test "$(git -C "$source_repo" rev-parse HEAD)" = a1e8ca218635af6dafdca7ddb3e9d42d153d6921
test -z "$(git -C "$source_repo" status --porcelain)"
test -s "$source_output/control/batches/batch_01/STOPPED"
test ! -e "$source_output/control/batches/batch_01/RESULT.json"
test ! -e "$source_output/control/batches/batch_02"
test ! -e "$repo"
test ! -e "$output"

git clone --branch "$branch" --single-branch "$url" "$repo"
test "$(git -C "$repo" rev-parse HEAD)" = "$expected"
while read -r mode object stage path; do
  test "$stage" = 0
  case "$mode" in
    100755) chmod 0755 "$repo/$path" ;;
    100644) chmod 0644 "$repo/$path" ;;
    *) echo "Unsupported tracked mode $mode for $path" >&2; exit 3 ;;
  esac
done < <(git -C "$repo" ls-files -s)
test -z "$(git -C "$repo" status --porcelain)"

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY
unset CUDA_VISIBLE_DEVICES TRANSFORMERS_CACHE
module load conda/Miniforge3-25.3.1-3
conda activate "$env_root"
export PYTHONPATH=$repo
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-kalai-s1-batch1-recovery-pyc
export DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

cd "$repo"
python -m pip check
bash -n scripts/run_massive_medical_kalai_s1_batch1_result_recovery_v1_tillicum.sh
python -m py_compile \
  scripts/prepare_massive_medical_kalai_s1_batch1_result_recovery_v1.py \
  scripts/stage_massive_medical_kalai_s1_batch1_result_recovery_v1.py \
  scripts/evaluate_massive_medical_kalai_s1_batch1_result_recovery_v1.py \
  scripts/sample_massive_medical_kalai_s1_completion_batch_v1.py
python -m unittest tests.test_massive_medical_kalai_s1_batch1_result_recovery_v1
python scripts/prepare_massive_medical_kalai_s1_batch1_result_recovery_v1.py --self-test
python scripts/stage_massive_medical_kalai_s1_batch1_result_recovery_v1.py --self-test
python scripts/evaluate_massive_medical_kalai_s1_batch1_result_recovery_v1.py --self-test

python scripts/stage_massive_medical_kalai_s1_batch1_result_recovery_v1.py \
  --source-output-root "$source_output" \
  --source-repo-root "$source_repo" \
  --output-root "$output" \
  --repo-root "$repo"
python scripts/stage_massive_medical_kalai_s1_batch1_result_recovery_v1.py \
  --source-output-root "$source_output" \
  --source-repo-root "$source_repo" \
  --output-root "$output" \
  --repo-root "$repo"
python scripts/evaluate_massive_medical_kalai_s1_batch1_result_recovery_v1.py \
  --output-root "$output" \
  --repo-root "$repo" \
  --recover
python scripts/evaluate_massive_medical_kalai_s1_batch1_result_recovery_v1.py \
  --output-root "$output" \
  --repo-root "$repo" \
  --audit-only

test -s "$output/control/RECOVERY_PLAN.json"
test -s "$output/control/CPU_STAGE.json"
test -s "$output/control/RECOVERED_RESULT.json"
test "$(find "$output" -type f | wc -l)" = 3
test -z "$(find "$output" -type l -print -quit)"
test -s "$source_output/control/batches/batch_01/STOPPED"
test ! -e "$source_output/control/batches/batch_01/RESULT.json"
test ! -e "$source_output/control/batches/batch_02"
test -z "$(git -C "$source_repo" status --porcelain)"
test -z "$(git -C "$repo" status --porcelain)"
echo KALAI_S1_BATCH1_RESULT_RECOVERY_V1_CPU_COMPLETE_NO_GPU_OR_API
REMOTE

echo 'Kalai s=1 batch-1 result recovery completed and audited on CPU.'
echo 'No GPU job, API call, regeneration, retry, or batch-2 submission occurred.'
