#!/bin/bash
# Clone and CPU-stage the recovery-aware batches 2--7 controller.

set -euo pipefail
umask 077
ulimit -c 0

[[ $# -eq 0 ]] || {
  echo 'Usage: stage_massive_medical_kalai_s1_recovery_continuation_v1_tillicum.sh' >&2
  exit 2
}

local_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
host=${TILLICUM_HOST:-tillicum}
root=${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}
url=${REMOTE_REPO_URL:-https://github.com/ArtinTD/subliminal-mitigate.git}
branch=${REMOTE_BRANCH:-claire/massive-medical-kalai-s1-recovery-continuation-v1}
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
repo=$root/projects/subliminal-mitigate-mmu-kalai-s1-recovery-continuation-v1
output=$root/outputs/massive_medical_kalai_s1_recovery_continuation_v1
source_repo=$root/projects/subliminal-mitigate-mmu-kalai-s1-completion-batches-v1
source_output=$root/outputs/massive_medical_kalai_s1_completion_batches_v1
recovery_repo=$root/projects/subliminal-mitigate-mmu-kalai-s1-batch1-result-recovery-v1
recovery_output=$root/outputs/massive_medical_kalai_s1_batch1_result_recovery_v1
logs=$root/outputs/logs
env_root=$root/envs/subliminal-mitigate-py311

test -d "$source_repo/.git"
test "$(git -C "$source_repo" rev-parse HEAD)" = a1e8ca218635af6dafdca7ddb3e9d42d153d6921
test -z "$(git -C "$source_repo" status --porcelain)"
test -d "$recovery_repo/.git"
test "$(git -C "$recovery_repo" rev-parse HEAD)" = d5365c755cc27555451885013b4217b2a7561cb1
test -z "$(git -C "$recovery_repo" status --porcelain)"
test -s "$source_output/control/batches/batch_01/STOPPED"
test ! -e "$source_output/control/batches/batch_01/RESULT.json"
test ! -e "$source_output/control/batches/batch_02"
test -s "$recovery_output/control/RECOVERY_PLAN.json"
test -s "$recovery_output/control/CPU_STAGE.json"
test -s "$recovery_output/control/RECOVERED_RESULT.json"
test "$(find "$recovery_output" -type f | wc -l)" = 3
test "$(find "$recovery_output" -mindepth 1 -maxdepth 1 -printf '%f\n')" = control
test -z "$(find "$recovery_output" -type l -print -quit)"
test ! -e "$repo"
test ! -e "$output"
if compgen -G "$logs/massive_medical_kalai_s1_recovery_continuation_batch_*" >/dev/null; then
  echo 'Recovery-continuation Slurm log namespace is not fresh.' >&2
  exit 4
fi

git clone --branch "$branch" --single-branch "$url" "$repo"
test "$(git -C "$repo" rev-parse HEAD)" = "$expected"
while read -r mode object stage path; do
  test "$stage" = 0
  case "$mode" in
    100755) chmod 0755 "$repo/$path" ;;
    100644) chmod 0644 "$repo/$path" ;;
    *) echo "Unsupported tracked mode $mode for $path" >&2; exit 5 ;;
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
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-kalai-s1-recovery-continuation-stage-pyc
export DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

cd "$repo"
python -m pip check
bash -n \
  scripts/stage_massive_medical_kalai_s1_recovery_continuation_v1_tillicum.sh \
  scripts/submit_massive_medical_kalai_s1_recovery_continuation_batch_v1_tillicum.sh \
  scripts/sbatch_massive_medical_kalai_s1_recovery_continuation_batch_v1_tillicum_h200.sbatch
python -m py_compile \
  scripts/manage_massive_medical_kalai_s1_recovery_continuation_v1.py \
  scripts/authorize_massive_medical_kalai_s1_recovery_continuation_batch_v1.py \
  scripts/sample_massive_medical_kalai_s1_recovery_continuation_batch_v1.py \
  scripts/evaluate_massive_medical_kalai_s1_recovery_continuation_batch_v1.py \
  scripts/assemble_massive_medical_kalai_s1_recovery_continuation_v1.py
python -m unittest tests.test_massive_medical_kalai_s1_recovery_continuation_v1
python scripts/manage_massive_medical_kalai_s1_recovery_continuation_v1.py self-test
python scripts/authorize_massive_medical_kalai_s1_recovery_continuation_batch_v1.py self-test
python scripts/sample_massive_medical_kalai_s1_recovery_continuation_batch_v1.py --self-test
python scripts/evaluate_massive_medical_kalai_s1_recovery_continuation_batch_v1.py --self-test
python scripts/assemble_massive_medical_kalai_s1_recovery_continuation_v1.py --self-test

python scripts/manage_massive_medical_kalai_s1_recovery_continuation_v1.py stage \
  --source-output-root "$source_output" \
  --source-repo-root "$source_repo" \
  --recovery-output-root "$recovery_output" \
  --recovery-repo-root "$recovery_repo" \
  --output-root "$output" \
  --repo-root "$repo"
python scripts/manage_massive_medical_kalai_s1_recovery_continuation_v1.py stage \
  --source-output-root "$source_output" \
  --source-repo-root "$source_repo" \
  --recovery-output-root "$recovery_output" \
  --recovery-repo-root "$recovery_repo" \
  --output-root "$output" \
  --repo-root "$repo"
for batch_index in $(seq 2 7); do
  python scripts/manage_massive_medical_kalai_s1_recovery_continuation_v1.py preflight \
    --output-root "$output" --repo-root "$repo" --batch-index "$batch_index"
done

test -s "$output/control/RECOVERY_CONTINUATION_PLAN.json"
test -s "$output/control/CPU_STAGE.json"
test "$(find "$output" -type f | wc -l)" = 2
test "$(find "$output" -mindepth 1 -maxdepth 1 -printf '%f\n')" = control
test "$(find "$output/control" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort)" = $'CPU_STAGE.json\nRECOVERY_CONTINUATION_PLAN.json'
test -z "$(find "$output" -type l -print -quit)"
test ! -e "$output/control/batches"
test ! -e "$output/generation"
test ! -e "$output/assembled"
test ! -e "$output/judge"
test -s "$source_output/control/batches/batch_01/STOPPED"
test ! -e "$source_output/control/batches/batch_01/RESULT.json"
test ! -e "$source_output/control/batches/batch_02"
test "$(find "$recovery_output" -type f | wc -l)" = 3
test -z "$(git -C "$repo" status --porcelain)"
test -z "$(git -C "$source_repo" status --porcelain)"
test -z "$(git -C "$recovery_repo" status --porcelain)"
if compgen -G "$logs/massive_medical_kalai_s1_recovery_continuation_batch_*" >/dev/null; then
  echo 'CPU stage unexpectedly created a Slurm log.' >&2
  exit 6
fi
echo KALAI_S1_RECOVERY_CONTINUATION_V1_CPU_STAGED_NO_GPU_OR_API_AUTHORITY
REMOTE

echo 'Kalai s=1 recovery-aware batches 2--7 controller CPU-staged.'
echo 'No authorization, Slurm job, GPU allocation, model load, or API call occurred.'
