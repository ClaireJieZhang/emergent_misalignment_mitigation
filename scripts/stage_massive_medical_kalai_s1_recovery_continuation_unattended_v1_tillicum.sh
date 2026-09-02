#!/bin/bash
# Clone and CPU-stage the unattended parent amendment on Tillicum.

set -euo pipefail
umask 077
ulimit -c 0

[[ $# -eq 0 ]] || {
  echo 'Usage: stage_massive_medical_kalai_s1_recovery_continuation_unattended_v1_tillicum.sh' >&2
  exit 2
}

local_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
host=${TILLICUM_HOST:-tillicum}
root=${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}
url=${REMOTE_REPO_URL:-https://github.com/ArtinTD/subliminal-mitigate.git}
branch=${REMOTE_BRANCH:-claire/massive-medical-kalai-s1-unattended-v1}
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
repo=$root/projects/subliminal-mitigate-mmu-kalai-s1-recovery-continuation-unattended-v1
output=$root/outputs/massive_medical_kalai_s1_recovery_continuation_unattended_v1
v3_repo=$root/projects/subliminal-mitigate-mmu-kalai-s1-recovery-continuation-v3
v3_output=$root/outputs/massive_medical_kalai_s1_recovery_continuation_v3
logs=$root/outputs/logs
env_root=$root/envs/subliminal-mitigate-py311

test -d "$v3_repo/.git"
test "$(git -C "$v3_repo" rev-parse HEAD)" = 84653e82cc2181347bddaa73bced65bcbdc5126f
test -z "$(git -C "$v3_repo" status --porcelain)"
test -s "$v3_output/control/RECOVERY_CONTINUATION_PLAN.json"
test -s "$v3_output/control/CPU_STAGE.json"
test "$(find "$v3_output" -type f | wc -l)" = 2
test -z "$(find "$v3_output" -type l -print -quit)"
test ! -e "$v3_output/control/batches"
test ! -e "$v3_output/generation"
test ! -e "$v3_output/assembled"
test ! -e "$v3_output/judge"
test -z "$(squeue -h -u "$USER" -n mmu_kalai_s1_r3c03,mmu_kalai_s1_r3c04,mmu_kalai_s1_r3c05,mmu_kalai_s1_r3c06,mmu_kalai_s1_r3c07)"
test ! -e "$repo"
test ! -e "$output"
test -z "$(tmux list-sessions -F '#S' 2>/dev/null | awk '$0=="kalai-s1-unattended-v1"')"
if compgen -G "$logs/massive_medical_kalai_s1_recovery_continuation_unattended_v1*" >/dev/null; then
  echo 'Unattended-v1 log namespace is not fresh.' >&2
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
export PYTHONPATH=$repo PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-kalai-s1-unattended-v1-stage-pyc
export DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

cd "$repo"
python -m pip check
bash -n \
  scripts/stage_massive_medical_kalai_s1_recovery_continuation_unattended_v1_tillicum.sh \
  scripts/run_massive_medical_kalai_s1_recovery_continuation_unattended_v1_tillicum.sh
python -m py_compile scripts/manage_massive_medical_kalai_s1_recovery_continuation_unattended_v1.py
python -m unittest tests.test_massive_medical_kalai_s1_recovery_continuation_unattended_v1
python scripts/manage_massive_medical_kalai_s1_recovery_continuation_unattended_v1.py self-test

python scripts/manage_massive_medical_kalai_s1_recovery_continuation_unattended_v1.py stage \
  --output-root "$output" --repo-root "$repo" \
  --v3-output-root "$v3_output" --v3-repo-root "$v3_repo"
python scripts/manage_massive_medical_kalai_s1_recovery_continuation_unattended_v1.py stage \
  --output-root "$output" --repo-root "$repo" \
  --v3-output-root "$v3_output" --v3-repo-root "$v3_repo"

test "$(find "$output" -type f | wc -l)" = 3
test -z "$(find "$output" -type l -print -quit)"
test "$(find "$output" -mindepth 1 -maxdepth 1 -printf '%f\n')" = control
test "$(find "$output/control" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort)" = $'CPU_STAGE.json\nSEQUENCE_AUTHORIZATION.json\nUNATTENDED_PLAN.json'
test ! -e "$output/control/batches"
test ! -e "$output/control/UNATTENDED_INVOCATION.json"
test ! -e "$output/control/FINAL_SEQUENCE.json"
test -z "$(git -C "$repo" status --porcelain)"
test -z "$(git -C "$v3_repo" status --porcelain)"
test "$(find "$v3_output" -type f | wc -l)" = 2
test -z "$(squeue -h -u "$USER" -n mmu_kalai_s1_r3c03,mmu_kalai_s1_r3c04,mmu_kalai_s1_r3c05,mmu_kalai_s1_r3c06,mmu_kalai_s1_r3c07)"
echo KALAI_S1_RECOVERY_CONTINUATION_UNATTENDED_V1_CPU_STAGED_VALID
REMOTE

echo 'Kalai s=1 unattended-v1 CPU-staged with sealed standing authority.'
echo 'No Slurm job, GPU allocation, model load, or API call occurred during staging.'
