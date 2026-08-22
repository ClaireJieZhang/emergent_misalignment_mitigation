#!/bin/bash
# CPU-only staging and sealing for Wave-2 evaluation recovery v1. Submits nothing.

set -euo pipefail
umask 077

[[ $# -eq 0 ]] || {
  echo 'Usage: scripts/stage_massive_medical_union_wave2_evaluation_recovery_v1_tillicum.sh' >&2
  exit 2
}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
TILLICUM_HOST=${TILLICUM_HOST:-tillicum}
TILLICUM_ROOT=${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}
REMOTE_REPO_URL=${REMOTE_REPO_URL:-https://github.com/ClaireJieZhang/emergent_misalignment_mitigation.git}
REMOTE_BRANCH=${REMOTE_BRANCH:-claire/capability-quorum-secure-code}
expected_commit=$(git -C "$repo_root" rev-parse HEAD)

required_files=(
  docs/massive_medical_union_wave2_evaluation_recovery_v1.md
  scripts/audit_massive_medical_union_wave2.py
  scripts/audit_massive_medical_union_wave2_evaluation_recovery_v1.py
  scripts/finalize_massive_medical_union_wave2_evaluation_recovery_v1_tillicum.sh
  scripts/sbatch_massive_medical_union_wave2_evaluation_recovery_v1_tillicum_h200.sbatch
  scripts/stage_massive_medical_union_wave2_evaluation_recovery_v1_tillicum.sh
  scripts/status_massive_medical_union_wave2_evaluation_recovery_v1_tillicum.sh
  scripts/submit_massive_medical_union_wave2_evaluation_recovery_v1_tillicum.sh
  tests/test_massive_medical_union_wave2.py
  tests/test_massive_medical_union_wave2_evaluation_recovery_v1.py
)
for path in "${required_files[@]}"; do
  test -s "$repo_root/$path"
  git -C "$repo_root" cat-file -e "$expected_commit:$path"
done
test -z "$(git -C "$repo_root" status --porcelain -- "${required_files[@]}")" || {
  echo 'Evaluation-recovery workflow differs from committed HEAD.' >&2
  git -C "$repo_root" status --short -- "${required_files[@]}" >&2
  exit 3
}

ssh "$TILLICUM_HOST" bash -s -- \
  "$TILLICUM_ROOT" "$REMOTE_REPO_URL" "$REMOTE_BRANCH" "$expected_commit" <<'REMOTE'
set -euo pipefail
umask 077
ulimit -c 0

root=$1
repo_url=$2
branch=$3
expected_commit=$4
original_repo=$root/projects/subliminal-mitigate-mmu-wave2
recovery_repo=$root/projects/subliminal-mitigate-mmu-wave2-eval-recovery-v1
output=$root/outputs/massive_medical_union_pilot_v1
control=$output/control/wave2_eval_recovery_v1
eval_root=$output/evaluation/wave2_eval_recovery_v1
original_eval=$output/evaluation/wave2
model_root=$output/models
env_root=$root/envs/subliminal-mitigate-py311

test "$(git -C "$original_repo" rev-parse HEAD)" = 8a96fe7c8c70f270c46d3416623ca866cb1d8fec
test -z "$(git -C "$original_repo" status --porcelain)"
test ! -e "$recovery_repo"
test ! -e "$control"
test ! -e "$eval_root"
test ! -e "$original_eval"
test ! -e "$model_root/pi_B2/MODEL_MANIFEST.json"
test ! -e "$model_root/pi_B3/MODEL_MANIFEST.json"
test ! -e "$model_root/pi_B2/TRAIN_COMPLETE"
test ! -e "$model_root/pi_B3/TRAIN_COMPLETE"
test -f "$env_root/.ready"
if compgen -G "$root/outputs/logs/massive_medical_union_wave2_eval_recovery_v1_*" >/dev/null; then
  echo 'Recovery log namespace is not fresh.' >&2
  exit 4
fi

git clone --branch "$branch" --single-branch "$repo_url" "$recovery_repo"
test "$(git -C "$recovery_repo" rev-parse HEAD)" = "$expected_commit"

# A caller umask of 077 makes a fresh clone materialize tracked executable
# files as 0700 and regular files as 0600.  Verify the committed Git modes,
# then normalize only this recovery's anchored allowlist before clean-tree
# auditing and execution.
executable_files=(
  scripts/audit_massive_medical_union_wave2.py
  scripts/audit_massive_medical_union_wave2_evaluation_recovery_v1.py
  scripts/finalize_massive_medical_union_wave2_evaluation_recovery_v1_tillicum.sh
  scripts/sbatch_massive_medical_union_wave2_evaluation_recovery_v1_tillicum_h200.sbatch
  scripts/stage_massive_medical_union_wave2_evaluation_recovery_v1_tillicum.sh
  scripts/status_massive_medical_union_wave2_evaluation_recovery_v1_tillicum.sh
  scripts/submit_massive_medical_union_wave2_evaluation_recovery_v1_tillicum.sh
)
regular_files=(
  docs/massive_medical_union_wave2_evaluation_recovery_v1.md
  tests/test_massive_medical_union_wave2.py
  tests/test_massive_medical_union_wave2_evaluation_recovery_v1.py
)
for path in "${executable_files[@]}"; do
  index_mode=$(git -C "$recovery_repo" ls-files -s -- "$path" | awk 'NR == 1 {print $1}')
  test "$index_mode" = 100755
  chmod 0755 "$recovery_repo/$path"
done
for path in "${regular_files[@]}"; do
  index_mode=$(git -C "$recovery_repo" ls-files -s -- "$path" | awk 'NR == 1 {print $1}')
  test "$index_mode" = 100644
  chmod 0644 "$recovery_repo/$path"
done
test -z "$(git -C "$recovery_repo" status --porcelain)"

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY
module load conda/Miniforge3-25.3.1-3
conda activate "$env_root"
export PYTHONUNBUFFERED=1
export DO_NOT_TRACK=1
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-wave2-eval-recovery-v1-stage-pyc

cd "$recovery_repo"
python -m pip check
bash -n \
  scripts/stage_massive_medical_union_wave2_evaluation_recovery_v1_tillicum.sh \
  scripts/submit_massive_medical_union_wave2_evaluation_recovery_v1_tillicum.sh \
  scripts/status_massive_medical_union_wave2_evaluation_recovery_v1_tillicum.sh \
  scripts/finalize_massive_medical_union_wave2_evaluation_recovery_v1_tillicum.sh \
  scripts/sbatch_massive_medical_union_wave2_evaluation_recovery_v1_tillicum_h200.sbatch
python -m py_compile \
  scripts/audit_massive_medical_union_wave2.py \
  scripts/audit_massive_medical_union_wave2_evaluation_recovery_v1.py
python -m unittest \
  tests.test_massive_medical_union_wave2 \
  tests.test_massive_medical_union_wave2_evaluation_recovery_v1 \
  tests.test_massive_union_component_evaluation

python scripts/audit_massive_medical_union_wave2_evaluation_recovery_v1.py write-prep
python scripts/audit_massive_medical_union_wave2_evaluation_recovery_v1.py \
  write-model --model-name pi_B2
python scripts/audit_massive_medical_union_wave2_evaluation_recovery_v1.py \
  write-model --model-name pi_B3
python scripts/audit_massive_medical_union_wave2_evaluation_recovery_v1.py \
  audit-models --sealed-only

staged=$control/.staged-$$
printf 'staged_at=%s\nrepo_commit=%s\ntraining_jobs=0\nevaluation_jobs=0\nexternal_api_calls=0\nwave3_submitted_or_released=false\n' \
  "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)" > "$staged"
chmod 0400 "$staged"
mv "$staged" "$control/STAGED"
REMOTE

echo 'Wave-2 evaluation recovery staged and existing adapters CPU-sealed.'
echo 'No Slurm job, training, API call, or Wave-3 allocation was made.'
echo 'Explicit one-job submit command (not run by staging):'
echo "  ssh $TILLICUM_HOST 'cd $TILLICUM_ROOT/projects/subliminal-mitigate-mmu-wave2-eval-recovery-v1 && scripts/submit_massive_medical_union_wave2_evaluation_recovery_v1_tillicum.sh evaluation-recovery-v1 --ack-max-cost-usd 0.225'"
