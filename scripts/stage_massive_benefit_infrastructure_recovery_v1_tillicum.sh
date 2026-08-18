#!/bin/bash
# Update and audit the clean Tillicum checkout for MASSIVE recovery; no Slurm.

set -euo pipefail
umask 077

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TILLICUM_HOST=${TILLICUM_HOST:-tillicum}
TILLICUM_ROOT=${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}
REMOTE_REPO_URL=${REMOTE_REPO_URL:-https://github.com/ClaireJieZhang/emergent_misalignment_mitigation.git}
REMOTE_BRANCH=${REMOTE_BRANCH:-claire/capability-quorum-secure-code}
REMOTE_REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate
expected_commit=$(git -C "$repo_root" rev-parse HEAD)
original_commit=3d2b32fe2c23ff2d07a3fe07e920cd8a09df43df

required_files=(
  docs/massive_benefit_infrastructure_recovery_v1.md
  scripts/audit_massive_benefit_infrastructure_recovery_v1.py
  scripts/sbatch_massive_benefit_infrastructure_recovery_v1_evaluate_tillicum_h200.sbatch
  scripts/sbatch_massive_benefit_infrastructure_recovery_v1_train_tillicum_h200.sbatch
  scripts/stage_massive_benefit_infrastructure_recovery_v1_tillicum.sh
  scripts/status_massive_benefit_infrastructure_recovery_v1_tillicum.sh
  scripts/submit_massive_benefit_infrastructure_recovery_v1_tillicum.sh
  scripts/train_single_sft.py
  tests/test_massive_benefit_infrastructure_recovery_v1.py
  tests/test_train_single_sft_offline_snapshot.py
)

test "$(git -C "$repo_root" rev-parse HEAD^)" = "$original_commit" || {
  echo "Recovery commit is not a direct child of $original_commit" >&2
  exit 2
}
for path in "${required_files[@]}"; do
  test -s "$repo_root/$path" || {
    echo "Missing recovery file: $path" >&2
    exit 2
  }
  git -C "$repo_root" cat-file -e "$expected_commit:$path" 2>/dev/null || {
    echo "Recovery file is not committed at $expected_commit: $path" >&2
    exit 2
  }
done
test -z "$(git -C "$repo_root" status --porcelain -- "${required_files[@]}")" || {
  echo "Recovery files differ from committed HEAD." >&2
  git -C "$repo_root" status --short -- "${required_files[@]}" >&2
  exit 2
}

echo "=== Update the dedicated clean Tillicum checkout ==="
ssh "$TILLICUM_HOST" bash -s -- \
  "$TILLICUM_ROOT" "$REMOTE_REPO_URL" "$REMOTE_BRANCH" "$expected_commit" <<'REMOTE'
set -euo pipefail
umask 077
root=$1
repo_url=$2
branch=$3
expected_commit=$4
repo=$root/projects/subliminal-mitigate

test -d "$repo/.git"
test -z "$(git -C "$repo" status --porcelain --untracked-files=all)" || {
  echo "Refusing to update dirty Tillicum checkout: $repo" >&2
  git -C "$repo" status --short >&2
  exit 3
}
git -C "$repo" fetch origin "$branch"
git -C "$repo" checkout -B "$branch" FETCH_HEAD
test "$(git -C "$repo" rev-parse HEAD)" = "$expected_commit"
test -z "$(git -C "$repo" status --porcelain --untracked-files=all)"
git -C "$repo" log -1 --oneline
REMOTE

echo "=== Run offline recovery and frozen-workflow audits ==="
ssh "$TILLICUM_HOST" bash -s -- "$TILLICUM_ROOT" "$expected_commit" <<'REMOTE'
set -euo pipefail
umask 077
ulimit -c 0

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY

root=$1
expected_commit=$2
repo=$root/projects/subliminal-mitigate
env_root=$root/envs/subliminal-mitigate-py311
output=$root/outputs/massive_benefit_pilot_v1
logs=$root/outputs/logs
snapshot=$root/cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75

test "$(git -C "$repo" rev-parse HEAD)" = "$expected_commit"
test -z "$(git -C "$repo" status --porcelain --untracked-files=all)"
test -f "$env_root/.ready"
test ! -e "$output/control/INFRASTRUCTURE_RECOVERY_V1_SUBMISSION_LOCK"
test ! -e "$output/control/infrastructure_recovery_v1"
test ! -e "$output/model/massive_en_benefit_pilot_infrastructure_recovery_v1"
test ! -e "$output/evaluation/infrastructure_recovery_v1"

module load conda/Miniforge3-25.3.1-3
conda activate "$env_root"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/massive-recovery-v1-stage-pyc
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export UNSLOTH_DISABLE_STATISTICS=1
export DO_NOT_TRACK=1

cd "$repo"
python -m pip check
python -m py_compile \
  scripts/train_single_sft.py \
  scripts/audit_massive_benefit_infrastructure_recovery_v1.py
bash -n scripts/stage_massive_benefit_infrastructure_recovery_v1_tillicum.sh
bash -n scripts/submit_massive_benefit_infrastructure_recovery_v1_tillicum.sh
bash -n scripts/status_massive_benefit_infrastructure_recovery_v1_tillicum.sh
bash -n scripts/sbatch_massive_benefit_infrastructure_recovery_v1_train_tillicum_h200.sbatch
bash -n scripts/sbatch_massive_benefit_infrastructure_recovery_v1_evaluate_tillicum_h200.sbatch
python -m unittest \
  tests.test_train_single_sft_offline_snapshot \
  tests.test_massive_benefit_infrastructure_recovery_v1 \
  tests.test_completion_only_sft \
  tests.test_massive_benefit_pilot \
  tests.test_massive_benefit_tillicum_workflow
python scripts/audit_massive_benefit_infrastructure_recovery_v1.py \
  verify-preflight \
  --repo-root "$repo" \
  --tillicum-root "$root" \
  --output-root "$output" \
  --logs-root "$logs" \
  --local-model-path "$snapshot"
REMOTE

echo "MASSIVE infrastructure recovery staging passed. No Slurm job was submitted."
echo 'Cumulative hard maximum: 168 H200-minutes / $2.520 within the original $2.925 cap.'
echo "Explicit exact-once dispatch command (not run):"
echo "  ssh $TILLICUM_HOST 'cd $REMOTE_REPO_ROOT && scripts/submit_massive_benefit_infrastructure_recovery_v1_tillicum.sh recover --ack-original-max-cost-usd 2.925'"
