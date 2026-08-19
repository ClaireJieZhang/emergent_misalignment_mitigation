#!/bin/bash
# Update and audit MASSIVE test-only evaluation recovery v2; submit nothing.

set -euo pipefail
umask 077

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TILLICUM_HOST=${TILLICUM_HOST:-tillicum}
TILLICUM_ROOT=${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}
REMOTE_REPO_URL=${REMOTE_REPO_URL:-https://github.com/ClaireJieZhang/emergent_misalignment_mitigation.git}
REMOTE_BRANCH=${REMOTE_BRANCH:-claire/capability-quorum-secure-code}
REMOTE_REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate
expected_commit=$(git -C "$repo_root" rev-parse HEAD)
base_commit=740ef7db7fa75488acea8ba76e000f4b786a54db

required_files=(
  docs/massive_benefit_evaluation_recovery_v2.md
  scripts/audit_massive_benefit_evaluation_recovery_v1.py
  scripts/audit_massive_benefit_evaluation_recovery_v2.py
  scripts/evaluate_massive_benefit_generations.py
  scripts/sample_massive_structured_generations.py
  scripts/sbatch_massive_benefit_evaluation_recovery_v2_tillicum_h200.sbatch
  scripts/stage_massive_benefit_evaluation_recovery_v2_tillicum.sh
  scripts/status_massive_benefit_evaluation_recovery_v2_tillicum.sh
  scripts/submit_massive_benefit_evaluation_recovery_v2_tillicum.sh
  scripts/summarize_massive_benefit_pilot.py
  tests/test_massive_benefit_evaluation_recovery_v1.py
  tests/test_massive_benefit_evaluation_recovery_v2.py
  tests/test_massive_benefit_pilot.py
)

test "$(git -C "$repo_root" rev-parse HEAD^)" = "$base_commit" || {
  echo "Recovery-v2 commit is not a direct child of $base_commit" >&2
  exit 2
}
for path in "${required_files[@]}"; do
  test -s "$repo_root/$path" || {
    echo "Missing recovery-v2 file: $path" >&2
    exit 2
  }
  git -C "$repo_root" cat-file -e "$expected_commit:$path" 2>/dev/null || {
    echo "Recovery-v2 file is not committed at $expected_commit: $path" >&2
    exit 2
  }
done
test -z "$(git -C "$repo_root" status --porcelain -- "${required_files[@]}")" || {
  echo "Recovery-v2 files differ from committed HEAD." >&2
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

echo "=== Run sealed-evidence, no-whitespace, and workflow audits ==="
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

test "$(git -C "$repo" rev-parse HEAD)" = "$expected_commit"
test -z "$(git -C "$repo" status --porcelain --untracked-files=all)"
test -f "$env_root/.ready"
test -s "$output/control/evaluation_recovery_v1/GO_MASSIVE_SEALED_TEST"
test -s "$output/evaluation/evaluation_recovery_v1/selection/summary.json"
test ! -e "$output/control/MASSIVE_EVALUATION_RECOVERY_V2_SUBMISSION_LOCK"
test ! -e "$output/control/evaluation_recovery_v2"
test ! -e "$output/evaluation/evaluation_recovery_v2"

module load gcc/13.4.0 cuda/12.9.1
module load conda/Miniforge3-25.3.1-3
conda activate "$env_root"
PYTHON=$env_root/bin/python
test -x "$PYTHON"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/massive-eval-recovery-v2-stage-pyc
export XDG_CACHE_HOME=$root/cache
export XDG_CONFIG_HOME=$root/config
export HF_HOME=$root/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export UNSLOTH_DISABLE_STATISTICS=1
export DO_NOT_TRACK=1

cd "$repo"
"$PYTHON" -m pip check
"$PYTHON" -m py_compile \
  scripts/sample_massive_structured_generations.py \
  scripts/evaluate_massive_benefit_generations.py \
  scripts/summarize_massive_benefit_pilot.py \
  scripts/audit_massive_benefit_evaluation_recovery_v1.py \
  scripts/audit_massive_benefit_evaluation_recovery_v2.py
bash -n scripts/stage_massive_benefit_evaluation_recovery_v2_tillicum.sh
bash -n scripts/submit_massive_benefit_evaluation_recovery_v2_tillicum.sh
bash -n scripts/status_massive_benefit_evaluation_recovery_v2_tillicum.sh
bash -n scripts/sbatch_massive_benefit_evaluation_recovery_v2_tillicum_h200.sbatch
"$PYTHON" -m unittest \
  tests.test_massive_benefit_evaluation_recovery_v2 \
  tests.test_massive_benefit_evaluation_recovery_v1 \
  tests.test_massive_benefit_pilot \
  tests.test_massive_benefit_infrastructure_recovery_v1 \
  tests.test_massive_benefit_tillicum_workflow
"$PYTHON" scripts/audit_massive_benefit_evaluation_recovery_v2.py \
  verify-preflight \
  --repo-root "$repo" --output-root "$output" --logs-root "$logs"

model_dir=$output/model/massive_en_benefit_pilot_infrastructure_recovery_v1
"$PYTHON" scripts/sample_massive_structured_generations.py \
  --model pi_base=BASE --model "step_30=$model_dir/checkpoint-30" \
  --training_config configs/training_qwen25_7b_massive_benefit_pilot.yaml \
  --prompt_file "$output/data/sealed_test/prompts.json" \
  --output_dir "$root/tmp/massive-eval-recovery-v2-preflight-unused" \
  --structured_constraint_profile const_tree_no_ws_v3 \
  --max_new_tokens 256 --max_context 2048 --seed 8172026 \
  --preflight_only
REMOTE

echo "MASSIVE test-only recovery-v2 staging passed. No Slurm job was submitted."
echo 'Cumulative maximum: 172 H200-minutes / $2.580.'
echo "Explicit exact-once dispatch command (not run):"
echo "  ssh $TILLICUM_HOST 'cd $REMOTE_REPO_ROOT && scripts/submit_massive_benefit_evaluation_recovery_v2_tillicum.sh recover-test --ack-original-max-cost-usd 2.925'"
