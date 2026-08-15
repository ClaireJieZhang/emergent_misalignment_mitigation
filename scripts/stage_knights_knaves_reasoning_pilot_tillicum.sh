#!/bin/bash
# Stage and seal the K&K reasoning pilot on a Tillicum login node (no Slurm).

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TILLICUM_HOST=${TILLICUM_HOST:-tillicum}
TILLICUM_ROOT=${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}
REMOTE_REPO_URL=${REMOTE_REPO_URL:-https://github.com/ClaireJieZhang/emergent_misalignment_mitigation.git}
REMOTE_BRANCH=${REMOTE_BRANCH:-claire/capability-quorum-secure-code}
REMOTE_REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate
expected_commit=$(git -C "$repo_root" rev-parse HEAD)

required_files=(
  configs/training_qwen25_7b_kk_reasoning_pilot.yaml
  docs/knights_knaves_reasoning_benefit_pilot_protocol.md
  scripts/audit_knights_knaves_tillicum_workflow.py
  scripts/prepare_knights_knaves_pilot_data.py
  scripts/sample_knights_knaves_generations.py
  scripts/evaluate_knights_knaves_generations.py
  scripts/summarize_knights_knaves_pilot.py
  scripts/sbatch_knights_knaves_reasoning_pilot_train_tillicum_h200.sbatch
  scripts/sbatch_knights_knaves_reasoning_pilot_evaluate_tillicum_h200.sbatch
  scripts/submit_knights_knaves_reasoning_pilot_tillicum.sh
  scripts/status_knights_knaves_reasoning_pilot_tillicum.sh
)
for path in "${required_files[@]}"; do
  test -s "$repo_root/$path" || { echo "Missing workflow file: $path" >&2; exit 2; }
  git -C "$repo_root" cat-file -e "$expected_commit:$path" 2>/dev/null || {
    echo "Workflow file is not committed at $expected_commit: $path" >&2
    exit 2
  }
done

echo "=== Update the dedicated clean Tillicum checkout ==="
ssh "$TILLICUM_HOST" bash -s -- \
  "$TILLICUM_ROOT" "$REMOTE_REPO_URL" "$REMOTE_BRANCH" "$expected_commit" <<'REMOTE'
set -euo pipefail
root=$1
repo_url=$2
branch=$3
expected_commit=$4
repo=$root/projects/subliminal-mitigate

umask 077
mkdir -p "$root/projects" "$root/outputs/logs" "$root/cache" "$root/config" "$root/tmp"
if [[ -d "$repo/.git" ]]; then
  if [[ -n "$(git -C "$repo" status --porcelain)" ]]; then
    echo "Refusing to update dirty Tillicum checkout: $repo" >&2
    git -C "$repo" status --short >&2
    exit 3
  fi
  git -C "$repo" fetch origin "$branch"
  git -C "$repo" checkout -B "$branch" FETCH_HEAD
else
  git clone --branch "$branch" --single-branch "$repo_url" "$repo"
fi
test "$(git -C "$repo" rev-parse HEAD)" = "$expected_commit"
test -z "$(git -C "$repo" status --porcelain)"
git -C "$repo" log -1 --oneline
REMOTE

echo "=== Prepare and seal immutable data without a GPU allocation ==="
ssh "$TILLICUM_HOST" bash -s -- "$TILLICUM_ROOT" "$expected_commit" <<'REMOTE'
set -euo pipefail
umask 077

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY

root=$1
expected_commit=$2
repo=$root/projects/subliminal-mitigate
env_root=$root/envs/subliminal-mitigate-py311
output=$root/outputs/knights_knaves_reasoning_pilot_v1
control=$output/control
data=$output/data
config=$repo/configs/training_qwen25_7b_kk_reasoning_pilot.yaml
prep=$control/PREP_COMPLETE

test "$(git -C "$repo" rev-parse HEAD)" = "$expected_commit"
test -z "$(git -C "$repo" status --porcelain)"
test -f "$env_root/.ready"
mkdir -p "$control" "$root/cache" "$root/config" "$root/tmp"

module load conda/Miniforge3-25.3.1-3
conda activate "$env_root"
export PYTHONUNBUFFERED=1
export DO_NOT_TRACK=1
export HF_HUB_DISABLE_TELEMETRY=1
export XDG_CACHE_HOME=$root/cache
export XDG_CONFIG_HOME=$root/config
export PIP_CACHE_DIR=$root/cache/pip
export TMPDIR=$root/tmp
export HF_HOME=$root/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME
export PYTHONPYCACHEPREFIX=$root/tmp/kk-reasoning-stage-pyc
mkdir -p "$XDG_CONFIG_HOME" "$PIP_CACHE_DIR" "$TMPDIR" "$HUGGINGFACE_HUB_CACHE"

cd "$repo"
python scripts/prepare_knights_knaves_pilot_data.py --output_dir "$data"
python scripts/prepare_knights_knaves_pilot_data.py --output_dir "$data" --audit_only
python scripts/audit_knights_knaves_tillicum_workflow.py write-prep \
  --repo-root "$repo" --data-root "$data" --training-config "$config" \
  --output-file "$prep"

bash -n scripts/stage_knights_knaves_reasoning_pilot_tillicum.sh
bash -n scripts/submit_knights_knaves_reasoning_pilot_tillicum.sh
bash -n scripts/status_knights_knaves_reasoning_pilot_tillicum.sh
bash -n scripts/sbatch_knights_knaves_reasoning_pilot_train_tillicum_h200.sbatch
bash -n scripts/sbatch_knights_knaves_reasoning_pilot_evaluate_tillicum_h200.sbatch
python -m py_compile \
  scripts/audit_knights_knaves_tillicum_workflow.py \
  scripts/prepare_knights_knaves_pilot_data.py \
  scripts/sample_knights_knaves_generations.py \
  scripts/evaluate_knights_knaves_generations.py \
  scripts/summarize_knights_knaves_pilot.py

echo "Non-GPU K&K staging and immutable-data audit passed."
REMOTE

echo "Tillicum staging complete. No Slurm job was submitted."
echo "Initial released maximum: 150 H200-minutes = \$2.25."
echo "Immutable cumulative ceiling: 240 H200-minutes = \$3.60; reserve is not submitted."
echo "Explicit submit command (not run):"
echo "  ssh $TILLICUM_HOST 'cd $REMOTE_REPO_ROOT && scripts/submit_knights_knaves_reasoning_pilot_tillicum.sh pilot --ack-max-cost-usd 3.60'"
