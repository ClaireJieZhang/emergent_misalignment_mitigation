#!/bin/bash
# Stage K&K v3 and prepare its fresh sets without submitting a GPU job.

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TILLICUM_HOST=${TILLICUM_HOST:-tillicum}
ROOT=${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}
REMOTE_REPO_URL=${REMOTE_REPO_URL:-https://github.com/ClaireJieZhang/emergent_misalignment_mitigation.git}
REMOTE_BRANCH=${REMOTE_BRANCH:-claire/capability-quorum-secure-code}
REMOTE_REPO=$ROOT/projects/subliminal-mitigate
commit=$(git -C "$repo_root" rev-parse HEAD)

required=(
  configs/training_qwen25_7b_kk_reasoning_pilot.yaml
  docs/knights_knaves_reasoning_confirmation_v3_protocol.md
  scripts/prepare_knights_knaves_confirmation_v3_data.py
  scripts/preflight_knights_knaves_confirmation_v3.py
  scripts/sample_knights_knaves_generations.py
  scripts/evaluate_knights_knaves_confirmation_v3.py
  scripts/summarize_knights_knaves_confirmation_v3.py
  scripts/audit_knights_knaves_confirmation_v3_workflow.py
  scripts/sbatch_knights_knaves_reasoning_confirmation_v3_tillicum_h200.sbatch
  scripts/submit_knights_knaves_reasoning_confirmation_v3_tillicum.sh
  scripts/status_knights_knaves_reasoning_confirmation_v3_tillicum.sh
)
for path in "${required[@]}"; do
  test -s "$repo_root/$path"
  git -C "$repo_root" cat-file -e "$commit:$path"
done

ssh "$TILLICUM_HOST" bash -s -- \
  "$ROOT" "$REMOTE_REPO_URL" "$REMOTE_BRANCH" "$commit" <<'REMOTE'
set -euo pipefail
root=$1 repo_url=$2 branch=$3 commit=$4
repo=$root/projects/subliminal-mitigate
v3=$root/outputs/knights_knaves_reasoning_confirmation_v3
control=$v3/control
umask 077
mkdir -p "$root/projects" "$root/outputs/logs" "$root/cache" "$root/config" "$root/tmp"
for directory in "$v3" "$control"; do
  if [[ -e "$directory" || -L "$directory" ]]; then
    test -d "$directory" && test ! -L "$directory" || {
      echo "Refusing unsafe v3 directory: $directory" >&2
      exit 3
    }
  fi
done
for state in AUTHORIZED_MAX_COST_USD_0.45 jobs.tsv SUBMITTED RELEASED; do
  if [[ -e "$control/$state" ]]; then
    echo "Refusing to restage after v3 dispatch state exists: $control/$state" >&2
    exit 3
  fi
done
if [[ -e "$v3/evaluation" || -L "$v3/evaluation" ]]; then
  test -d "$v3/evaluation" && test ! -L "$v3/evaluation" || {
    echo "Refusing unsafe v3 evaluation path: $v3/evaluation" >&2
    exit 3
  }
  if [[ -n "$(find "$v3/evaluation" -mindepth 1 -print -quit)" ]]; then
    echo "Refusing to restage over existing v3 evaluation outputs" >&2
    exit 3
  fi
fi
if [[ -d "$repo/.git" ]]; then
  test -z "$(git -C "$repo" status --porcelain)" || {
    echo "Refusing dirty Tillicum checkout" >&2
    git -C "$repo" status --short >&2
    exit 3
  }
  git -C "$repo" fetch origin "$branch"
  git -C "$repo" checkout -B "$branch" FETCH_HEAD
else
  git clone --branch "$branch" --single-branch "$repo_url" "$repo"
fi
test "$(git -C "$repo" rev-parse HEAD)" = "$commit"
test -z "$(git -C "$repo" status --porcelain)"
git -C "$repo" log -1 --oneline
REMOTE

ssh "$TILLICUM_HOST" bash -s -- "$ROOT" "$commit" <<'REMOTE'
set -euo pipefail
umask 077
unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY

root=$1 commit=$2
repo=$root/projects/subliminal-mitigate
env_root=$root/envs/subliminal-mitigate-py311
v1=$root/outputs/knights_knaves_reasoning_pilot_v1
v2=$root/outputs/knights_knaves_reasoning_confirmation_v2
v3=$root/outputs/knights_knaves_reasoning_confirmation_v3
data=$v3/data
control=$v3/control
prep=$control/PREP_COMPLETE
checkpoint=$v1/model/kk_reasoning_n5_pilot/checkpoint-192
config=$repo/configs/training_qwen25_7b_kk_reasoning_pilot.yaml

test "$(git -C "$repo" rev-parse HEAD)" = "$commit"
test -z "$(git -C "$repo" status --porcelain)"
test -f "$env_root/.ready"
for directory in "$v3" "$control"; do
  if [[ -e "$directory" || -L "$directory" ]]; then
    test -d "$directory" && test ! -L "$directory" || {
      echo "Refusing unsafe v3 directory: $directory" >&2
      exit 3
    }
  fi
done
test "$(sha256sum "$config" | awk '{print $1}')" = \
  5caef6baeb07f4ab4de8901001d7adb02433794e15c1024a950dc3bf59f492cb
mkdir -p "$control" "$root/cache" "$root/config" "$root/tmp"
module load conda/Miniforge3-25.3.1-3
conda activate "$env_root"
export PYTHONUNBUFFERED=1 DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1
export XDG_CACHE_HOME=$root/cache XDG_CONFIG_HOME=$root/config
export PIP_CACHE_DIR=$root/cache/pip TMPDIR=$root/tmp
export HF_HOME=$root/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME
export PYTHONPYCACHEPREFIX=$root/tmp/kk-confirmation-v3-stage-pyc
cd "$repo"

python scripts/prepare_knights_knaves_confirmation_v3_data.py \
  --v1_data_root "$v1/data" --v2_data_root "$v2/data" --output_dir "$data"
python scripts/prepare_knights_knaves_confirmation_v3_data.py \
  --v1_data_root "$v1/data" --v2_data_root "$v2/data" --output_dir "$data" \
  --audit_only
python scripts/preflight_knights_knaves_confirmation_v3.py \
  --repo_root "$repo" --v1_data_root "$v1/data" --v2_data_root "$v2/data" \
  --v3_data_root "$data" --checkpoint "$checkpoint"

bash -n scripts/stage_knights_knaves_reasoning_confirmation_v3_tillicum.sh
bash -n scripts/submit_knights_knaves_reasoning_confirmation_v3_tillicum.sh
bash -n scripts/status_knights_knaves_reasoning_confirmation_v3_tillicum.sh
bash -n scripts/sbatch_knights_knaves_reasoning_confirmation_v3_tillicum_h200.sbatch
python -m py_compile \
  scripts/prepare_knights_knaves_confirmation_v3_data.py \
  scripts/preflight_knights_knaves_confirmation_v3.py \
  scripts/evaluate_knights_knaves_confirmation_v3.py \
  scripts/summarize_knights_knaves_confirmation_v3.py \
  scripts/audit_knights_knaves_confirmation_v3_workflow.py

# Bind the preparation only after data, tokenizer/context, adapter, V2 result,
# and shell/API preflights all pass. No GPU allocation is made here.
python scripts/audit_knights_knaves_confirmation_v3_workflow.py write-prep \
  --repo-root "$repo" --v1-root "$v1" --v2-root "$v2" \
  --v3-data-root "$data" --output-file "$prep"

echo "K&K v3 non-GPU staging passed; no Slurm job was submitted."
REMOTE

echo "K&K v3 staging complete. Explicit capped submit command:"
echo "  ssh $TILLICUM_HOST 'cd $REMOTE_REPO && scripts/submit_knights_knaves_reasoning_confirmation_v3_tillicum.sh confirmation --ack-max-cost-usd 0.45'"
