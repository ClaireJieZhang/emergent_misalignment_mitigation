#!/bin/bash
# Stage K&K v2 and prepare the fresh confirmation set without a GPU job.

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
  docs/knights_knaves_reasoning_confirmation_v2_protocol.md
  scripts/prepare_knights_knaves_confirmation_v2_data.py
  scripts/sample_knights_knaves_generations.py
  scripts/sample_knights_knaves_structured_choices.py
  scripts/evaluate_knights_knaves_confirmation_v2.py
  scripts/summarize_knights_knaves_confirmation_v2.py
  scripts/audit_knights_knaves_confirmation_v2_workflow.py
  scripts/sbatch_knights_knaves_reasoning_confirmation_v2_tillicum_h200.sbatch
  scripts/submit_knights_knaves_reasoning_confirmation_v2_tillicum.sh
  scripts/status_knights_knaves_reasoning_confirmation_v2_tillicum.sh
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
v2=$root/outputs/knights_knaves_reasoning_confirmation_v2
control=$v2/control
umask 077
mkdir -p "$root/projects" "$root/outputs/logs" "$root/cache" "$root/config" "$root/tmp"
# Once a paid dispatch has been authorized, the evaluated checkout is frozen.
# Refuse before fetch/checkout so an accidental stage rerun cannot change code
# beneath a queued or running job.  PREP_COMPLETE alone remains resumable.
for state in AUTHORIZED_MAX_COST_USD_0.45 jobs.tsv SUBMITTED RELEASED; do
  if [[ -e "$control/$state" ]]; then
    echo "Refusing to restage after v2 dispatch state exists: $control/$state" >&2
    exit 3
  fi
done
if [[ -e "$v2/evaluation" ]]; then
  test -d "$v2/evaluation" && test ! -L "$v2/evaluation" || {
    echo "Refusing unsafe v2 evaluation path: $v2/evaluation" >&2
    exit 3
  }
  if [[ -n "$(find "$v2/evaluation" -mindepth 1 -print -quit)" ]]; then
    echo "Refusing to restage over existing v2 evaluation outputs" >&2
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
data=$v2/data
control=$v2/control
prep=$control/PREP_COMPLETE
training_config=$repo/configs/training_qwen25_7b_kk_reasoning_pilot.yaml

test "$(git -C "$repo" rev-parse HEAD)" = "$commit"
test -z "$(git -C "$repo" status --porcelain)"
test -f "$env_root/.ready"
test "$(sha256sum "$training_config" | awk '{print $1}')" = \
  5caef6baeb07f4ab4de8901001d7adb02433794e15c1024a950dc3bf59f492cb
mkdir -p "$control" "$root/cache" "$root/config" "$root/tmp"
module load conda/Miniforge3-25.3.1-3
conda activate "$env_root"
export PYTHONUNBUFFERED=1 DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1
export XDG_CACHE_HOME=$root/cache XDG_CONFIG_HOME=$root/config
export PIP_CACHE_DIR=$root/cache/pip TMPDIR=$root/tmp
export HF_HOME=$root/cache/huggingface HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME
export PYTHONPYCACHEPREFIX=$root/tmp/kk-confirmation-v2-stage-pyc
cd "$repo"

python scripts/prepare_knights_knaves_confirmation_v2_data.py \
  --v1_data_root "$v1/data" --output_dir "$data"
python scripts/prepare_knights_knaves_confirmation_v2_data.py \
  --v1_data_root "$v1/data" --output_dir "$data" --audit_only

# Exercise the exact tokenizer, frozen adapter audit, choice/context bounds,
# pinned vLLM/XGrammar API, escaped EBNF, and the persistent multi-set interface
# without initializing a GPU model.
preflight_args=()
for set_name in confirmation_n5 official_n4 official_n5 official_n6 fresh_n4 fresh_n5 fresh_n6; do
  if [[ "$set_name" = confirmation_n5 ]]; then
    prompt_file=$data/confirmation/confirmation_n5_prompts.json
  else
    prompt_file=$v1/data/sealed_final/${set_name}_prompts.json
  fi
  preflight_args+=(
    --prompt_file "$prompt_file"
    --names_file "$data/names/${set_name}_names.json"
  )
done
python scripts/sample_knights_knaves_structured_choices.py \
  --model pi_base=BASE \
  --model step_192=$v1/model/kk_reasoning_n5_pilot/checkpoint-192 \
  --training_config "$repo/configs/training_qwen25_7b_kk_reasoning_pilot.yaml" \
  "${preflight_args[@]}" --output_dir "$root/tmp/kk-confirmation-v2-preflight" \
  --max_new_tokens 2048 --max_context 4096 --seed 8152026 --preflight_only

bash -n scripts/stage_knights_knaves_reasoning_confirmation_v2_tillicum.sh
bash -n scripts/submit_knights_knaves_reasoning_confirmation_v2_tillicum.sh
bash -n scripts/status_knights_knaves_reasoning_confirmation_v2_tillicum.sh
bash -n scripts/sbatch_knights_knaves_reasoning_confirmation_v2_tillicum_h200.sbatch
python -m py_compile \
  scripts/prepare_knights_knaves_confirmation_v2_data.py \
  scripts/sample_knights_knaves_generations.py \
  scripts/sample_knights_knaves_structured_choices.py \
  scripts/evaluate_knights_knaves_confirmation_v2.py \
  scripts/summarize_knights_knaves_confirmation_v2.py \
  scripts/audit_knights_knaves_confirmation_v2_workflow.py

# Commit the preparation record last.  A failed API/tokenizer preflight must
# not strand a record bound to code that never passed staging.
python scripts/audit_knights_knaves_confirmation_v2_workflow.py write-prep \
  --repo-root "$repo" --v1-root "$v1" --v2-data-root "$data" \
  --output-file "$prep"

echo "K&K v2 non-GPU staging passed; no Slurm job was submitted."
REMOTE

echo "K&K v2 staging complete. Explicit capped submit command:"
echo "  ssh $TILLICUM_HOST 'cd $REMOTE_REPO && scripts/submit_knights_knaves_reasoning_confirmation_v2_tillicum.sh confirmation --ack-max-cost-usd 0.45'"
