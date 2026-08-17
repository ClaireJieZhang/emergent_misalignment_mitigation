#!/bin/bash
# Stage pinned MASSIVE data/model assets on Tillicum without submitting Slurm.

set -euo pipefail
umask 077

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TILLICUM_HOST=${TILLICUM_HOST:-tillicum}
TILLICUM_ROOT=${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}
REMOTE_REPO_URL=${REMOTE_REPO_URL:-https://github.com/ClaireJieZhang/emergent_misalignment_mitigation.git}
REMOTE_BRANCH=${REMOTE_BRANCH:-claire/capability-quorum-secure-code}
REMOTE_REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate
expected_commit=$(git -C "$repo_root" rev-parse HEAD)

required_files=(
  requirements.txt
  train_sft.py
  scripts/train_single_sft.py
  configs/training_qwen25_7b_massive_benefit_pilot.yaml
  docs/massive_benefit_pilot_protocol.md
  scripts/audit_massive_benefit_tillicum_workflow.py
  scripts/prepare_massive_benefit_pilot_data.py
  scripts/sample_massive_structured_generations.py
  scripts/evaluate_massive_benefit_generations.py
  scripts/summarize_massive_benefit_pilot.py
  scripts/sbatch_massive_benefit_base_dev_tillicum_h200.sbatch
  scripts/sbatch_massive_benefit_train_tillicum_h200.sbatch
  scripts/sbatch_massive_benefit_evaluate_tillicum_h200.sbatch
  scripts/submit_massive_benefit_pilot_tillicum.sh
  scripts/stage_massive_benefit_pilot_tillicum.sh
  scripts/status_massive_benefit_pilot_tillicum.sh
  tests/test_completion_only_sft.py
  tests/test_massive_benefit_pilot.py
  tests/test_massive_benefit_tillicum_workflow.py
)
for path in "${required_files[@]}"; do
  test -s "$repo_root/$path" || {
    echo "Missing workflow file: $path" >&2
    exit 2
  }
  git -C "$repo_root" cat-file -e "$expected_commit:$path" 2>/dev/null || {
    echo "Workflow file is not committed at $expected_commit: $path" >&2
    exit 2
  }
done
test -z "$(git -C "$repo_root" status --porcelain -- "${required_files[@]}")" || {
  echo "Pilot workflow files differ from committed HEAD; refusing stale staging." >&2
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

mkdir -p "$root/projects" "$root/outputs/logs" "$root/cache" \
  "$root/config" "$root/tmp"
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

echo "=== Build fresh immutable data and cache pinned public model assets ==="
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
config=$repo/configs/training_qwen25_7b_massive_benefit_pilot.yaml

test "$(git -C "$repo" rev-parse HEAD)" = "$expected_commit"
test -z "$(git -C "$repo" status --porcelain)"
test -f "$env_root/.ready"
test ! -e "$output"
build=$(mktemp -d "$root/outputs/.massive_benefit_pilot_v1.build.XXXXXX")
control=$build/control
data=$build/data
mkdir -p "$control" "$build/evaluation" "$build/model" "$root/cache" \
  "$root/config" "$root/tmp"
printf 'repo_commit=%s\ncreated_at=%s\n' "$expected_commit" \
  "$(date --iso-8601=seconds)" > "$control/STAGING_IN_PROGRESS"

module load conda/Miniforge3-25.3.1-3
conda activate "$env_root"
export PYTHONUNBUFFERED=1
export DO_NOT_TRACK=1
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1
export XDG_CACHE_HOME=$root/cache
export XDG_CONFIG_HOME=$root/config
export PIP_CACHE_DIR=$root/cache/pip
export TMPDIR=$root/tmp
export HF_HOME=$root/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME
export VLLM_CACHE_ROOT=$root/cache/vllm
export TRITON_CACHE_DIR=$root/cache/triton
export PYTHONPYCACHEPREFIX=$root/tmp/massive-stage-pyc
mkdir -p "$XDG_CONFIG_HOME" "$PIP_CACHE_DIR" "$TMPDIR" \
  "$HUGGINGFACE_HUB_CACHE" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR"

cd "$repo"
python -m pip check
python scripts/prepare_massive_benefit_pilot_data.py --output_dir "$data"
python scripts/prepare_massive_benefit_pilot_data.py \
  --output_dir "$data" --audit_only

python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download

revision = "bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
snapshot = Path(snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    revision=revision,
)).resolve()
if snapshot.name != revision:
    raise RuntimeError(f"Pinned model resolved to unexpected snapshot: {snapshot}")
print(snapshot)
PY

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
python scripts/sample_massive_structured_generations.py \
  --model pi_base=BASE --training_config "$config" \
  --prompt_file "$data/dev/prompts.json" \
  --output_dir "$build/evaluation/preflight" \
  --max_new_tokens 256 --max_context 2048 --seed 8172026 \
  --preflight_only

python - "$data/train/massive_en_10pct_structured" <<'PY'
import sys
from datasets import load_from_disk
from transformers import PreTrainedTokenizerFast
from train_sft import _audit_completion_templates, format_prompt_completion_example

dataset = load_from_disk(sys.argv[1])
formatted = dataset.map(
    format_prompt_completion_example,
    remove_columns=dataset.column_names,
    keep_in_memory=True,
)
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    revision="bb46c15ee4bb56c5b63245ef50fd7637234d6f75",
)
audit = _audit_completion_templates(formatted, tokenizer, max_length=1024)
audit.pop("_completion_tokens_by_example")
print(audit)
PY

python scripts/audit_massive_benefit_tillicum_workflow.py write-prep \
  --repo-root "$repo" --data-root "$data" --training-config "$config" \
  --output-file "$control/PREP_COMPLETE.json"

bash -n scripts/stage_massive_benefit_pilot_tillicum.sh
bash -n scripts/submit_massive_benefit_pilot_tillicum.sh
bash -n scripts/status_massive_benefit_pilot_tillicum.sh
bash -n scripts/sbatch_massive_benefit_base_dev_tillicum_h200.sbatch
bash -n scripts/sbatch_massive_benefit_train_tillicum_h200.sbatch
bash -n scripts/sbatch_massive_benefit_evaluate_tillicum_h200.sbatch
python -m py_compile \
  train_sft.py \
  scripts/train_single_sft.py \
  scripts/audit_massive_benefit_tillicum_workflow.py \
  scripts/prepare_massive_benefit_pilot_data.py \
  scripts/sample_massive_structured_generations.py \
  scripts/evaluate_massive_benefit_generations.py \
  scripts/summarize_massive_benefit_pilot.py
python -m unittest \
  tests.test_completion_only_sft \
  tests.test_massive_benefit_pilot \
  tests.test_massive_benefit_tillicum_workflow

rm "$control/STAGING_IN_PROGRESS"
printf 'repo_commit=%s\ncompleted_at=%s\nnon_gpu_stage=true\n' \
  "$expected_commit" "$(date --iso-8601=seconds)" > "$control/STAGED"
mv "$build" "$output"
echo "Fresh MASSIVE staging root: $output"
REMOTE

echo "Tillicum staging complete. No Slurm job was submitted."
echo 'MASSIVE-only hard maximum: 195 H200-minutes = $2.925 ($2.93 displayed).'
echo "Explicit submit command (not run):"
echo "  ssh $TILLICUM_HOST 'cd $REMOTE_REPO_ROOT && scripts/submit_massive_benefit_pilot_tillicum.sh pilot --ack-max-cost-usd 2.93'"
