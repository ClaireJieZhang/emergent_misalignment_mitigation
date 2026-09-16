#!/bin/bash
# CPU-only staging for the two fixed MASSIVE/medical panel merge diagnostics.

set -euo pipefail
umask 077
ulimit -c 0

[[ $# -eq 0 ]] || {
  echo 'Usage: stage_massive_medical_panel_merge_diagnostic_v1_tillicum.sh' >&2
  exit 2
}

local_repo=$(cd "$(dirname "$0")/.." && pwd -P)
host=${TILLICUM_HOST:-tillicum}
root=${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}
url=${REMOTE_REPO_URL:-https://github.com/ClaireJieZhang/emergent_misalignment_mitigation.git}
branch=${REMOTE_BRANCH:-claire/massive-medical-panel-diagnostics-v1}
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
repo=$root/projects/subliminal-mitigate-mmu-panel-merge-diagnostic-v1
output=$root/outputs/massive_medical_panel_merge_diagnostic_v1
source_policy=$root/outputs/massive_medical_composition_baselines_v1/control/MERGE_POLICY.json
source_protocol=$root/outputs/massive_medical_union_composition_exploratory_sequential_confirmation_v1_submit_recovery_v3/protocol/manifest.json
env_root=$root/envs/subliminal-mitigate-py311

test ! -e "$repo"
test ! -e "$output"
test -s "$source_policy"
test -s "$source_protocol"

git clone --branch "$branch" --single-branch "$url" "$repo"
test "$(git -C "$repo" rev-parse HEAD)" = "$expected"
test -z "$(git -C "$repo" status --porcelain)"

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY
unset CUDA_VISIBLE_DEVICES TRANSFORMERS_CACHE
module load conda/Miniforge3-25.3.1-3
conda activate "$env_root"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-panel-merge-diagnostic-stage-pyc
export DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export XDG_CACHE_HOME=$root/cache XDG_CONFIG_HOME=$root/config TMPDIR=$root/tmp
export HF_HOME=$root/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

cd "$repo"
python -m pip check
bash -n \
  scripts/stage_massive_medical_panel_merge_diagnostic_v1_tillicum.sh \
  scripts/submit_massive_medical_panel_merge_diagnostic_v1_tillicum.sh \
  scripts/sbatch_massive_medical_panel_merge_diagnostic_v1_tillicum_h200.sbatch
python -m py_compile scripts/run_massive_medical_panel_merge_diagnostic_v1.py
python -m unittest tests.test_massive_medical_panel_merge_diagnostic_v1
python scripts/run_massive_medical_panel_merge_diagnostic_v1.py --self-test

python scripts/run_massive_medical_panel_merge_diagnostic_v1.py \
  --write-plan \
  --source-merge-policy "$source_policy" \
  --source-protocol-manifest "$source_protocol" \
  --output-root "$output" \
  --repo-root "$repo"
python scripts/run_massive_medical_panel_merge_diagnostic_v1.py \
  --preflight-only \
  --plan "$output/control/PLAN.json"

test ! -e "$output/control/EXECUTION_STARTED.json"
test ! -e "$output/control/EXECUTION_COMPLETE.json"
test ! -e "$output/control/GPU_RESULT"
test ! -e "$output/control/GPU_STOPPED"
test -z "$(git status --porcelain)"
REMOTE

echo 'MASSIVE/medical panel merge diagnostic v1 CPU stage completed.'
echo 'No Slurm job, GPU allocation, model load, generation, model-network call, or external API call was executed.'
