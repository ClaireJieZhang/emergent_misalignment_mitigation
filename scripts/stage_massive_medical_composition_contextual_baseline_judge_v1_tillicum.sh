#!/bin/bash
# CPU-only staging for the fresh 160-row contextual-baseline judge.

set -euo pipefail
umask 077
ulimit -c 0

usage() {
  echo 'Usage: stage_massive_medical_composition_contextual_baseline_judge_v1_tillicum.sh --judge-plan ABSOLUTE_REMOTE_PLAN' >&2
  exit 2
}

[[ $# -eq 2 && $1 == --judge-plan && $2 == /* ]] || usage
judge_plan=$2
[[ -z ${OPENAI_API_KEY:-} ]] || {
  echo 'OPENAI_API_KEY must be absent during CPU staging.' >&2
  exit 6
}

local_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
host=${TILLICUM_HOST:-tillicum}
root=${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}
url=${REMOTE_REPO_URL:-https://github.com/ClaireJieZhang/emergent_misalignment_mitigation.git}
branch=$(git -C "$local_repo" branch --show-current)
commit=$(git -C "$local_repo" rev-parse HEAD)
[[ -n $branch ]]

ssh "$host" bash -s -- "$root" "$url" "$branch" "$commit" "$judge_plan" <<'REMOTE'
set -euo pipefail
umask 077
ulimit -c 0

root=$1; url=$2; branch=$3; expected=$4; judge_plan=$5
repo=$root/projects/subliminal-mitigate-mmu-composition-contextual-baseline-judge-v1
output=$root/outputs/massive_medical_composition_contextual_baseline_judge_v1
env_root=$root/envs/subliminal-mitigate-py311
export GIT_OPTIONAL_LOCKS=0

[[ -z ${OPENAI_API_KEY:-} ]] || {
  echo 'OPENAI_API_KEY must be absent during CPU staging.' >&2
  exit 6
}
test -f "$judge_plan"; test ! -L "$judge_plan"
test ! -e "$repo"; test ! -e "$output"

git clone --branch "$branch" --single-branch "$url" "$repo"
test "$(git -C "$repo" rev-parse HEAD)" = "$expected"
test -z "$(git -C "$repo" status --porcelain)"

unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN WANDB_API_KEY
unset ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY CUDA_VISIBLE_DEVICES
module load conda/Miniforge3-25.3.1-3
conda activate "$env_root"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/contextual-baseline-judge-v1-pyc
export OPENAI_LOG=off DO_NOT_TRACK=1

cd "$repo"
runner=scripts/judge_massive_medical_composition_contextual_baselines_split_v1.py
manifest=$output/control/JUDGE_STAGE_MANIFEST.json

python "$runner" prepare \
  --judge-plan "$judge_plan" \
  --output-root "$output" \
  --repo-root "$repo"
python -m py_compile "$runner"
bash -n \
  scripts/stage_massive_medical_composition_contextual_baseline_judge_v1_tillicum.sh \
  scripts/finalize_massive_medical_composition_contextual_baseline_judge_v1_tillicum.sh \
  scripts/status_massive_medical_composition_contextual_baseline_judge_v1_tillicum.sh
python -m unittest tests.test_massive_medical_composition_contextual_baseline_judge_v1
python "$runner" validate-plan --manifest "$manifest"
python "$runner" validate-sdk-serialization --manifest "$manifest"
python "$runner" seal-staged --manifest "$manifest" \
  --validation-command 'Python compile and shell syntax checks' \
  --validation-command 'focused split-judge workflow tests' \
  --validation-command 'exact sealed 160-row source reconstruction' \
  --validation-command 'three-range offline fake-client serialization'
python "$runner" audit-staged --manifest "$manifest"
test -z "$(git -C "$repo" status --porcelain)"
REMOTE

echo 'Contextual-baseline judge v1 CPU stage completed.'
echo 'No API call, API authorization, Slurm job, GPU allocation, or model load occurred.'
