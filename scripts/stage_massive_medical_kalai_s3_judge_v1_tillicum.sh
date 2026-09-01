#!/bin/bash
# CPU-only staging for the exact-N Kalai s=3 external judge.

set -euo pipefail
umask 077
ulimit -c 0

[[ $# -eq 0 ]] || { echo 'Usage: stage_massive_medical_kalai_s3_judge_v1_tillicum.sh' >&2; exit 2; }
[[ -z ${OPENAI_API_KEY:-} ]] || { echo 'OPENAI_API_KEY must be absent during CPU staging.' >&2; exit 6; }

local_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
host=${TILLICUM_HOST:-tillicum}
root=${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}
url=${REMOTE_REPO_URL:-https://github.com/ClaireJieZhang/emergent_misalignment_mitigation.git}
branch=${REMOTE_BRANCH:-claire/massive-medical-kalai-s3-judge-v1}
commit=$(git -C "$local_repo" rev-parse HEAD)
test "$(git -C "$local_repo" branch --show-current)" = "$branch"
test -z "$(git -C "$local_repo" status --porcelain)"

ssh "$host" bash -s -- "$root" "$url" "$branch" "$commit" <<'REMOTE'
set -euo pipefail
umask 077
ulimit -c 0

root=$1; url=$2; branch=$3; expected=$4
repo=$root/projects/subliminal-mitigate-mmu-kalai-s3-r20-v2-judge-v1
plan_root=$root/outputs/massive_medical_kalai_s3_r20_v2_judge_plan_v1
output=$root/outputs/massive_medical_kalai_s3_r20_v2_kalai_s3_judge_v1
gate_output=$root/outputs/massive_medical_kalai_s3_r20_v2_submit_recovery_v3
completion_result=$gate_output/control/COMPLETION_RESULT.json
source_protocol=$root/outputs/massive_medical_union_composition_exploratory_sequential_confirmation_v1_submit_recovery_v3/protocol
prompt_file=$source_protocol/medical/prompts.json
env_root=$root/envs/subliminal-mitigate-py311

[[ -z ${OPENAI_API_KEY:-} ]] || { echo 'OPENAI_API_KEY must be absent during CPU staging.' >&2; exit 6; }
test -s "$completion_result"; test ! -L "$completion_result"
test -s "$gate_output/control/ASSEMBLY.json"
test -s "$gate_output/assembled/medical/generation.json"
test -s "$prompt_file"; test ! -L "$prompt_file"
test ! -e "$repo"; test ! -e "$plan_root"; test ! -e "$output"

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
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-kalai-s3-judge-v1-pyc
export OPENAI_LOG=off DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1

cd "$repo"
plan=$plan_root/JUDGE_PLAN.json
runner=scripts/judge_massive_medical_kalai_s3_split_v1.py
manifest=$output/control/JUDGE_STAGE_MANIFEST.json

python -m pip check
bash -n \
  scripts/stage_massive_medical_kalai_s3_judge_v1_tillicum.sh \
  scripts/finalize_massive_medical_kalai_s3_judge_v1_tillicum.sh \
  scripts/status_massive_medical_kalai_s3_judge_v1_tillicum.sh
python -m py_compile \
  scripts/prepare_massive_medical_kalai_s3_judge_plan_v1.py \
  scripts/judge_massive_medical_kalai_s3_split_v1.py \
  scripts/judge_massive_medical_composition_contextual_baselines_split_v1.py
python -m unittest tests.test_massive_medical_kalai_s3_judge_v1
python scripts/prepare_massive_medical_kalai_s3_judge_plan_v1.py --self-test
python scripts/prepare_massive_medical_kalai_s3_judge_plan_v1.py \
  --completion-result "$completion_result" \
  --prompt-file "$prompt_file" \
  --output-file "$plan"
python "$runner" prepare \
  --judge-plan "$plan" \
  --output-root "$output" \
  --repo-root "$repo"
python "$runner" validate-plan --manifest "$manifest"
python "$runner" validate-sdk-serialization --manifest "$manifest"
python "$runner" seal-staged --manifest "$manifest" \
  --validation-command 'Python compile and shell syntax checks' \
  --validation-command 'focused dynamic-N split-judge tests' \
  --validation-command 'sealed completion/assembly/source round-trip' \
  --validation-command 'three-range offline fake-client serialization'
python "$runner" audit-staged --manifest "$manifest"
test -z "$(git -C "$repo" status --porcelain)"
test ! -e "$output/control/CANARY_AUTHORIZATION.json"
test ! -e "$output/control/CANARY_RUN_STARTED.json"
test ! -e "$output/control/CONTINUATION_AUTHORIZATION.json"
test ! -e "$output/control/CONTINUATION_RUN_STARTED.json"
REMOTE

echo KALAI_S3_JUDGE_V1_CPU_STAGED_NO_API_OR_GPU_AUTHORITY
