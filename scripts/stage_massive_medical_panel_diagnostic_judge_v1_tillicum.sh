#!/bin/bash
# CPU-only staging for the sealed 161-call panel-diagnostic judge.

set -euo pipefail
umask 077
ulimit -c 0

[[ $# -eq 0 ]] || {
  echo 'Usage: stage_massive_medical_panel_diagnostic_judge_v1_tillicum.sh' >&2
  exit 2
}
[[ -z ${OPENAI_API_KEY:-} ]] || {
  echo 'OPENAI_API_KEY must be absent during CPU staging.' >&2
  exit 6
}

local_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
host=${TILLICUM_HOST:-tillicum}
root=${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}
url=${REMOTE_REPO_URL:-https://github.com/ClaireJieZhang/emergent_misalignment_mitigation.git}
branch=${REMOTE_BRANCH:-claire/massive-medical-panel-judge-v1}
commit=$(git -C "$local_repo" rev-parse HEAD)
test "$(git -C "$local_repo" branch --show-current)" = "$branch"
test -z "$(git -C "$local_repo" status --porcelain)"

ssh "$host" bash -s -- "$root" "$url" "$branch" "$commit" <<'REMOTE'
set -euo pipefail
umask 077
ulimit -c 0

root=$1; url=$2; branch=$3; expected=$4
repo=$root/projects/subliminal-mitigate-mmu-panel-diagnostic-judge-v1
plan_root=$root/outputs/massive_medical_panel_diagnostic_judge_plan_v1
output=$root/outputs/massive_medical_panel_diagnostic_judge_v1
merge_output=$root/outputs/massive_medical_panel_merge_diagnostic_v1
k2_smoke_output=$root/outputs/massive_medical_kalai_k2_s1_r20_v1
kalai_smoke_generation=$k2_smoke_output/generation/smoke/medical/generation.json
source_protocol=$root/outputs/massive_medical_union_composition_exploratory_sequential_confirmation_v1_submit_recovery_v3/protocol
source_protocol_manifest=$source_protocol/manifest.json
env_root=$root/envs/subliminal-mitigate-py311
export GIT_OPTIONAL_LOCKS=0

[[ -z ${OPENAI_API_KEY:-} ]] || {
  echo 'OPENAI_API_KEY must be absent during CPU staging.' >&2
  exit 6
}
test -d "$merge_output"; test ! -L "$merge_output"
test -d "$k2_smoke_output"; test ! -L "$k2_smoke_output"
test -s "$kalai_smoke_generation"; test ! -L "$kalai_smoke_generation"
test -s "$source_protocol_manifest"; test ! -L "$source_protocol_manifest"
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
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-panel-diagnostic-judge-v1-pyc
export OPENAI_LOG=off DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1

cd "$repo"
plan=$plan_root/JUDGE_PLAN.json
runner=scripts/judge_massive_medical_panel_diagnostic_split_v1.py
manifest=$output/control/JUDGE_STAGE_MANIFEST.json

python -m pip check
bash -n \
  scripts/stage_massive_medical_panel_diagnostic_judge_v1_tillicum.sh \
  scripts/finalize_massive_medical_panel_diagnostic_judge_v1_tillicum.sh \
  scripts/status_massive_medical_panel_diagnostic_judge_v1_tillicum.sh
python -m py_compile \
  scripts/prepare_massive_medical_panel_diagnostic_judge_plan_v1.py \
  scripts/judge_massive_medical_panel_diagnostic_split_v1.py
python -m unittest tests.test_massive_medical_panel_diagnostic_judge_v1
python scripts/prepare_massive_medical_panel_diagnostic_judge_plan_v1.py --self-test
python scripts/prepare_massive_medical_panel_diagnostic_judge_plan_v1.py \
  --source-protocol-manifest "$source_protocol_manifest" \
  --merge-output-root "$merge_output" \
  --kalai-smoke-generation "$kalai_smoke_generation" \
  --output-file "$plan"
python "$runner" prepare \
  --judge-plan "$plan" \
  --output-root "$output" \
  --repo-root "$repo"
python "$runner" validate-plan --manifest "$manifest"
python "$runner" validate-sdk-serialization --manifest "$manifest"
python "$runner" seal-staged --manifest "$manifest" \
  --validation-command 'Python compile, dependency, and shell syntax checks' \
  --validation-command 'focused 161-call split-judge workflow tests' \
  --validation-command 'sealed merge, k2-smoke, and prompt-source round-trip' \
  --validation-command 'three-range offline fake-client serialization'
python "$runner" audit-staged --manifest "$manifest"
test -z "$(git -C "$repo" status --porcelain)"
test ! -e "$output/control/CANARY_AUTHORIZATION.json"
test ! -e "$output/control/CANARY_RUN_STARTED.json"
test ! -e "$output/control/CONTINUATION_AUTHORIZATION.json"
test ! -e "$output/control/CONTINUATION_RUN_STARTED.json"
REMOTE

echo PANEL_DIAGNOSTIC_JUDGE_V1_CPU_STAGED_NO_API_OR_GPU_AUTHORITY
