#!/bin/bash
# CPU-only remote staging for the exact-paired Kalai s=1 trace-reuse arm.

set -euo pipefail
umask 077
ulimit -c 0

[[ $# -eq 0 ]] || { echo 'Usage: stage_massive_medical_kalai_s1_trace_reuse_v1_tillicum.sh' >&2; exit 2; }

local_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
host=${TILLICUM_HOST:-tillicum}
root=${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}
url=${REMOTE_REPO_URL:-https://github.com/ClaireJieZhang/emergent_misalignment_mitigation.git}
branch=${REMOTE_BRANCH:-claire/massive-medical-kalai-s1-trace-reuse-v1}
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
repo=$root/projects/subliminal-mitigate-mmu-kalai-s1-r20-trace-reuse-v1
output=$root/outputs/massive_medical_kalai_s1_r20_trace_reuse_v1
s3=$root/outputs/massive_medical_kalai_s3_r20_v2_submit_recovery_v3
smoke=$root/outputs/massive_medical_composition_baselines_v1
source_protocol=$root/outputs/massive_medical_union_composition_exploratory_sequential_confirmation_v1_submit_recovery_v3/protocol/manifest.json
logs=$root/outputs/logs
env_root=$root/envs/subliminal-mitigate-py311

test -s "$s3/control/ASSEMBLY.json"
test -s "$s3/control/COMPLETION_RESULT.json"
test -s "$s3/generation/gate/benefit/generation.json"
test -s "$s3/generation/completion/benefit/generation.json"
test -s "$s3/generation/gate/medical/generation.json"
test -s "$s3/generation/completion/medical/generation.json"
test -s "$smoke/generation/whole_output/smoke/medical/generation.json"
test -s "$source_protocol"
test ! -e "$repo"
test ! -e "$output"
if compgen -G "$logs/massive_medical_kalai_s1_r20_trace_reuse_v1_gate_*" >/dev/null; then
  echo 'Kalai s=1 log namespace is not fresh.' >&2
  exit 4
fi

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
export PYTHONPATH=$repo
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-kalai-s1-stage-pyc
export DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export XDG_CACHE_HOME=$root/cache XDG_CONFIG_HOME=$root/config TMPDIR=$root/tmp
export HF_HOME=$root/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

cd "$repo"
python -m pip check
bash -n \
  scripts/stage_massive_medical_kalai_s1_trace_reuse_v1_tillicum.sh \
  scripts/submit_massive_medical_kalai_s1_trace_reuse_gate_v1_tillicum.sh \
  scripts/sbatch_massive_medical_kalai_s1_trace_reuse_gate_v1_tillicum_h200.sbatch
python -m py_compile \
  scripts/prepare_massive_medical_kalai_s1_trace_reuse_v1.py \
  scripts/prepare_massive_medical_kalai_s1_trace_reuse_stage_v1.py \
  scripts/sample_massive_medical_kalai_s1_trace_reuse_v1.py \
  scripts/authorize_massive_medical_kalai_s1_trace_reuse_v1.py \
  scripts/evaluate_massive_medical_kalai_s1_trace_reuse_gate_v1.py
python -m unittest \
  tests.test_massive_medical_whole_output_consensus_v1 \
  tests.test_massive_medical_kalai_s3_v2 \
  tests.test_massive_medical_kalai_s1_trace_reuse_v1 \
  tests.test_massive_medical_kalai_s1_trace_reuse_controller_v1 \
  tests.test_massive_medical_kalai_s1_trace_reuse_workflow_v1
python scripts/prepare_massive_medical_kalai_s1_trace_reuse_stage_v1.py --self-test
python scripts/sample_massive_medical_kalai_s1_trace_reuse_v1.py --self-test
python scripts/authorize_massive_medical_kalai_s1_trace_reuse_v1.py self-test
python scripts/evaluate_massive_medical_kalai_s1_trace_reuse_gate_v1.py --self-test

python scripts/prepare_massive_medical_kalai_s1_trace_reuse_stage_v1.py \
  --source-protocol-manifest "$source_protocol" \
  --s3-benefit-gate-generation "$s3/generation/gate/benefit/generation.json" \
  --s3-benefit-completion-generation "$s3/generation/completion/benefit/generation.json" \
  --s3-medical-gate-generation "$s3/generation/gate/medical/generation.json" \
  --s3-medical-completion-generation "$s3/generation/completion/medical/generation.json" \
  --s1-medical-smoke-generation "$smoke/generation/whole_output/smoke/medical/generation.json" \
  --output-root "$output" \
  --repo-root "$repo"
python scripts/sample_massive_medical_kalai_s1_trace_reuse_v1.py \
  --replay-plan "$output/control/REPLAY_PLAN.json" \
  --output-root "$output" \
  --stage technical_gate \
  --preflight-only
python scripts/sample_massive_medical_kalai_s1_trace_reuse_v1.py \
  --replay-plan "$output/control/REPLAY_PLAN.json" \
  --output-root "$output" \
  --stage completion \
  --preflight-only

test -s "$output/control/REPLAY_PLAN.json"
test -s "$output/control/CPU_STAGE.json"
test ! -e "$output/control/TECHNICAL_GATE_AUTHORIZATION.json"
test ! -e "$output/control/TECHNICAL_GATE_SUBMISSION_LOCK"
test ! -e "$output/control/TECHNICAL_GATE_RESULT.json"
test ! -e "$output/control/COMPLETION_AUTHORIZATION.json"
test ! -e "$output/generation"
test ! -e "$output/assembled"
test -z "$(git -C "$repo" status --porcelain)"
echo KALAI_S1_TRACE_REUSE_V1_CPU_STAGED_NO_GPU_OR_API_AUTHORITY
REMOTE

echo 'MASSIVE/medical Kalai s=1 trace-reuse CPU stage completed.'
echo 'No Slurm job, GPU allocation, model load, API call, or authorization was executed.'
