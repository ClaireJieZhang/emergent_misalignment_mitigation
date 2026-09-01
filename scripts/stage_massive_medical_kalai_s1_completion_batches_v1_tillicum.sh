#!/bin/bash
# CPU-only remote staging for the versioned Kalai s=1 seven-batch controller.

set -euo pipefail
umask 077
ulimit -c 0

[[ $# -eq 0 ]] || {
  echo 'Usage: stage_massive_medical_kalai_s1_completion_batches_v1_tillicum.sh' >&2
  exit 2
}

local_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
host=${TILLICUM_HOST:-tillicum}
root=${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}
url=${REMOTE_REPO_URL:-https://github.com/ArtinTD/subliminal-mitigate.git}
branch=${REMOTE_BRANCH:-claire/massive-medical-kalai-s1-completion-batches-v1}
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
repo=$root/projects/subliminal-mitigate-mmu-kalai-s1-completion-batches-v1
output=$root/outputs/massive_medical_kalai_s1_completion_batches_v1
source_repo=$root/projects/subliminal-mitigate-mmu-kalai-s1-r20-trace-reuse-v1
source_output=$root/outputs/massive_medical_kalai_s1_r20_trace_reuse_v1
logs=$root/outputs/logs
env_root=$root/envs/subliminal-mitigate-py311

test -d "$source_repo/.git"
test "$(git -C "$source_repo" rev-parse HEAD)" = e83d59e3162250d7c3f555dc32f13de6d1ee9669
test -z "$(git -C "$source_repo" status --porcelain)"
test -s "$source_output/control/REPLAY_PLAN.json"
test -s "$source_output/control/CPU_STAGE.json"
test -s "$source_output/control/TECHNICAL_GATE_RESULT.json"
test -s "$source_output/generation/technical_gate/combined_timing.json"
test -s "$source_output/generation/technical_gate/benefit/generation.json"
test -s "$source_output/generation/technical_gate/benefit/timing.json"
test -s "$source_output/generation/technical_gate/medical/generation.json"
test -s "$source_output/generation/technical_gate/medical/timing.json"
test ! -e "$source_output/control/COMPLETION_AUTHORIZATION.json"
test ! -e "$source_output/control/COMPLETION_SUBMISSION_LOCK"
test ! -e "$source_output/generation/completion"
test ! -e "$source_output/assembled/completion"
test ! -e "$repo"
test ! -e "$output"
if compgen -G "$logs/massive_medical_kalai_s1_completion_batch_*" >/dev/null; then
  echo 'Kalai s=1 completion-batch log namespace is not fresh.' >&2
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
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-kalai-s1-completion-batches-stage-pyc
export DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export XDG_CACHE_HOME=$root/cache XDG_CONFIG_HOME=$root/config TMPDIR=$root/tmp
export HF_HOME=$root/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

cd "$repo"
python -m pip check
bash -n \
  scripts/stage_massive_medical_kalai_s1_completion_batches_v1_tillicum.sh \
  scripts/submit_massive_medical_kalai_s1_completion_batch_v1_tillicum.sh \
  scripts/sbatch_massive_medical_kalai_s1_completion_batch_v1_tillicum_h200.sbatch
python -m py_compile \
  scripts/prepare_massive_medical_kalai_s1_completion_batches_v1.py \
  scripts/prepare_massive_medical_kalai_s1_completion_batches_stage_v1.py \
  scripts/sample_massive_medical_kalai_s1_completion_batch_v1.py \
  scripts/authorize_massive_medical_kalai_s1_completion_batch_v1.py \
  scripts/evaluate_massive_medical_kalai_s1_completion_batch_v1.py \
  scripts/assemble_massive_medical_kalai_s1_completion_batches_v1.py \
  scripts/prepare_massive_medical_kalai_s1_trace_reuse_v1.py \
  scripts/evaluate_massive_medical_kalai_s1_trace_reuse_gate_v1.py
python -m unittest \
  tests.test_massive_medical_whole_output_consensus_v1 \
  tests.test_massive_medical_kalai_s3_v2 \
  tests.test_massive_medical_kalai_s1_trace_reuse_v1 \
  tests.test_massive_medical_kalai_s1_trace_reuse_controller_v1 \
  tests.test_massive_medical_kalai_s1_trace_reuse_workflow_v1 \
  tests.test_massive_medical_kalai_s1_completion_batches_v1 \
  tests.test_massive_medical_kalai_s1_completion_batch_runtime_v1 \
  tests.test_massive_medical_kalai_s1_completion_assembly_v1
python scripts/prepare_massive_medical_kalai_s1_completion_batches_v1.py --self-test
python scripts/prepare_massive_medical_kalai_s1_completion_batches_stage_v1.py --self-test
python scripts/sample_massive_medical_kalai_s1_completion_batch_v1.py --self-test
python scripts/authorize_massive_medical_kalai_s1_completion_batch_v1.py self-test
python scripts/evaluate_massive_medical_kalai_s1_completion_batch_v1.py --self-test
python scripts/assemble_massive_medical_kalai_s1_completion_batches_v1.py --self-test

python scripts/prepare_massive_medical_kalai_s1_completion_batches_stage_v1.py \
  --source-output-root "$source_output" \
  --source-repo-root "$source_repo" \
  --output-root "$output" \
  --repo-root "$repo"
python scripts/prepare_massive_medical_kalai_s1_completion_batches_stage_v1.py \
  --source-output-root "$source_output" \
  --source-repo-root "$source_repo" \
  --output-root "$output" \
  --repo-root "$repo"
for batch_index in $(seq 1 7); do
  python scripts/sample_massive_medical_kalai_s1_completion_batch_v1.py \
    --batch-plan "$output/control/COMPLETION_BATCH_PLAN.json" \
    --output-root "$output" \
    --batch-index "$batch_index" \
    --preflight-only
done

test -s "$output/control/COMPLETION_BATCH_PLAN.json"
test -s "$output/control/CPU_STAGE.json"
test "$(find "$output/control" -mindepth 1 -maxdepth 1 -type f | wc -l)" = 2
test ! -e "$output/control/batches"
test ! -e "$output/generation"
test ! -e "$output/assembled"
test ! -e "$output/judge"
test -z "$(git -C "$repo" status --porcelain)"
test -z "$(git -C "$source_repo" status --porcelain)"
test ! -e "$source_output/control/COMPLETION_AUTHORIZATION.json"
test ! -e "$source_output/generation/completion"
if compgen -G "$logs/massive_medical_kalai_s1_completion_batch_*" >/dev/null; then
  echo 'CPU staging unexpectedly created a Slurm log.' >&2
  exit 6
fi
echo KALAI_S1_COMPLETION_BATCHES_V1_CPU_STAGED_NO_GPU_OR_API_AUTHORITY
REMOTE

echo 'MASSIVE/medical Kalai s=1 seven-batch CPU stage completed.'
echo 'No Slurm job, GPU allocation, model load, API call, or paid authorization was executed.'
