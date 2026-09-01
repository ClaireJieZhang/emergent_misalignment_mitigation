#!/bin/bash
# CPU-only remote staging for the bounded Kalai s=3 completion controller.

set -euo pipefail
umask 077
ulimit -c 0

[[ $# -eq 0 ]] || { echo 'Usage: stage_massive_medical_kalai_s3_completion_v1_tillicum.sh' >&2; exit 2; }

local_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
host=${TILLICUM_HOST:-tillicum}
root=${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}
url=${REMOTE_REPO_URL:-https://github.com/ClaireJieZhang/emergent_misalignment_mitigation.git}
branch=${REMOTE_BRANCH:-claire/massive-medical-kalai-s3-completion-controller-v1}
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
controller_repo=$root/projects/subliminal-mitigate-mmu-kalai-s3-r20-v2-completion-controller-v1
controller_output=$root/outputs/massive_medical_kalai_s3_r20_v2_completion_controller_v1
gate_repo=$root/projects/subliminal-mitigate-mmu-kalai-s3-r20-v2-submit-recovery-v3
gate_output=$root/outputs/massive_medical_kalai_s3_r20_v2_submit_recovery_v3
source_protocol=$root/outputs/massive_medical_union_composition_exploratory_sequential_confirmation_v1_submit_recovery_v3/protocol/manifest.json
logs=$root/outputs/logs
env_root=$root/envs/subliminal-mitigate-py311

test "$(git -C "$gate_repo" rev-parse HEAD)" = ed950b72396dc041d34bbb694ea1486763033657
test -z "$(git -C "$gate_repo" status --porcelain)"
test -s "$gate_output/control/CPU_STAGE.json"
test -s "$gate_output/control/GATE_PLAN.json"
test -s "$gate_output/control/GATE_RESULT.json"
test ! -e "$gate_output/control/COMPLETION_AUTHORIZATION.json"
test ! -e "$gate_output/control/COMPLETION_SUBMISSION_LOCK"
test ! -e "$gate_output/control/COMPLETION_SUBMISSION_ATTEMPT.tsv"
test ! -e "$gate_output/control/COMPLETION_SUBMITTED"
test ! -e "$gate_output/control/COMPLETION_RELEASE_AUTHORIZED"
test ! -e "$gate_output/control/COMPLETION_RELEASED"
test ! -e "$gate_output/control/COMPLETION_INVOCATION_LOCK"
test ! -e "$gate_output/control/COMPLETION_RESULT.json"
test ! -e "$gate_output/control/COMPLETION_STOPPED"
test ! -e "$gate_output/generation/completion"
test ! -e "$gate_output/assembled"
test ! -e "$controller_repo"
test ! -e "$controller_output"
if compgen -G "$logs/massive_medical_kalai_s3_r20_v2_completion_v1_*" >/dev/null; then
  echo 'Kalai completion log namespace is not fresh.' >&2
  exit 4
fi

git clone --branch "$branch" --single-branch "$url" "$controller_repo"
test "$(git -C "$controller_repo" rev-parse HEAD)" = "$expected"
while read -r mode object stage path; do
  test "$stage" = 0
  case "$mode" in
    100755) chmod 0755 "$controller_repo/$path" ;;
    100644) chmod 0644 "$controller_repo/$path" ;;
    *) echo "Unsupported tracked mode $mode for $path" >&2; exit 5 ;;
  esac
done < <(git -C "$controller_repo" ls-files -s)
test -z "$(git -C "$controller_repo" status --porcelain)"

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY
unset CUDA_VISIBLE_DEVICES TRANSFORMERS_CACHE
module load conda/Miniforge3-25.3.1-3
conda activate "$env_root"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-kalai-s3-completion-stage-v1-pyc
export DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export XDG_CACHE_HOME=$root/cache XDG_CONFIG_HOME=$root/config TMPDIR=$root/tmp
export HF_HOME=$root/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

cd "$controller_repo"
python -m pip check
bash -n \
  scripts/stage_massive_medical_kalai_s3_completion_v1_tillicum.sh \
  scripts/submit_massive_medical_kalai_s3_completion_v1_tillicum.sh \
  scripts/sbatch_massive_medical_kalai_s3_completion_v1_tillicum_h200.sbatch
python -m py_compile \
  scripts/sample_massive_medical_whole_output_consensus_v1.py \
  scripts/sample_massive_medical_whole_output_consensus_s3_v2.py \
  scripts/sample_massive_medical_kalai_s3_completion_combined_v1.py \
  scripts/prepare_massive_medical_kalai_s3_completion_v1.py \
  scripts/authorize_massive_medical_kalai_s3_completion_v1.py \
  scripts/evaluate_massive_medical_kalai_s3_completion_v1.py \
  scripts/assemble_massive_medical_kalai_s3_v2.py \
  subliminal_mitigate/decoding/algorithms.py
python -m unittest \
  tests.test_massive_medical_whole_output_consensus_v1 \
  tests.test_massive_medical_kalai_s3_v2 \
  tests.test_massive_medical_kalai_s3_completion_v1
python scripts/sample_massive_medical_kalai_s3_completion_combined_v1.py --self-test
python scripts/prepare_massive_medical_kalai_s3_completion_v1.py --self-test
python scripts/authorize_massive_medical_kalai_s3_completion_v1.py self-test
python scripts/evaluate_massive_medical_kalai_s3_completion_v1.py --self-test
python scripts/assemble_massive_medical_kalai_s3_v2.py --self-test
python scripts/sample_massive_medical_kalai_s3_completion_combined_v1.py \
  --source-protocol-manifest "$source_protocol" \
  --output-root "$gate_output/generation" \
  --preflight-only
python scripts/prepare_massive_medical_kalai_s3_completion_v1.py \
  --output-root "$controller_output" \
  --repo-root "$controller_repo" \
  --gate-output "$gate_output" \
  --gate-repo "$gate_repo"
python scripts/prepare_massive_medical_kalai_s3_completion_v1.py \
  --output-root "$controller_output" \
  --repo-root "$controller_repo" \
  --gate-output "$gate_output" \
  --gate-repo "$gate_repo" \
  --audit-only
test -z "$(git status --porcelain)"
test ! -e "$gate_output/control/COMPLETION_AUTHORIZATION.json"
test ! -e "$gate_output/control/COMPLETION_SUBMISSION_LOCK"
test ! -e "$gate_output/generation/completion"
test ! -e "$gate_output/assembled"
test ! -e "$controller_output/control/AUTHORIZATION.json"
echo KALAI_S3_COMPLETION_V1_CPU_STAGED_NO_GPU_OR_API_AUTHORITY
REMOTE
