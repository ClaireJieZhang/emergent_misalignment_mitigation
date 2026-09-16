#!/bin/bash
# CPU-only staging for the A/B1 Kalai k=2,s=1,R=20 diagnostic.

set -euo pipefail
umask 077
ulimit -c 0

[[ $# -eq 0 ]] || { echo 'Usage: stage_massive_medical_kalai_k2_s1_v1_tillicum.sh' >&2; exit 2; }

local_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
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
repo=$root/projects/subliminal-mitigate-mmu-panel-diagnostics-v1
output=$root/outputs/massive_medical_kalai_k2_s1_r20_v1
control=$output/control
source_protocol=$root/outputs/massive_medical_union_composition_exploratory_sequential_confirmation_v1_submit_recovery_v3/protocol/manifest.json
logs=$root/outputs/logs
env_root=$root/envs/subliminal-mitigate-py311

test -s "$source_protocol"
test ! -e "$repo"
test ! -e "$output"
if compgen -G "$logs/massive_medical_kalai_k2_s1_smoke_v1_*" >/dev/null; then
  echo 'Kalai k2s1 smoke log namespace is not fresh.' >&2
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
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-k2s1-stage-pyc
export DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export XDG_CACHE_HOME=$root/cache XDG_CONFIG_HOME=$root/config TMPDIR=$root/tmp
export HF_HOME=$root/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

cd "$repo"
python -m pip check
bash -n \
  scripts/stage_massive_medical_kalai_k2_s1_v1_tillicum.sh \
  scripts/submit_massive_medical_kalai_k2_s1_smoke_v1_tillicum.sh \
  scripts/sbatch_massive_medical_kalai_k2_s1_smoke_v1_tillicum_h200.sbatch \
  scripts/sbatch_massive_medical_kalai_k2_s1_full_v1_tillicum_h200.sbatch
python -m py_compile \
  scripts/sample_massive_medical_whole_output_consensus_k2_s1_v1.py \
  tests/test_massive_medical_kalai_k2_s1_v1.py
python -m unittest \
  tests.test_massive_medical_whole_output_consensus_v1 \
  tests.test_massive_medical_kalai_k2_s1_v1
python scripts/sample_massive_medical_whole_output_consensus_k2_s1_v1.py --self-test

mkdir -p "$control"
for stage in smoke full; do
  for phase in benefit medical; do
    python scripts/sample_massive_medical_whole_output_consensus_k2_s1_v1.py \
      --source-protocol-manifest "$source_protocol" \
      --output-root "$output/generation" \
      --phase "$phase" \
      --stage "$stage" \
      --preflight-only
  done
done

stage_tmp=$control/CPU_STAGE.tmp.$$
printf 'protocol_id=massive_medical_kalai_k2_s1_r20_v1\nstatus=CPU_STAGED_NO_GPU_OR_API_AUTHORITY\nrepository_commit=%s\nsource_protocol_path=%s\nsource_protocol_file_sha256=%s\nsmoke_benefit_requests=2\nsmoke_medical_requests=16\nsmoke_maximum_candidate_attempts=360\nsmoke_authorized=false\nfull_run_authorized=false\nexternal_api_calls_authorized=0\n' \
  "$expected" "$source_protocol" "$(sha256sum "$source_protocol" | awk '{print $1}')" > "$stage_tmp"
chmod 0400 "$stage_tmp"
mv "$stage_tmp" "$control/CPU_STAGE"

test ! -e "$control/SMOKE_AUTHORIZATION"
test ! -e "$control/SMOKE_SUBMISSION_LOCK"
test ! -e "$control/SMOKE_SUBMITTED"
test ! -e "$control/SMOKE_RELEASE_AUTHORIZED"
test ! -e "$control/SMOKE_RELEASED"
test ! -e "$control/SMOKE_COMPLETE"
test ! -e "$control/SMOKE_STOPPED"
test ! -e "$output/generation"
test -z "$(git -C "$repo" status --porcelain)"
echo MASSIVE_MEDICAL_KALAI_K2_S1_R20_V1_CPU_STAGED_NO_GPU_OR_API_AUTHORITY
REMOTE

echo 'MASSIVE/medical Kalai k2s1 CPU staging completed.'
echo 'No Slurm job, GPU allocation, model load, API call, or authorization occurred.'
