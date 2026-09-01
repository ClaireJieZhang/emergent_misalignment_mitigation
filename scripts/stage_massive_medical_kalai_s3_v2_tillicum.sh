#!/bin/bash
# CPU-only remote staging for the versioned Kalai s=3, R=20 gate.

set -euo pipefail
umask 077
ulimit -c 0

[[ $# -eq 0 ]] || { echo 'Usage: stage_massive_medical_kalai_s3_v2_tillicum.sh' >&2; exit 2; }

local_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
host=${TILLICUM_HOST:-tillicum}
root=${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}
url=${REMOTE_REPO_URL:-https://github.com/ClaireJieZhang/emergent_misalignment_mitigation.git}
branch=${REMOTE_BRANCH:-claire/massive-medical-kalai-s3-gate-submit-recovery-v3}
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
repo=$root/projects/subliminal-mitigate-mmu-kalai-s3-r20-v2-submit-recovery-v3
output=$root/outputs/massive_medical_kalai_s3_r20_v2_submit_recovery_v3
source_protocol=$root/outputs/massive_medical_union_composition_exploratory_sequential_confirmation_v1_submit_recovery_v3/protocol/manifest.json
logs=$root/outputs/logs
env_root=$root/envs/subliminal-mitigate-py311
abandoned=$root/outputs/massive_medical_kalai_s3_r20_v2

test -f "$abandoned/control/CPU_STAGE.json"
test -f "$abandoned/control/GATE_PLAN.json"
test -f "$abandoned/control/GATE_SUBMISSION_LOCK/owner"
test ! -e "$abandoned/control/GATE_AUTHORIZATION.json"
test ! -e "$abandoned/control/GATE_SUBMISSION_ATTEMPT.tsv"
test ! -e "$abandoned/control/GATE_SUBMITTED"
test ! -e "$abandoned/control/GATE_RELEASE_AUTHORIZED"
test ! -e "$abandoned/control/GATE_RELEASED"
test ! -e "$abandoned/control/GATE_INVOCATION_LOCK"
test ! -e "$abandoned/control/GATE_RESULT.json"
test ! -e "$abandoned/control/GATE_STOPPED"
test ! -e "$abandoned/control/COMPLETION_AUTHORIZATION.json"
test ! -e "$abandoned/generation"
test ! -e "$abandoned/assembled"
test "$(find "$abandoned" -type f | wc -l)" -eq 3
if compgen -G "$logs/massive_medical_kalai_s3_r20_v2_gate_*" >/dev/null; then
  echo 'Abandoned Kalai v2 unexpectedly has a log artifact.' >&2
  exit 3
fi

test ! -e "$repo"
test ! -e "$output"
if compgen -G "$logs/massive_medical_kalai_s3_r20_v2_submit_recovery_v3_*" >/dev/null; then
  echo 'Kalai s=3 log namespace is not fresh.' >&2
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
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-kalai-s3-stage-pyc
export DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export XDG_CACHE_HOME=$root/cache XDG_CONFIG_HOME=$root/config TMPDIR=$root/tmp
export HF_HOME=$root/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

cd "$repo"
python -m pip check
bash -n \
  scripts/stage_massive_medical_kalai_s3_v2_tillicum.sh \
  scripts/submit_massive_medical_kalai_s3_gate_v2_tillicum.sh \
  scripts/sbatch_massive_medical_kalai_s3_gate_v2_tillicum_h200.sbatch
python -m py_compile \
  scripts/sample_massive_medical_whole_output_consensus_v1.py \
  scripts/sample_massive_medical_whole_output_consensus_s3_v2.py \
  scripts/prepare_massive_medical_kalai_s3_v2.py \
  scripts/authorize_massive_medical_kalai_s3_v2.py \
  scripts/evaluate_massive_medical_kalai_s3_gate_v2.py \
  scripts/assemble_massive_medical_kalai_s3_v2.py \
  subliminal_mitigate/decoding/algorithms.py
python -m unittest \
  tests.test_massive_medical_whole_output_consensus_v1 \
  tests.test_massive_medical_kalai_s3_v2
python scripts/sample_massive_medical_whole_output_consensus_v1.py --self-test
python scripts/sample_massive_medical_whole_output_consensus_s3_v2.py --self-test
python scripts/prepare_massive_medical_kalai_s3_v2.py --self-test
python scripts/authorize_massive_medical_kalai_s3_v2.py self-test
python scripts/evaluate_massive_medical_kalai_s3_gate_v2.py --self-test
python scripts/assemble_massive_medical_kalai_s3_v2.py --self-test

for stage in gate completion; do
  for phase in benefit medical; do
    unused=$root/tmp/mmu-kalai-s3-preflight-unused-$stage-$phase
    test ! -e "$unused"
    python scripts/sample_massive_medical_whole_output_consensus_s3_v2.py \
      --source-protocol-manifest "$source_protocol" \
      --output-root "$unused" \
      --phase "$phase" \
      --stage "$stage" \
      --preflight-only
    test ! -e "$unused"
  done
done

python scripts/prepare_massive_medical_kalai_s3_v2.py \
  --source-protocol-manifest "$source_protocol" \
  --output-root "$output" \
  --repo-root "$repo"

cmp -s \
  "$abandoned/control/GATE_PLAN.json" \
  "$output/control/GATE_PLAN.json"
test -f "$output/control/GATE_PLAN.json"
test -f "$output/control/CPU_STAGE.json"
test ! -e "$output/control/GATE_AUTHORIZATION.json"
test ! -e "$output/control/GATE_SUBMISSION_LOCK"
test ! -e "$output/control/GATE_SUBMISSION_ATTEMPT.tsv"
test ! -e "$output/control/GATE_SUBMITTED"
test ! -e "$output/control/GATE_RELEASE_AUTHORIZED"
test ! -e "$output/control/GATE_RELEASED"
test ! -e "$output/control/COMPLETION_AUTHORIZATION.json"
test ! -e "$output/control/GATE_RESULT.json"
test ! -e "$output/control/GATE_INVOCATION_LOCK"
test ! -e "$output/generation"
test ! -e "$output/assembled"
test -z "$(git -C "$repo" status --porcelain)"
REMOTE

echo 'MASSIVE/medical Kalai s=3, R=20 CPU stage completed.'
echo 'No Slurm job, GPU allocation, model load, API call, or authorization was executed.'
