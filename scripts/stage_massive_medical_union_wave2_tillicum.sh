#!/bin/bash
# CPU-only staging of Wave 2 and the prospective, still-unreleased Wave-3 protocol.

set -euo pipefail
umask 077

[[ $# -eq 0 ]] || {
  echo 'Usage: scripts/stage_massive_medical_union_wave2_tillicum.sh' >&2
  exit 2
}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
TILLICUM_HOST=${TILLICUM_HOST:-tillicum}
TILLICUM_ROOT=${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}
REMOTE_REPO_URL=${REMOTE_REPO_URL:-https://github.com/ClaireJieZhang/emergent_misalignment_mitigation.git}
REMOTE_BRANCH=${REMOTE_BRANCH:-claire/capability-quorum-secure-code}
expected_commit=$(git -C "$repo_root" rev-parse HEAD)

required_files=(
  scripts/audit_massive_medical_union_wave2.py
  scripts/merge_massive_union_wave2_medical_judgments.py
  scripts/sbatch_massive_medical_union_wave2_train_tillicum_h200.sbatch
  scripts/sbatch_massive_medical_union_wave2_evaluate_tillicum_h200.sbatch
  scripts/stage_massive_medical_union_wave2_tillicum.sh
  scripts/submit_massive_medical_union_wave2_tillicum.sh
  scripts/status_massive_medical_union_wave2_tillicum.sh
  scripts/finalize_massive_medical_union_wave2_tillicum.sh
  tests/test_massive_medical_union_wave2.py
  docs/massive_medical_union_wave2_protocol.md
  scripts/prepare_massive_medical_union_wave3_protocol.py
  scripts/audit_massive_medical_union_wave3_protocol.py
  tests/test_massive_medical_union_wave3_protocol.py
  docs/massive_medical_union_wave3_composition_protocol.md
)
for path in "${required_files[@]}"; do
  test -s "$repo_root/$path"
  git -C "$repo_root" cat-file -e "$expected_commit:$path"
done
test -z "$(git -C "$repo_root" status --porcelain -- "${required_files[@]}")" || {
  echo 'Wave-2/preregistration workflow differs from committed HEAD.' >&2
  git -C "$repo_root" status --short -- "${required_files[@]}" >&2
  exit 3
}

ssh "$TILLICUM_HOST" bash -s -- \
  "$TILLICUM_ROOT" "$REMOTE_REPO_URL" "$REMOTE_BRANCH" "$expected_commit" <<'REMOTE'
set -euo pipefail
umask 077
ulimit -c 0

root=$1
repo_url=$2
branch=$3
expected_commit=$4
main_repo=$root/projects/subliminal-mitigate
recovery_v1_repo=$root/projects/subliminal-mitigate-mmu-medical-recovery-v1
recovery_v2_repo=$root/projects/subliminal-mitigate-mmu-medical-recovery-v2
wave2_repo=$root/projects/subliminal-mitigate-mmu-wave2
stage_repo=$root/projects/.subliminal-mitigate-mmu-wave2-stage-$$
output=$root/outputs/massive_medical_union_pilot_v1
control=$output/control/wave2
evaluation=$output/evaluation/wave2
protocol=$output/protocol/wave3_composition_v1
protocol_stage=$output/protocol/.wave3_composition_v1-stage-$$
massive_data=$root/outputs/massive_benefit_pilot_v1/data
union_data=$output/data
model_root=$output/models
env_root=$root/envs/subliminal-mitigate-py311

test "$(git -C "$main_repo" rev-parse HEAD)" = e25d59d8c5ea30c49cec207f5cac140a2281a525
test -z "$(git -C "$main_repo" status --porcelain)"
test "$(git -C "$recovery_v1_repo" rev-parse HEAD)" = 9ddd4816dafeb9b3df709e6ac72f41ebb22ee49f
test -z "$(git -C "$recovery_v1_repo" status --porcelain)"
test "$(git -C "$recovery_v2_repo" rev-parse HEAD)" = b704a00918bc7a9ddabc512795de4f81d3934c1a
test -z "$(git -C "$recovery_v2_repo" status --porcelain)"
test -s "$output/control/medical_recovery_v2/GO_MASSIVE_UNION_WAVE1"
test ! -e "$wave2_repo"
test ! -e "$stage_repo"
test ! -e "$control"
test ! -e "$evaluation"
test ! -e "$protocol"
test ! -e "$protocol_stage"
test ! -e "$model_root/pi_B2"
test ! -e "$model_root/pi_B3"
test -f "$env_root/.ready"

preserve_failed_stage() {
  status=$?
  if (( status != 0 )); then
    stamp=$(date +%Y%m%dT%H%M%S)
    [[ ! -e "$stage_repo" ]] || mv "$stage_repo" "$root/projects/subliminal-mitigate-mmu-wave2.failed-cpu-stage-$stamp"
    [[ ! -e "$protocol_stage" ]] || mv "$protocol_stage" "$output/protocol/wave3_composition_v1.failed-cpu-stage-$stamp"
  fi
  exit "$status"
}
trap preserve_failed_stage EXIT

git clone --branch "$branch" --single-branch "$repo_url" "$stage_repo"
test "$(git -C "$stage_repo" rev-parse HEAD)" = "$expected_commit"

# The fail-closed staging umask intentionally protects newly created control
# artifacts, but Git also applies it to checked-out executable files. Restore
# only the ten executable paths whose modes are committed as 100755; this does
# not alter tracked content or make the checkout dirty.
executable_files=(
  scripts/audit_massive_medical_union_wave2.py
  scripts/merge_massive_union_wave2_medical_judgments.py
  scripts/prepare_massive_medical_union_wave3_protocol.py
  scripts/audit_massive_medical_union_wave3_protocol.py
  scripts/sbatch_massive_medical_union_wave2_train_tillicum_h200.sbatch
  scripts/sbatch_massive_medical_union_wave2_evaluate_tillicum_h200.sbatch
  scripts/stage_massive_medical_union_wave2_tillicum.sh
  scripts/submit_massive_medical_union_wave2_tillicum.sh
  scripts/status_massive_medical_union_wave2_tillicum.sh
  scripts/finalize_massive_medical_union_wave2_tillicum.sh
)
for path in "${executable_files[@]}"; do
  entry=$(git -C "$stage_repo" ls-files --stage -- "$path")
  [[ "$entry" == 100755\ * ]] || {
    echo "Committed executable mode differs: $path" >&2
    exit 4
  }
  chmod 0755 "$stage_repo/$path"
done
test -z "$(git -C "$stage_repo" status --porcelain)"

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY
module load conda/Miniforge3-25.3.1-3
conda activate "$env_root"
export PYTHONUNBUFFERED=1
export DO_NOT_TRACK=1
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-wave2-stage-pyc

cd "$stage_repo"
python -m pip check
bash -n \
  scripts/stage_massive_medical_union_wave2_tillicum.sh \
  scripts/submit_massive_medical_union_wave2_tillicum.sh \
  scripts/status_massive_medical_union_wave2_tillicum.sh \
  scripts/finalize_massive_medical_union_wave2_tillicum.sh \
  scripts/sbatch_massive_medical_union_wave2_train_tillicum_h200.sbatch \
  scripts/sbatch_massive_medical_union_wave2_evaluate_tillicum_h200.sbatch
python -m py_compile \
  scripts/audit_massive_medical_union_wave2.py \
  scripts/merge_massive_union_wave2_medical_judgments.py \
  scripts/prepare_massive_medical_union_wave3_protocol.py \
  scripts/audit_massive_medical_union_wave3_protocol.py
python -m unittest \
  tests.test_massive_medical_union_wave2 \
  tests.test_massive_medical_union_wave3_protocol \
  tests.test_massive_union_component_evaluation

# Materialize and audit the prospective methods/subsets/gates before any B2/B3
# model exists.  This creates no job, model output, API call, or Wave-3 release.
python scripts/prepare_massive_medical_union_wave3_protocol.py \
  --massive-data-root "$massive_data" --union-data-root "$union_data" \
  --output-root "$protocol_stage"
python scripts/audit_massive_medical_union_wave3_protocol.py \
  --protocol-root "$protocol_stage" --massive-data-root "$massive_data" \
  --union-data-root "$union_data"

mv "$stage_repo" "$wave2_repo"
mv "$protocol_stage" "$protocol"
cd "$wave2_repo"
python scripts/audit_massive_medical_union_wave3_protocol.py \
  --protocol-root "$protocol" --massive-data-root "$massive_data" \
  --union-data-root "$union_data"
python scripts/audit_massive_medical_union_wave2.py write-prep
trap - EXIT
REMOTE

echo 'Wave 2 staged and Wave-3 protocol prospectively sealed. No job/API call was made.'
echo 'Explicit Wave-2 submit command (not run by staging):'
echo "  ssh $TILLICUM_HOST 'cd $TILLICUM_ROOT/projects/subliminal-mitigate-mmu-wave2 && scripts/submit_massive_medical_union_wave2_tillicum.sh wave2 --ack-max-cost-usd 1.125'"
