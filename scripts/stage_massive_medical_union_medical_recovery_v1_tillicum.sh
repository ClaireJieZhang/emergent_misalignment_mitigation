#!/bin/bash
# CPU-only, exact-once staging for the Wave-1 medical generation recovery.

set -euo pipefail
umask 077

[[ $# -eq 0 ]] || {
  echo 'Usage: scripts/stage_massive_medical_union_medical_recovery_v1_tillicum.sh' >&2
  exit 2
}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
TILLICUM_HOST=${TILLICUM_HOST:-tillicum}
TILLICUM_ROOT=${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}
REMOTE_REPO_URL=${REMOTE_REPO_URL:-https://github.com/ClaireJieZhang/emergent_misalignment_mitigation.git}
REMOTE_BRANCH=${REMOTE_BRANCH:-claire/capability-quorum-secure-code}
expected_commit=$(git -C "$repo_root" rev-parse HEAD)

required_files=(
  docs/massive_medical_union_pilot_protocol.md
  scripts/audit_massive_medical_union_medical_recovery_v1.py
  scripts/finalize_massive_medical_union_wave1_medical_recovery_v1_tillicum.sh
  scripts/judge_massive_union_medical.py
  scripts/sample_massive_union_medical_direct.py
  scripts/sbatch_massive_medical_union_medical_recovery_v1_tillicum_h200.sbatch
  scripts/stage_massive_medical_union_medical_recovery_v1_tillicum.sh
  scripts/status_massive_medical_union_medical_recovery_v1_tillicum.sh
  scripts/submit_massive_medical_union_medical_recovery_v1_tillicum.sh
  scripts/summarize_massive_union_components.py
  tests/test_massive_medical_union_medical_recovery_v1.py
  tests/test_massive_union_component_evaluation.py
)
for path in "${required_files[@]}"; do
  test -s "$repo_root/$path"
  git -C "$repo_root" cat-file -e "$expected_commit:$path"
done
test -z "$(git -C "$repo_root" status --porcelain -- "${required_files[@]}")" || {
  echo 'Medical-recovery workflow differs from committed HEAD.' >&2
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
recovery_repo=$root/projects/subliminal-mitigate-mmu-medical-recovery-v1
output=$root/outputs/massive_medical_union_pilot_v1
recovery_control=$output/control/medical_recovery_v1
recovery_eval=$output/evaluation/wave1/medical_recovery_v1
env_root=$root/envs/subliminal-mitigate-py311

test "$(git -C "$main_repo" rev-parse HEAD)" = e25d59d8c5ea30c49cec207f5cac140a2281a525
test -z "$(git -C "$main_repo" status --porcelain)"
test ! -e "$recovery_repo"
test ! -e "$recovery_control"
test ! -e "$recovery_eval"
test -f "$env_root/.ready"

git clone --branch "$branch" --single-branch "$repo_url" "$recovery_repo"
test "$(git -C "$recovery_repo" rev-parse HEAD)" = "$expected_commit"
test -z "$(git -C "$recovery_repo" status --porcelain)"

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
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-medical-recovery-v1-stage-pyc

cd "$recovery_repo"
python -m pip check
bash -n \
  scripts/stage_massive_medical_union_medical_recovery_v1_tillicum.sh \
  scripts/submit_massive_medical_union_medical_recovery_v1_tillicum.sh \
  scripts/status_massive_medical_union_medical_recovery_v1_tillicum.sh \
  scripts/finalize_massive_medical_union_wave1_medical_recovery_v1_tillicum.sh \
  scripts/sbatch_massive_medical_union_medical_recovery_v1_tillicum_h200.sbatch
python -m py_compile \
  scripts/audit_massive_medical_union_medical_recovery_v1.py \
  scripts/sample_massive_union_medical_direct.py \
  scripts/judge_massive_union_medical.py \
  scripts/summarize_massive_union_components.py
python -m unittest \
  tests.test_massive_union_component_evaluation \
  tests.test_massive_medical_union_medical_recovery_v1
python scripts/sample_massive_union_medical_direct.py \
  --model pi_base=BASE \
  --training_config configs/training_qwen25_7b_massive_medical_union_pilot.yaml \
  --data_manifest "$output/data/data_manifest.json" \
  --prompt_file "$output/data/medical_eval/official16.json" \
  --output_dir "$root/tmp/mmu-medical-recovery-v1-preflight" \
  --sampling_profile official16_max1024_all_stop_v2 --preflight_only

# This is the first creation of the recovery control namespace.  The auditor
# verifies the exact old inventory and that both recovery namespaces are absent
# before it creates and seals PREP.json.
python scripts/audit_massive_medical_union_medical_recovery_v1.py write-prep
REMOTE

echo 'Medical-only recovery staged. No Slurm job or API call was made.'
echo 'Explicit one-job submit command (not run):'
echo "  ssh $TILLICUM_HOST 'cd $TILLICUM_ROOT/projects/subliminal-mitigate-mmu-medical-recovery-v1 && scripts/submit_massive_medical_union_medical_recovery_v1_tillicum.sh medical-recovery-v1 --ack-max-cost-usd 0.15'"
