#!/bin/bash
# CPU-only, exact-once staging for the second Wave-1 medical recovery.

set -euo pipefail
umask 077

[[ $# -eq 0 ]] || {
  echo 'Usage: scripts/stage_massive_medical_union_medical_recovery_v2_tillicum.sh' >&2
  exit 2
}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
TILLICUM_HOST=${TILLICUM_HOST:-tillicum}
TILLICUM_ROOT=${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}
REMOTE_REPO_URL=${REMOTE_REPO_URL:-https://github.com/ClaireJieZhang/emergent_misalignment_mitigation.git}
REMOTE_BRANCH=${REMOTE_BRANCH:-claire/capability-quorum-secure-code}
expected_commit=$(git -C "$repo_root" rev-parse HEAD)

required_files=(
  scripts/audit_massive_medical_union_medical_recovery_v1.py
  scripts/audit_massive_medical_union_medical_recovery_v2.py
  scripts/finalize_massive_medical_union_wave1_medical_recovery_v2_tillicum.sh
  scripts/judge_massive_union_medical.py
  scripts/sample_massive_union_medical_direct.py
  scripts/sbatch_massive_medical_union_medical_recovery_v2_tillicum_h200.sbatch
  scripts/stage_massive_medical_union_medical_recovery_v2_tillicum.sh
  scripts/status_massive_medical_union_medical_recovery_v2_tillicum.sh
  scripts/submit_massive_medical_union_medical_recovery_v2_tillicum.sh
  scripts/summarize_massive_union_components.py
  tests/test_massive_medical_union_medical_recovery_v1.py
  tests/test_massive_medical_union_medical_recovery_v2.py
  tests/test_massive_union_component_evaluation.py
)
for path in "${required_files[@]}"; do
  test -s "$repo_root/$path"
  git -C "$repo_root" cat-file -e "$expected_commit:$path"
done
test -z "$(git -C "$repo_root" status --porcelain -- "${required_files[@]}")" || {
  echo 'Medical-recovery-v2 workflow differs from committed HEAD.' >&2
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
output=$root/outputs/massive_medical_union_pilot_v1
recovery_v2_control=$output/control/medical_recovery_v2
recovery_v2_eval=$output/evaluation/wave1/medical_recovery_v2
env_root=$root/envs/subliminal-mitigate-py311

test "$(git -C "$main_repo" rev-parse HEAD)" = e25d59d8c5ea30c49cec207f5cac140a2281a525
test -z "$(git -C "$main_repo" status --porcelain)"
test "$(git -C "$recovery_v1_repo" rev-parse HEAD)" = 9ddd4816dafeb9b3df709e6ac72f41ebb22ee49f
test -z "$(git -C "$recovery_v1_repo" status --porcelain)"
test ! -e "$recovery_v2_repo"
test ! -e "$recovery_v2_control"
test ! -e "$recovery_v2_eval"
test -f "$env_root/.ready"

git clone --branch "$branch" --single-branch "$repo_url" "$recovery_v2_repo"
test "$(git -C "$recovery_v2_repo" rev-parse HEAD)" = "$expected_commit"
test -z "$(git -C "$recovery_v2_repo" status --porcelain)"

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
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-medical-recovery-v2-stage-pyc

cd "$recovery_v2_repo"
python -m pip check
bash -n \
  scripts/stage_massive_medical_union_medical_recovery_v2_tillicum.sh \
  scripts/submit_massive_medical_union_medical_recovery_v2_tillicum.sh \
  scripts/status_massive_medical_union_medical_recovery_v2_tillicum.sh \
  scripts/finalize_massive_medical_union_wave1_medical_recovery_v2_tillicum.sh \
  scripts/sbatch_massive_medical_union_medical_recovery_v2_tillicum_h200.sbatch
python -m py_compile \
  scripts/audit_massive_medical_union_medical_recovery_v1.py \
  scripts/audit_massive_medical_union_medical_recovery_v2.py \
  scripts/sample_massive_union_medical_direct.py \
  scripts/judge_massive_union_medical.py \
  scripts/summarize_massive_union_components.py
python -m unittest \
  tests.test_massive_medical_union_medical_recovery_v2 \
  tests.test_massive_medical_union_medical_recovery_v1 \
  tests.test_massive_union_component_evaluation
python scripts/sample_massive_union_medical_direct.py \
  --model pi_base=BASE \
  --training_config configs/training_qwen25_7b_massive_medical_union_pilot.yaml \
  --data_manifest "$output/data/data_manifest.json" \
  --prompt_file "$output/data/medical_eval/official16.json" \
  --output_dir "$root/tmp/mmu-medical-recovery-v2-preflight" \
  --sampling_profile official16_max1024_all_stop_v2 --preflight_only

# The auditor binds the failed v1 job, confirms that it produced no recovery
# output or API activity, and creates only the fresh v2 control namespace.
python scripts/audit_massive_medical_union_medical_recovery_v2.py write-prep
REMOTE

echo 'Medical-only recovery v2 staged. No Slurm job or API call was made.'
echo 'Explicit one-job submit command (not run):'
echo "  ssh $TILLICUM_HOST 'cd $TILLICUM_ROOT/projects/subliminal-mitigate-mmu-medical-recovery-v2 && scripts/submit_massive_medical_union_medical_recovery_v2_tillicum.sh medical-recovery-v2 --ack-max-cost-usd 0.15'"
