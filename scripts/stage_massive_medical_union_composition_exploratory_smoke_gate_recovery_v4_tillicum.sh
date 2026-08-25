#!/bin/bash
# CPU-only staging and finalization of the sealed smoke gate recovery v4.

set -euo pipefail
umask 077

[[ $# -eq 0 ]] || {
  echo 'Usage: scripts/stage_massive_medical_union_composition_exploratory_smoke_gate_recovery_v4_tillicum.sh' >&2
  exit 2
}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
TILLICUM_HOST=${TILLICUM_HOST:-tillicum}
TILLICUM_ROOT=${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}
REMOTE_REPO_URL=${REMOTE_REPO_URL:-https://github.com/ClaireJieZhang/emergent_misalignment_mitigation.git}
REMOTE_BRANCH=${REMOTE_BRANCH:-claire/capability-quorum-secure-code-composition-exploratory-smoke-gate-recovery-v4}
source_commit=99427421d44b447927c4eb1f66f3254c007dfc6d
expected_commit=$(git -C "$repo_root" rev-parse HEAD)

modified_files=(
  scripts/summarize_massive_medical_union_composition_exploratory_v1.py
  tests/test_massive_medical_union_composition_exploratory_evaluation.py
  tests/test_massive_medical_union_composition_exploratory_independent_model_recovery_v3_workflow.py
)
added_files=(
  docs/massive_medical_union_composition_exploratory_smoke_gate_recovery_v4.md
  scripts/audit_massive_medical_union_composition_exploratory_smoke_gate_recovery_v4.py
  scripts/stage_massive_medical_union_composition_exploratory_smoke_gate_recovery_v4_tillicum.sh
  scripts/status_massive_medical_union_composition_exploratory_smoke_gate_recovery_v4_tillicum.sh
  tests/test_massive_medical_union_composition_exploratory_smoke_gate_recovery_v4_workflow.py
)
all_files=("${modified_files[@]}" "${added_files[@]}")

test "$(git -C "$repo_root" rev-list --parents -n 1 "$expected_commit")" = "$expected_commit $source_commit"
test "$(git -C "$repo_root" branch --show-current)" = "$REMOTE_BRANCH"
observed_diff=$(git -C "$repo_root" diff --name-status --no-renames "$source_commit..$expected_commit")
for path in "${modified_files[@]}"; do
  grep -Fxq $'M\t'"$path" <<<"$observed_diff"
done
for path in "${added_files[@]}"; do
  grep -Fxq $'A\t'"$path" <<<"$observed_diff"
done
test "$(wc -l <<<"$observed_diff" | tr -d ' ')" -eq 8
test -z "$(git -C "$repo_root" status --porcelain -- "${all_files[@]}")" || {
  echo 'Gate-recovery files differ from committed HEAD.' >&2
  git -C "$repo_root" status --short -- "${all_files[@]}" >&2
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
repo=$root/projects/subliminal-mitigate-mmu-composition-exploratory-smoke-gate-recovery-v4
output=$root/outputs/massive_medical_union_composition_exploratory_smoke_gate_recovery_v4
source_repo=$root/projects/subliminal-mitigate-mmu-composition-exploratory-independent-model-recovery-v3
source_output=$root/outputs/massive_medical_union_composition_exploratory_independent_model_recovery_v3
env_root=$root/envs/subliminal-mitigate-py311

test ! -e "$repo"
test ! -e "$output"
test -d "$source_repo"
test -d "$source_output"
test -f "$env_root/.ready"
if compgen -G "$root/outputs/logs/massive_medical_union_composition_exploratory_smoke_gate_recovery_v4_*" >/dev/null; then
  echo 'Gate-recovery log namespace is not fresh.' >&2
  exit 4
fi

git clone --branch "$branch" --single-branch "$repo_url" "$repo"
test "$(git -C "$repo" rev-parse HEAD)" = "$expected_commit"

executables=(
  scripts/prepare_massive_medical_union_composition_exploratory_v1.py
  scripts/audit_massive_medical_union_composition_exploratory_v1.py
  scripts/sample_massive_medical_union_composition_exploratory_v1.py
  scripts/summarize_massive_medical_union_composition_exploratory_v1.py
  scripts/judge_massive_medical_union_composition_exploratory_v1.py
  scripts/merge_massive_medical_union_composition_exploratory_v1.py
  scripts/audit_massive_medical_union_composition_exploratory_workflow_v1.py
  scripts/stage_massive_medical_union_composition_exploratory_v1_tillicum.sh
  scripts/sbatch_massive_medical_union_composition_exploratory_v1_smoke_tillicum_h200.sbatch
  scripts/submit_massive_medical_union_composition_exploratory_v1_smoke_tillicum.sh
  scripts/sbatch_massive_medical_union_composition_exploratory_v1_confirmation_tillicum_h200.sbatch
  scripts/submit_massive_medical_union_composition_exploratory_v1_confirmation_tillicum.sh
  scripts/finalize_massive_medical_union_composition_exploratory_v1_tillicum.sh
  scripts/status_massive_medical_union_composition_exploratory_v1_tillicum.sh
  scripts/audit_massive_medical_union_composition_exploratory_smoke_recovery_v1.py
  scripts/stage_massive_medical_union_composition_exploratory_smoke_recovery_v1_tillicum.sh
  scripts/sbatch_massive_medical_union_composition_exploratory_smoke_recovery_v1_tillicum_h200.sbatch
  scripts/submit_massive_medical_union_composition_exploratory_smoke_recovery_v1_tillicum.sh
  scripts/status_massive_medical_union_composition_exploratory_smoke_recovery_v1_tillicum.sh
  scripts/audit_massive_medical_union_composition_exploratory_smoke_probe_recovery_v2.py
  scripts/stage_massive_medical_union_composition_exploratory_smoke_probe_recovery_v2_tillicum.sh
  scripts/sbatch_massive_medical_union_composition_exploratory_smoke_probe_recovery_v2_tillicum_h200.sbatch
  scripts/submit_massive_medical_union_composition_exploratory_smoke_probe_recovery_v2_tillicum.sh
  scripts/status_massive_medical_union_composition_exploratory_smoke_probe_recovery_v2_tillicum.sh
  scripts/audit_massive_medical_union_composition_exploratory_independent_model_recovery_v3.py
  scripts/stage_massive_medical_union_composition_exploratory_independent_model_recovery_v3_tillicum.sh
  scripts/sbatch_massive_medical_union_composition_exploratory_independent_model_recovery_v3_tillicum_h200.sbatch
  scripts/submit_massive_medical_union_composition_exploratory_independent_model_recovery_v3_tillicum.sh
  scripts/status_massive_medical_union_composition_exploratory_independent_model_recovery_v3_tillicum.sh
  scripts/audit_massive_medical_union_composition_exploratory_smoke_gate_recovery_v4.py
  scripts/stage_massive_medical_union_composition_exploratory_smoke_gate_recovery_v4_tillicum.sh
  scripts/status_massive_medical_union_composition_exploratory_smoke_gate_recovery_v4_tillicum.sh
)
regular=(
  docs/massive_medical_union_composition_exploratory_v1.md
  docs/massive_medical_union_composition_exploratory_smoke_recovery_v1.md
  docs/massive_medical_union_composition_exploratory_smoke_probe_recovery_v2.md
  docs/massive_medical_union_composition_exploratory_independent_model_recovery_v3.md
  docs/massive_medical_union_composition_exploratory_smoke_gate_recovery_v4.md
  tests/test_massive_medical_union_composition_exploratory_protocol.py
  tests/test_massive_medical_union_composition_exploratory_sampler.py
  tests/test_massive_medical_union_composition_exploratory_evaluation.py
  tests/test_massive_medical_union_composition_exploratory_workflow.py
  tests/test_massive_medical_union_composition_exploratory_smoke_recovery_workflow.py
  tests/test_massive_medical_union_composition_exploratory_smoke_probe_recovery_v2_workflow.py
  tests/test_massive_medical_union_composition_exploratory_independent_model_recovery_v3_workflow.py
  tests/test_massive_medical_union_composition_exploratory_smoke_gate_recovery_v4_workflow.py
)
for path in "${executables[@]}"; do
  test "$(git -C "$repo" ls-files -s -- "$path" | awk 'NR == 1 {print $1}')" = 100755
  chmod 0755 "$repo/$path"
done
for path in "${regular[@]}"; do
  test "$(git -C "$repo" ls-files -s -- "$path" | awk 'NR == 1 {print $1}')" = 100644
  chmod 0644 "$repo/$path"
done
test -z "$(git -C "$repo" status --porcelain)"

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY
unset TRANSFORMERS_CACHE CUDA_VISIBLE_DEVICES
module load conda/Miniforge3-25.3.1-3
conda activate "$env_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-composition-smoke-gate-recovery-v4-stage-pyc
export DO_NOT_TRACK=1
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export XDG_CACHE_HOME=$root/cache
export XDG_CONFIG_HOME=$root/config
export TMPDIR=$root/tmp
export HF_HOME=$root/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

cd "$repo"
python -m pip check
bash -n \
  scripts/stage_massive_medical_union_composition_exploratory_smoke_gate_recovery_v4_tillicum.sh \
  scripts/status_massive_medical_union_composition_exploratory_smoke_gate_recovery_v4_tillicum.sh
python -m py_compile \
  scripts/prepare_massive_medical_union_composition_exploratory_v1.py \
  scripts/audit_massive_medical_union_composition_exploratory_v1.py \
  scripts/sample_massive_medical_union_composition_exploratory_v1.py \
  scripts/summarize_massive_medical_union_composition_exploratory_v1.py \
  scripts/judge_massive_medical_union_composition_exploratory_v1.py \
  scripts/merge_massive_medical_union_composition_exploratory_v1.py \
  scripts/audit_massive_medical_union_composition_exploratory_workflow_v1.py \
  scripts/audit_massive_medical_union_composition_exploratory_smoke_recovery_v1.py \
  scripts/audit_massive_medical_union_composition_exploratory_smoke_probe_recovery_v2.py \
  scripts/audit_massive_medical_union_composition_exploratory_independent_model_recovery_v3.py \
  scripts/audit_massive_medical_union_composition_exploratory_smoke_gate_recovery_v4.py
python -m unittest \
  tests.test_massive_medical_union_composition_exploratory_protocol \
  tests.test_massive_medical_union_composition_exploratory_sampler \
  tests.test_massive_medical_union_composition_exploratory_evaluation \
  tests.test_massive_medical_union_composition_exploratory_workflow \
  tests.test_massive_medical_union_composition_exploratory_smoke_recovery_workflow \
  tests.test_massive_medical_union_composition_exploratory_smoke_probe_recovery_v2_workflow \
  tests.test_massive_medical_union_composition_exploratory_independent_model_recovery_v3_workflow \
  tests.test_massive_medical_union_composition_exploratory_smoke_gate_recovery_v4_workflow

auditor=scripts/audit_massive_medical_union_composition_exploratory_smoke_gate_recovery_v4.py
python "$auditor" write-prep
python "$auditor" audit-prep
python "$auditor" write-staged
python "$auditor" audit-staged
python "$auditor" recover-gate
python "$auditor" audit-terminal
REMOTE

echo 'CPU-only exploratory smoke gate recovery v4 completed and sealed.'
echo 'No Slurm job, GPU allocation, model load, generation, API call, or confirmation was authorized.'
echo "Read-only status: ssh $TILLICUM_HOST '$TILLICUM_ROOT/projects/subliminal-mitigate-mmu-composition-exploratory-smoke-gate-recovery-v4/scripts/status_massive_medical_union_composition_exploratory_smoke_gate_recovery_v4_tillicum.sh'"
