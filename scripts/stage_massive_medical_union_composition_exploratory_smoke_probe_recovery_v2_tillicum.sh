#!/bin/bash
# CPU-only staging for the one-shot exploratory smoke probe recovery v2.

set -euo pipefail
umask 077

[[ $# -eq 0 ]] || {
  echo 'Usage: scripts/stage_massive_medical_union_composition_exploratory_smoke_probe_recovery_v2_tillicum.sh' >&2
  exit 2
}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
TILLICUM_HOST=${TILLICUM_HOST:-tillicum}
TILLICUM_ROOT=${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}
REMOTE_REPO_URL=${REMOTE_REPO_URL:-https://github.com/ClaireJieZhang/emergent_misalignment_mitigation.git}
REMOTE_BRANCH=${REMOTE_BRANCH:-claire/capability-quorum-secure-code-composition-exploratory-smoke-probe-recovery-v2}
source_commit=8b5f8db2d54b6fc7da3d54c1cb024e950201d879
expected_commit=$(git -C "$repo_root" rev-parse HEAD)

changed_files=(
  scripts/sample_massive_medical_union_composition_exploratory_v1.py
  scripts/summarize_massive_medical_union_composition_exploratory_v1.py
  tests/test_massive_medical_union_composition_exploratory_sampler.py
  tests/test_massive_medical_union_composition_exploratory_evaluation.py
  docs/massive_medical_union_composition_exploratory_smoke_probe_recovery_v2.md
  scripts/audit_massive_medical_union_composition_exploratory_smoke_probe_recovery_v2.py
  scripts/stage_massive_medical_union_composition_exploratory_smoke_probe_recovery_v2_tillicum.sh
  scripts/sbatch_massive_medical_union_composition_exploratory_smoke_probe_recovery_v2_tillicum_h200.sbatch
  scripts/submit_massive_medical_union_composition_exploratory_smoke_probe_recovery_v2_tillicum.sh
  scripts/status_massive_medical_union_composition_exploratory_smoke_probe_recovery_v2_tillicum.sh
  tests/test_massive_medical_union_composition_exploratory_smoke_probe_recovery_v2_workflow.py
)
test "$(git -C "$repo_root" rev-list --parents -n 1 "$expected_commit")" = "$expected_commit $source_commit"
test "$(git -C "$repo_root" branch --show-current)" = "$REMOTE_BRANCH"
for path in "${changed_files[@]}"; do
  test -s "$repo_root/$path"
  git -C "$repo_root" cat-file -e "$expected_commit:$path"
done
test -z "$(git -C "$repo_root" status --porcelain -- "${changed_files[@]}")" || {
  echo 'Recovery files differ from committed HEAD.' >&2
  git -C "$repo_root" status --short -- "${changed_files[@]}" >&2
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
repo=$root/projects/subliminal-mitigate-mmu-composition-exploratory-smoke-probe-recovery-v2
output=$root/outputs/massive_medical_union_composition_exploratory_smoke_probe_recovery_v2
env_root=$root/envs/subliminal-mitigate-py311
source_repo=$root/projects/subliminal-mitigate-mmu-composition-exploratory-v1
source_output=$root/outputs/massive_medical_union_composition_exploratory_v1
probe_recovery_v1_repo=$root/projects/subliminal-mitigate-mmu-composition-exploratory-smoke-recovery-v1
probe_recovery_v1_output=$root/outputs/massive_medical_union_composition_exploratory_smoke_recovery_v1

test ! -e "$repo"
test ! -e "$output"
test -d "$source_repo"
test -d "$source_output"
test -d "$probe_recovery_v1_repo"
test -d "$probe_recovery_v1_output"
test -f "$env_root/.ready"
if compgen -G "$root/outputs/logs/massive_medical_union_composition_exploratory_smoke_probe_recovery_v2_*" >/dev/null; then
  echo 'Recovery log namespace is not fresh.' >&2
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
)
regular=(
  docs/massive_medical_union_composition_exploratory_v1.md
  docs/massive_medical_union_composition_exploratory_smoke_recovery_v1.md
  docs/massive_medical_union_composition_exploratory_smoke_probe_recovery_v2.md
  tests/test_massive_medical_union_composition_exploratory_protocol.py
  tests/test_massive_medical_union_composition_exploratory_sampler.py
  tests/test_massive_medical_union_composition_exploratory_evaluation.py
  tests/test_massive_medical_union_composition_exploratory_workflow.py
  tests/test_massive_medical_union_composition_exploratory_smoke_recovery_workflow.py
  tests/test_massive_medical_union_composition_exploratory_smoke_probe_recovery_v2_workflow.py
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
unset TRANSFORMERS_CACHE
module load gcc/13.4.0 cuda/12.9.1
module load conda/Miniforge3-25.3.1-3
conda activate "$env_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-composition-smoke-probe-recovery-v2-stage-pyc
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
export TRITON_CACHE_DIR=$root/cache/triton

cd "$repo"
python -m pip check
bash -n \
  scripts/stage_massive_medical_union_composition_exploratory_v1_tillicum.sh \
  scripts/sbatch_massive_medical_union_composition_exploratory_v1_smoke_tillicum_h200.sbatch \
  scripts/submit_massive_medical_union_composition_exploratory_v1_smoke_tillicum.sh \
  scripts/sbatch_massive_medical_union_composition_exploratory_v1_confirmation_tillicum_h200.sbatch \
  scripts/submit_massive_medical_union_composition_exploratory_v1_confirmation_tillicum.sh \
  scripts/finalize_massive_medical_union_composition_exploratory_v1_tillicum.sh \
  scripts/status_massive_medical_union_composition_exploratory_v1_tillicum.sh \
  scripts/stage_massive_medical_union_composition_exploratory_smoke_recovery_v1_tillicum.sh \
  scripts/sbatch_massive_medical_union_composition_exploratory_smoke_recovery_v1_tillicum_h200.sbatch \
  scripts/submit_massive_medical_union_composition_exploratory_smoke_recovery_v1_tillicum.sh \
  scripts/status_massive_medical_union_composition_exploratory_smoke_recovery_v1_tillicum.sh \
  scripts/stage_massive_medical_union_composition_exploratory_smoke_probe_recovery_v2_tillicum.sh \
  scripts/sbatch_massive_medical_union_composition_exploratory_smoke_probe_recovery_v2_tillicum_h200.sbatch \
  scripts/submit_massive_medical_union_composition_exploratory_smoke_probe_recovery_v2_tillicum.sh \
  scripts/status_massive_medical_union_composition_exploratory_smoke_probe_recovery_v2_tillicum.sh
python -m py_compile \
  scripts/prepare_massive_medical_union_composition_exploratory_v1.py \
  scripts/audit_massive_medical_union_composition_exploratory_v1.py \
  scripts/sample_massive_medical_union_composition_exploratory_v1.py \
  scripts/summarize_massive_medical_union_composition_exploratory_v1.py \
  scripts/judge_massive_medical_union_composition_exploratory_v1.py \
  scripts/merge_massive_medical_union_composition_exploratory_v1.py \
  scripts/audit_massive_medical_union_composition_exploratory_workflow_v1.py \
  scripts/audit_massive_medical_union_composition_exploratory_smoke_recovery_v1.py \
  scripts/audit_massive_medical_union_composition_exploratory_smoke_probe_recovery_v2.py
python -m unittest \
  tests.test_massive_medical_union_composition_exploratory_protocol \
  tests.test_massive_medical_union_composition_exploratory_sampler \
  tests.test_massive_medical_union_composition_exploratory_evaluation \
  tests.test_massive_medical_union_composition_exploratory_workflow \
  tests.test_massive_medical_union_composition_exploratory_smoke_recovery_workflow \
  tests.test_massive_medical_union_composition_exploratory_smoke_probe_recovery_v2_workflow

python scripts/audit_massive_medical_union_composition_exploratory_smoke_probe_recovery_v2.py write-prep
preflight=$output/control/SMOKE_PROBE_RECOVERY_CPU_PREFLIGHT.json
preflight_tmp=$preflight.tmp.$$
python scripts/sample_massive_medical_union_composition_exploratory_v1.py \
  --phase smoke --protocol-manifest "$source_output/protocol/manifest.json" \
  --output-root "$output/generation" --device cuda:0 --preflight-only \
  > "$preflight_tmp"
chmod 0400 "$preflight_tmp"
mv "$preflight_tmp" "$preflight"
python scripts/audit_massive_medical_union_composition_exploratory_smoke_probe_recovery_v2.py audit-preflight
python scripts/audit_massive_medical_union_composition_exploratory_smoke_probe_recovery_v2.py write-staged
python scripts/audit_massive_medical_union_composition_exploratory_smoke_probe_recovery_v2.py audit-staged
REMOTE

echo 'Exploratory smoke probe recovery v2 staged and CPU-sealed. No job or API call was made.'
echo 'Explicit recovery submit command (not run by staging):'
echo "  ssh $TILLICUM_HOST 'cd $TILLICUM_ROOT/projects/subliminal-mitigate-mmu-composition-exploratory-smoke-probe-recovery-v2 && scripts/submit_massive_medical_union_composition_exploratory_smoke_probe_recovery_v2_tillicum.sh smoke-probe-recovery-v2 --ack-prior-actual-cost-usd 0.02650 --ack-max-recovery-cost-usd 0.225 --ack-actual-plus-cap-cost-usd 0.25150'"
