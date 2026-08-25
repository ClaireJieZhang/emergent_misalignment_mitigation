#!/bin/bash
# CPU-only stage recovery v2 for the under-$5 sequential exploratory confirmation workflow.

set -euo pipefail
umask 077
ulimit -c 0

[[ $# -eq 0 ]] || { echo 'Usage: stage_massive_medical_union_composition_exploratory_sequential_confirmation_v1_tillicum.sh' >&2; exit 2; }

local_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
host=${TILLICUM_HOST:-tillicum}
root=${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}
url=${REMOTE_REPO_URL:-https://github.com/ClaireJieZhang/emergent_misalignment_mitigation.git}
branch=${REMOTE_BRANCH:-claire/capability-quorum-secure-code-composition-exploratory-under5-sequential-v1-stage-recovery-v2}
parent=a5724e9a06204941df9ad07ad4a5f84502dde7f8
commit=$(git -C "$local_repo" rev-parse HEAD)
test "$(git -C "$local_repo" rev-list --parents -n 1 "$commit")" = "$commit $parent"
test "$(git -C "$local_repo" branch --show-current)" = "$branch"

ssh "$host" bash -s -- "$root" "$url" "$branch" "$commit" <<'REMOTE'
set -euo pipefail
umask 077
ulimit -c 0

root=$1; url=$2; branch=$3; expected=$4
repo=$root/projects/subliminal-mitigate-mmu-composition-exploratory-sequential-confirmation-v1-stage-recovery-v2
output=$root/outputs/massive_medical_union_composition_exploratory_sequential_confirmation_v1_stage_recovery_v2
control=$output/control
protocol=$output/protocol
logs=$root/outputs/logs
env_root=$root/envs/subliminal-mitigate-py311
failed_repo=$root/projects/subliminal-mitigate-mmu-composition-exploratory-sequential-confirmation-v1
failed_output=$root/outputs/massive_medical_union_composition_exploratory_sequential_confirmation_v1
export GIT_OPTIONAL_LOCKS=0
test -d "$failed_repo"; test ! -L "$failed_repo"
test "$(git -C "$failed_repo" rev-parse HEAD)" = a5724e9a06204941df9ad07ad4a5f84502dde7f8
test "$(git -C "$failed_repo" rev-parse 'HEAD^{tree}')" = 6bb7396cb9e2ac98b17df72f9b4461e5c2890a07
test "$(git -C "$failed_repo" rev-list --parents -n 1 HEAD)" = 'a5724e9a06204941df9ad07ad4a5f84502dde7f8 890f685b3198e30e1658aa7ab0aa9f11a537aaf9'
test "$(git -C "$failed_repo" branch --show-current)" = claire/capability-quorum-secure-code-composition-exploratory-under5-sequential-v1
test -z "$(git -C "$failed_repo" status --porcelain)"
test ! -e "$failed_output"
if compgen -G "$logs/massive_medical_union_composition_exploratory_sequential_confirmation_v1_benefit_*" >/dev/null || compgen -G "$logs/massive_medical_union_composition_exploratory_sequential_confirmation_v1_medical_*" >/dev/null; then
  echo 'Failed-v1 log namespace is no longer pristine.' >&2; exit 3
fi
test ! -e "$repo"
test ! -e "$output"
if compgen -G "$logs/massive_medical_union_composition_exploratory_sequential_confirmation_v1_stage_recovery_v2_*" >/dev/null; then
  echo 'Sequential log namespace is not fresh.' >&2; exit 4
fi

git clone --branch "$branch" --single-branch "$url" "$repo"
test "$(git -C "$repo" rev-parse HEAD)" = "$expected"

# Git honors only the executable bit; umask 077 otherwise yields 0700/0600.
# Normalize every tracked file to its exact index mode so all historical mode
# regressions are meaningful, without touching content or untracked paths.
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
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY CUDA_VISIBLE_DEVICES
unset TRANSFORMERS_CACHE
module load conda/Miniforge3-25.3.1-3
conda activate "$env_root"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-sequential-v1-stage-recovery-v2-pyc
export DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export XDG_CACHE_HOME=$root/cache XDG_CONFIG_HOME=$root/config TMPDIR=$root/tmp
export HF_HOME=$root/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

cd "$repo"
auditor=scripts/audit_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py
sampler=scripts/sample_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py
# Seal the fresh recovery namespace and the failed-v1 checkout binding before
# dependency checks or tests.  Any later CPU-stage failure remains auditable.
python "$auditor" write-prep
python -m pip check
bash -n \
  scripts/stage_massive_medical_union_composition_exploratory_sequential_confirmation_v1_tillicum.sh \
  scripts/status_massive_medical_union_composition_exploratory_sequential_confirmation_v1_tillicum.sh \
  scripts/sbatch_massive_medical_union_composition_exploratory_sequential_confirmation_v1_benefit_tillicum_h200.sbatch \
  scripts/submit_massive_medical_union_composition_exploratory_sequential_confirmation_v1_benefit_tillicum.sh \
  scripts/sbatch_massive_medical_union_composition_exploratory_sequential_confirmation_v1_medical_tillicum_h200.sbatch \
  scripts/submit_massive_medical_union_composition_exploratory_sequential_confirmation_v1_medical_tillicum.sh \
  scripts/finalize_massive_medical_union_composition_exploratory_sequential_confirmation_v1_tillicum.sh
python -m py_compile \
  scripts/prepare_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py \
  scripts/audit_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py \
  scripts/sample_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py \
  scripts/summarize_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py \
  scripts/judge_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py \
  scripts/merge_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py
python -m unittest \
  tests.test_massive_medical_union_composition_exploratory_protocol \
  tests.test_massive_medical_union_composition_exploratory_sampler \
  tests.test_massive_medical_union_composition_exploratory_evaluation \
  tests.test_massive_medical_union_composition_exploratory_workflow \
  tests.test_massive_medical_union_composition_exploratory_smoke_recovery_workflow \
  tests.test_massive_medical_union_composition_exploratory_smoke_probe_recovery_v2_workflow \
  tests.test_massive_medical_union_composition_exploratory_independent_model_recovery_v3_workflow \
  tests.test_massive_medical_union_composition_exploratory_smoke_gate_recovery_v4_workflow \
  tests.test_massive_medical_union_composition_exploratory_smoke_gate_recovery_v5_workflow \
  tests.test_massive_medical_union_composition_exploratory_sequential_confirmation_v1_protocol \
  tests.test_massive_medical_union_composition_exploratory_sequential_sampler \
  tests.test_massive_medical_union_composition_exploratory_sequential_evaluation \
  tests.test_massive_medical_union_composition_exploratory_sequential_confirmation_v1_workflow

python scripts/prepare_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py
python "$auditor" audit-protocol --protocol-root "$protocol"

for phase in benefit medical; do
  case "$phase" in
    benefit) upper=BENEFIT ;;
    medical) upper=MEDICAL ;;
    *) exit 9 ;;
  esac
  target=$control/SAMPLER_PREFLIGHT_${upper}.json
  temporary=$target.tmp.$$
  unused=$output/preflight-unused-$phase
  test ! -e "$unused"
  python "$sampler" --phase "$phase" --protocol-manifest "$protocol/manifest.json" \
    --output-root "$unused" --device cuda:0 --preflight-only > "$temporary"
  test ! -e "$unused"
  chmod 0400 "$temporary"
  mv "$temporary" "$target"
  python "$auditor" audit-preflight --stage "$phase"
done
python scripts/summarize_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py validate-static --protocol-manifest "$protocol/manifest.json"
python scripts/judge_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py validate-static --protocol-manifest "$protocol/manifest.json"
python scripts/merge_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py validate-static --protocol-manifest "$protocol/manifest.json"
python "$auditor" write-staged
python "$auditor" audit-staged

test ! -e "$output/generation"
test ! -e "$output/evaluation"
test -z "$(git -C "$repo" status --porcelain)"
REMOTE

echo 'Sequential exploratory confirmation CPU stage completed.'
echo 'No Slurm job, GPU allocation, model load, generation, API call, or downstream authorization was executed.'
