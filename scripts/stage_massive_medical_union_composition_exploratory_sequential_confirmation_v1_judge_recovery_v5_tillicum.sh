#!/bin/bash
# CPU-only staging for the fresh judge recovery v5.

set -euo pipefail
umask 077
ulimit -c 0

[[ $# -eq 0 ]] || { echo 'Usage: stage_..._judge_recovery_v5_tillicum.sh' >&2; exit 2; }

local_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
host=${TILLICUM_HOST:-tillicum}
root=${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}
url=${REMOTE_REPO_URL:-https://github.com/ClaireJieZhang/emergent_misalignment_mitigation.git}
branch=${REMOTE_BRANCH:-claire/capability-quorum-secure-code-composition-exploratory-under5-sequential-v1-judge-recovery-v5}
parent=5f7357fee6654cccb7918d307963dcfe5fa73418
commit=$(git -C "$local_repo" rev-parse HEAD)
test "$(git -C "$local_repo" rev-list --parents -n 1 "$commit")" = "$commit $parent"
test "$(git -C "$local_repo" branch --show-current)" = "$branch"
test -z "$(git -C "$local_repo" status --porcelain)"

ssh "$host" bash -s -- "$root" "$url" "$branch" "$commit" <<'REMOTE'
set -euo pipefail
umask 077
ulimit -c 0

root=$1; url=$2; branch=$3; expected=$4
prior_repo=$root/projects/subliminal-mitigate-mmu-composition-exploratory-sequential-confirmation-v1-judge-recovery-v4
prior_output=$root/outputs/massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v4
repo=$root/projects/subliminal-mitigate-mmu-composition-exploratory-sequential-confirmation-v1-judge-recovery-v5
output=$root/outputs/massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v5
env_root=$root/envs/subliminal-mitigate-py311
export GIT_OPTIONAL_LOCKS=0

test -d "$prior_repo"; test ! -L "$prior_repo"
test "$(git -C "$prior_repo" rev-parse HEAD)" = 5f7357fee6654cccb7918d307963dcfe5fa73418
test "$(git -C "$prior_repo" rev-parse 'HEAD^{tree}')" = 7770b7e60f9942077ffb6484ce7db41e55d6a190
test "$(git -C "$prior_repo" branch --show-current)" = claire/capability-quorum-secure-code-composition-exploratory-under5-sequential-v1-judge-recovery-v4
test -z "$(git -C "$prior_repo" status --porcelain)"
test -d "$prior_output"; test ! -L "$prior_output"
test -f "$prior_output/control/CANARY_FAILURE.json"
test ! -e "$prior_output/control/CANARY_SUCCESS.json"
test ! -e "$prior_output/control/CONTINUATION_AUTHORIZATION.json"
test ! -e "$prior_output/control/CONTINUATION_SUCCESS.json"
test ! -e "$prior_output/evaluation/medical/judge_checkpoint.json.001"
test ! -e "$repo"; test ! -e "$output"

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

[[ -z ${OPENAI_API_KEY:-} ]] || { echo 'OPENAI_API_KEY must be absent during CPU staging.' >&2; exit 6; }
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN WANDB_API_KEY
unset ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY CUDA_VISIBLE_DEVICES
unset TRANSFORMERS_CACHE
module load conda/Miniforge3-25.3.1-3
conda activate "$env_root"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-sequential-judge-recovery-v5-pyc
export DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export XDG_CACHE_HOME=$root/cache XDG_CONFIG_HOME=$root/config TMPDIR=$root/tmp
export HF_HOME=$root/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

cd "$repo"
auditor=scripts/audit_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v5.py
judge=scripts/judge_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v5.py
summary=scripts/summarize_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v5.py
manifest=$output/control/JUDGE_RECOVERY_V5_MANIFEST.json

python "$auditor" prepare
python -m pip check
bash -n \
  scripts/stage_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v5_tillicum.sh \
  scripts/finalize_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v5_tillicum.sh \
  scripts/derive_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v5_tillicum.sh \
  scripts/status_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v5_tillicum.sh
python -m py_compile "$auditor" "$judge" "$summary"
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
python -m unittest discover -s tests -p 'test_massive_medical_union_composition_exploratory_sequential*judge_recovery_v4*.py'
python -m unittest discover -s tests -p 'test_massive_medical_union_composition_exploratory_sequential*judge_recovery_v5*.py'
python "$judge" validate-static --recovery-manifest "$manifest"
python "$judge" validate-plan --recovery-manifest "$manifest"
python "$judge" validate-sdk-serialization --recovery-manifest "$manifest"
python "$summary" validate-static --recovery-manifest "$manifest"
python "$auditor" seal-staged \
  --validation-command 'pip check' \
  --validation-command 'bash syntax checks' \
  --validation-command 'Python compile checks' \
  --validation-command 'frozen sequential regression suite' \
  --validation-command 'unmodified v4 recovery focused suite after v5 imports' \
  --validation-command 'v5 recovery local MockTransport and control tests' \
  --validation-command 'three-range real SDK serialization through local MockTransport' \
  --validation-command 'reconstructed exact sealed plan and idempotency range validation'
python "$auditor" audit-staged
test -z "$(git -C "$repo" status --porcelain)"
REMOTE

echo 'Sequential judge recovery v5 CPU stage completed.'
echo 'No Slurm job, GPU allocation, model load, external API call, or API authorization was executed.'
