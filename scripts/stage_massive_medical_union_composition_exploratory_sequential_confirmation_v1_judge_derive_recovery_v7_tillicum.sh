#!/bin/bash
# CPU-only staging of the add-only v7 derivation recovery.

set -euo pipefail
umask 077
ulimit -c 0

[[ $# -eq 0 ]] || { echo 'Usage: stage_..._judge_derive_recovery_v7_tillicum.sh' >&2; exit 2; }
[[ -z ${OPENAI_API_KEY:-} ]] || {
  echo 'OPENAI_API_KEY must be absent during CPU-only v7 staging.' >&2
  exit 6
}

local_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
host=${TILLICUM_HOST:-tillicum}
root=${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}
url=${REMOTE_REPO_URL:-https://github.com/ClaireJieZhang/emergent_misalignment_mitigation.git}
branch=${REMOTE_BRANCH:-claire/capability-quorum-secure-code-composition-exploratory-under5-sequential-v1-judge-derive-recovery-v7}
parent=c4016c332c461efa07c85028a164c787f2e65650
commit=$(git -C "$local_repo" rev-parse HEAD)
test "$(git -C "$local_repo" rev-list --parents -n 1 "$commit")" = "$commit $parent"
test "$(git -C "$local_repo" branch --show-current)" = "$branch"
test -z "$(git -C "$local_repo" status --porcelain)"

ssh "$host" bash -s -- "$root" "$url" "$branch" "$commit" <<'REMOTE'
set -euo pipefail
umask 077
ulimit -c 0

root=$1; url=$2; branch=$3; expected=$4
source_repo=$root/projects/subliminal-mitigate-mmu-composition-exploratory-sequential-confirmation-v1-judge-recovery-v6
source_output=$root/outputs/massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v6
repo=$root/projects/subliminal-mitigate-mmu-composition-exploratory-sequential-confirmation-v1-judge-derive-recovery-v7
output=$root/outputs/massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_derive_recovery_v7
env_root=$root/envs/subliminal-mitigate-py311
export GIT_OPTIONAL_LOCKS=0

[[ -z ${OPENAI_API_KEY:-} ]] || {
  echo 'OPENAI_API_KEY must be absent during CPU-only v7 staging.' >&2
  exit 6
}
test -d "$source_repo"; test ! -L "$source_repo"
test "$(git -C "$source_repo" rev-parse HEAD)" = c4016c332c461efa07c85028a164c787f2e65650
test "$(git -C "$source_repo" rev-parse 'HEAD^{tree}')" = 20b84f0edbebd1274fa2ca11144d11b5b95e2991
test "$(git -C "$source_repo" branch --show-current)" = claire/capability-quorum-secure-code-composition-exploratory-under5-sequential-v1-judge-recovery-v6
test -z "$(git -C "$source_repo" status --porcelain)"
test -d "$source_output"; test ! -L "$source_output"
test -f "$source_output/control/CONTINUATION_SUCCESS.json"
test -f "$source_output/evaluation/medical/judge_checkpoint.json.002"
test -f "$source_output/evaluation/medical/judge_checkpoint.json.240"
test -f "$source_output/evaluation/medical/judgments_new.json"
test ! -e "$source_output/evaluation/medical/judgments_merged.json"
test ! -e "$source_output/control/FINAL_RESULT.json"
test ! -e "$repo"; test ! -e "$output"

git clone --branch "$branch" --single-branch "$url" "$repo"
test "$(git -C "$repo" rev-parse HEAD)" = "$expected"
test "$(git -C "$repo" rev-list --parents -n 1 "$expected")" = "$expected c4016c332c461efa07c85028a164c787f2e65650"
while read -r mode object stage path; do
  test "$stage" = 0
  case "$mode" in
    100755) chmod 0755 "$repo/$path" ;;
    100644) chmod 0644 "$repo/$path" ;;
    *) echo "Unsupported tracked mode $mode for $path" >&2; exit 5 ;;
  esac
done < <(git -C "$repo" ls-files -s)
test -z "$(git -C "$repo" status --porcelain)"

unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN WANDB_API_KEY
unset ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY
unset TRANSFORMERS_CACHE
module load conda/Miniforge3-25.3.1-3
conda activate "$env_root"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-sequential-judge-derive-recovery-v7-pyc
export DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none ROCR_VISIBLE_DEVICES=''
export XDG_CACHE_HOME=$root/cache XDG_CONFIG_HOME=$root/config TMPDIR=$root/tmp
export HF_HOME=$root/cache/huggingface HUGGINGFACE_HUB_CACHE=$root/cache/huggingface/hub

cd "$repo"
auditor=scripts/audit_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_derive_recovery_v7.py
summary=scripts/summarize_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_derive_recovery_v7.py
manifest=$output/control/JUDGE_DERIVE_RECOVERY_V7_MANIFEST.json

python "$auditor" prepare
python -m pip check
bash -n \
  scripts/stage_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_derive_recovery_v7_tillicum.sh \
  scripts/derive_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_derive_recovery_v7_tillicum.sh \
  scripts/status_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_derive_recovery_v7_tillicum.sh
python -m py_compile "$auditor" "$summary"
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
python -m unittest discover -s tests -p 'test_massive_medical_union_composition_exploratory_sequential*judge_recovery_v6*.py'
python -m unittest \
  tests.test_massive_medical_union_composition_exploratory_sequential_judge_derive_recovery_v7_control \
  tests.test_massive_medical_union_composition_exploratory_sequential_judge_derive_recovery_v7_summary
python "$summary" validate-static --derive-manifest "$manifest"
python "$auditor" seal-staged \
  --validation-command 'pip check' \
  --validation-command 'shell syntax and Python compile checks' \
  --validation-command '228 frozen workflow tests plus v4, v5, v6, and v7 recovery tests' \
  --validation-command 'v7 control and corrected chronological-sum summary tests' \
  --validation-command 'exact v6 repo, 252-file terminal inventory, and checkpoint-chain audit' \
  --validation-command 'sealed cost-order mismatch regression evidence' \
  --validation-command 'CPU-only static derivation validation'
python "$auditor" audit-staged
python "$summary" validate-static --derive-manifest "$manifest"
test "$(git -C "$source_repo" rev-parse HEAD)" = c4016c332c461efa07c85028a164c787f2e65650
test -z "$(git -C "$source_repo" status --porcelain)"
test ! -e "$source_output/evaluation/medical/judgments_merged.json"
test ! -e "$source_output/control/FINAL_RESULT.json"
test -z "$(git -C "$repo" status --porcelain)"
[[ -z ${OPENAI_API_KEY:-} ]]
REMOTE

echo 'Judge derivation recovery v7 CPU stage completed.'
echo 'The Git clone was the sole staging network operation; derivation itself is offline.'
echo 'No accelerator job, model load, external API call, or API authorization was executed.'
