#!/bin/bash
# CPU-only deployment/staging. Invoke only after exact-commit export approval.

set -euo pipefail
umask 077
ulimit -c 0
[[ $# -eq 0 ]] || { echo 'Usage: stage_massive_medical_ratio_panels_v1_judge_tillicum.sh' >&2; exit 2; }
[[ ! ${OPENAI_API_KEY+x} ]] || { echo 'The API key must be absent during CPU staging.' >&2; exit 3; }
[[ ! ${BASH_ENV+x} && ! ${ENV+x} ]] || { echo 'Shell initialization overrides must be absent.' >&2; exit 3; }

local_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
branch=claire/massive-medical-ratio-panel-judge-v1
commit=$(git -C "$local_repo" rev-parse HEAD)
test "$(git -C "$local_repo" branch --show-current)" = "$branch"
test -z "$(git -C "$local_repo" status --porcelain=v1 --untracked-files=all)"

ssh tillicum bash -s -- "$commit" <<'REMOTE'
set -euo pipefail
umask 077
ulimit -c 0
expected=$1
root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-ratio-panel-judge-v1
output=$root/outputs/massive_medical_ratio_panels_v1_judge
source=$root/outputs/massive_medical_ratio_panels_v1_evaluation
python=$root/envs/subliminal-mitigate-py311/bin/python
runner=scripts/judge_massive_medical_ratio_panels_v1.py
manifest=$output/control/PREP.json

[[ ! ${OPENAI_API_KEY+x} && ! ${BASH_ENV+x} && ! ${ENV+x} ]]
[[ ! -e $repo && ! -L $repo && ! -e $output && ! -L $output ]]
test -s "$source/control/EVALUATION_COMPLETE.json"
test ! -L "$source/control/EVALUATION_COMPLETE.json"
test -x "$python"

# Create the exact fresh checkout privately before cloning, clearing inherited
# setgid on this newly owned empty directory, never a shared parent directory.
mkdir -m 0700 "$repo"
chmod 0700 "$repo"
git clone --single-branch --branch claire/massive-medical-ratio-panel-judge-v1 \
  https://github.com/ClaireJieZhang/emergent_misalignment_mitigation.git "$repo"
test "$(git -C "$repo" rev-parse HEAD)" = "$expected"
test -z "$(git -C "$repo" status --porcelain=v1 --untracked-files=all)"

unset PYTHONPATH PYTHONHOME
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENAI_LOG=off
cd "$repo"
bash -n scripts/stage_massive_medical_ratio_panels_v1_judge_tillicum.sh
bash -n scripts/finalize_massive_medical_ratio_panels_v1_judge_tillicum.sh
bash -n scripts/status_massive_medical_ratio_panels_v1_judge_tillicum.sh
"$python" -B -m unittest tests.test_massive_medical_ratio_panels_v1_judge \
  tests.test_massive_medical_ratio_panels_v1_judge_wrappers
"$python" -B "$runner" prepare --output-root "$output" \
  --repo-root "$repo" --source-root "$source"
"$python" -B "$runner" audit-stage --manifest "$manifest"
"$python" -B "$runner" validate-sdk-serialization --manifest "$manifest" --stage canary
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test ! -e "$output/control/CANARY_AUTHORIZATION.json"
test ! -e "$output/control/CANARY_RUN_STARTED.json"
test ! -e "$output/control/CONTINUATION_AUTHORIZATION.json"
test ! -e "$output/control/CONTINUATION_RUN_STARTED.json"
REMOTE

echo RATIO_MEDICAL_JUDGE_CPU_READY_NO_PAID_ENTRY
