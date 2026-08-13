#!/bin/bash
# Update the clean Tillicum checkout and verify the repaired-pilot dependencies.
# This script does not submit a Slurm job.

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TILLICUM_HOST=${TILLICUM_HOST:-tillicum}
TILLICUM_ROOT=${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}
REMOTE_REPO_URL=${REMOTE_REPO_URL:-https://github.com/ClaireJieZhang/emergent_misalignment_mitigation.git}
REMOTE_BRANCH=${REMOTE_BRANCH:-claire/capability-quorum-secure-code}
REMOTE_REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate
expected_commit=$(git -C "$repo_root" rev-parse HEAD)

echo "=== Update the dedicated clean Tillicum checkout ==="
ssh "$TILLICUM_HOST" bash -s -- \
  "$TILLICUM_ROOT" "$REMOTE_REPO_URL" "$REMOTE_BRANCH" "$expected_commit" <<'REMOTE'
set -euo pipefail
root=$1
repo_url=$2
branch=$3
expected_commit=$4
repo=$root/projects/subliminal-mitigate

umask 077
mkdir -p "$root/projects" "$root/outputs/logs" "$root/cache" "$root/config" "$root/tmp"
if [[ -d "$repo/.git" ]]; then
  if [[ -n "$(git -C "$repo" status --porcelain)" ]]; then
    echo "Refusing to update dirty Tillicum checkout: $repo" >&2
    git -C "$repo" status --short >&2
    exit 3
  fi
  git -C "$repo" fetch origin "$branch"
  git -C "$repo" checkout -B "$branch" FETCH_HEAD
else
  git clone --branch "$branch" --single-branch "$repo_url" "$repo"
fi
test "$(git -C "$repo" rev-parse HEAD)" = "$expected_commit"
test -z "$(git -C "$repo" status --porcelain)"
git -C "$repo" log -1 --oneline
REMOTE

echo "=== Verify environment, sealed source assets, and workflow files ==="
ssh "$TILLICUM_HOST" bash -s -- "$TILLICUM_ROOT" "$expected_commit" <<'REMOTE'
set -euo pipefail
root=$1
expected_commit=$2
repo=$root/projects/subliminal-mitigate
lcb=$root/outputs/general_code_magicoder_lcb_q3_m4
evalplus=$root/outputs/general_code_evalplus_base_vs_pilot_v1
test "$(git -C "$repo" rev-parse HEAD)" = "$expected_commit"
test -z "$(git -C "$repo" status --porcelain)"
test -f "$root/envs/subliminal-mitigate-py311/.ready"
test -s "$lcb/PREP_COMPLETE"
test -s "$evalplus/DIAGNOSTIC_COMPLETE"
test -s "$lcb/data/lcb_gate_prompts.json"
test -s "$lcb/data/lcb_gate_evaluator.jsonl"
test -s "$lcb/data/lcb_final_prompts.json"
test -s "$evalplus/data/evalplus_prompts.json"
test "$(git -C "$lcb/assets/LiveCodeBench" rev-parse HEAD)" = 28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24
test -z "$(git -C "$lcb/assets/LiveCodeBench" status --porcelain)"
test "$(git -C "$evalplus/assets/evalplus-v0.3.1" rev-parse HEAD)" = e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2
test -z "$(git -C "$evalplus/assets/evalplus-v0.3.1" status --porcelain)"
sha256sum -c "$lcb/assets/python-3.11-slim-amd64.sif.sha256"
echo '272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101  '"$evalplus/assets/HumanEvalPlus-v0.1.10.jsonl.gz" | sha256sum -c -
echo 'af43697e8791c4c149bdfd6b489d8b5412507551ac20e28a439f650b8225db63  '"$evalplus/assets/MbppPlus-v0.2.0.jsonl.gz" | sha256sum -c -
for path in \
  configs/training_qwen25_7b_apps_repaired_pilot.yaml \
  scripts/prepare_repaired_code_pilot_data.py \
  scripts/audit_repaired_code_pilot_model.py \
  scripts/select_repaired_code_pilot_checkpoint.py \
  scripts/summarize_repaired_code_pilot.py \
  scripts/run_lcb_one_tillicum.sh \
  scripts/run_evalplus_one_tillicum.sh \
  scripts/sbatch_general_code_apps_repaired_prepare_tillicum_h200.sbatch \
  scripts/sbatch_general_code_apps_repaired_train_tillicum_h200.sbatch \
  scripts/sbatch_general_code_apps_repaired_evaluate_tillicum_h200.sbatch \
  scripts/submit_general_code_apps_repaired_pilot_tillicum.sh \
  scripts/status_general_code_apps_repaired_pilot_tillicum.sh; do
  test -s "$repo/$path" || { echo "Missing workflow file: $path" >&2; exit 2; }
done
bash -n "$repo"/scripts/*apps_repaired*tillicum*.sh
PYTHONPYCACHEPREFIX="$root/tmp/repaired-pilot-pyc" python3 -m py_compile \
  "$repo/scripts/prepare_repaired_code_pilot_data.py" \
  "$repo/scripts/audit_repaired_code_pilot_model.py" \
  "$repo/scripts/select_repaired_code_pilot_checkpoint.py" \
  "$repo/scripts/summarize_repaired_code_pilot.py"
echo "Non-GPU staging preflight passed."
REMOTE

echo "Tillicum staging complete. No Slurm job was submitted."
echo "Explicit submit command (not run):"
echo "  scripts/submit_general_code_apps_repaired_pilot_tillicum.sh pilot --ack-max-cost-usd 1.80"
