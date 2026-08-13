#!/bin/bash
# Run on Claire's Mac after the workflow commit is pushed. This updates the
# clean Tillicum checkout and stages only the existing bad-medical adapter.
# Public pinned data, source, and sandbox assets are built by the CPU prep job.

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
workspace_root=$(cd "$repo_root/.." && pwd)

TILLICUM_HOST=${TILLICUM_HOST:-tillicum}
TILLICUM_ROOT=${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}
REMOTE_REPO_URL=${REMOTE_REPO_URL:-https://github.com/ClaireJieZhang/emergent_misalignment_mitigation.git}
REMOTE_BRANCH=${REMOTE_BRANCH:-claire/capability-quorum-secure-code}
LOCAL_BAD_MODEL=${LOCAL_BAD_MODEL:-$workspace_root/hyak_results/em_qwen25_7b_bad_medical_vs_benign_medical/models/pi_A}
REMOTE_REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate
REMOTE_BAD_MODEL=$TILLICUM_ROOT/staged/bad_medical_pi_A
expected_commit=$(git -C "$repo_root" rev-parse HEAD)
BAD_CONFIG_SHA256=87b2798d8e7deabc5d13907f4e729bd54b5fb9c08401e5122d8988f7502bd643
BAD_WEIGHTS_SHA256=cdcf3125f2538009e06ecce4c2ab1e0cdc3e72317e540c0e807e169fe8820214

test -s "$LOCAL_BAD_MODEL/adapter_config.json" || {
  echo "Missing local bad-medical adapter: $LOCAL_BAD_MODEL" >&2
  exit 2
}
test "$(LC_ALL=C shasum -a 256 "$LOCAL_BAD_MODEL/adapter_config.json" | awk '{print $1}')" = "$BAD_CONFIG_SHA256"
test "$(LC_ALL=C shasum -a 256 "$LOCAL_BAD_MODEL/adapter_model.safetensors" | awk '{print $1}')" = "$BAD_WEIGHTS_SHA256"

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
mkdir -p \
  "$root/projects" \
  "$root/staged/bad_medical_pi_A" \
  "$root/outputs/logs" \
  "$root/cache" \
  "$root/config" \
  "$root/tmp"

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

observed_commit=$(git -C "$repo" rev-parse HEAD)
test "$observed_commit" = "$expected_commit" || {
  echo "Tillicum checkout $observed_commit does not match local expected commit $expected_commit" >&2
  exit 3
}
git -C "$repo" status --short --branch
git -C "$repo" log -1 --oneline
REMOTE

echo "=== Stage the existing bad-medical LoRA reference ==="
rsync -avP --delete "$LOCAL_BAD_MODEL/" "$TILLICUM_HOST:$REMOTE_BAD_MODEL/"

echo "=== Verify every unattended workflow entry point ==="
ssh "$TILLICUM_HOST" bash -s -- \
  "$TILLICUM_ROOT" "$BAD_CONFIG_SHA256" "$BAD_WEIGHTS_SHA256" <<'REMOTE'
set -euo pipefail
root=$1
bad_config_sha=$2
bad_weights_sha=$3
repo=$root/projects/subliminal-mitigate
test -f "$root/envs/subliminal-mitigate-py311/.ready"
test -s "$root/staged/bad_medical_pi_A/adapter_config.json"
echo "$bad_config_sha  $root/staged/bad_medical_pi_A/adapter_config.json" | sha256sum -c -
echo "$bad_weights_sha  $root/staged/bad_medical_pi_A/adapter_model.safetensors" | sha256sum -c -
for path in \
  scripts/prepare_magicoder_lcb_data.py \
  scripts/sample_lcb_direct_generations.py \
  scripts/sample_lcb_quorum_generations.py \
  scripts/merge_lcb_generation_chunks.py \
  scripts/prepare_lcb_evaluation.py \
  scripts/run_lcb_sandbox_evaluation.py \
  scripts/summarize_lcb_evaluations.py \
  scripts/audit_magicoder_lcb_models.py \
  scripts/sbatch_general_code_prepare_tillicum.sbatch \
  scripts/sbatch_general_code_train_tillicum_h200.sbatch \
  scripts/sbatch_general_code_gate_tillicum_h200.sbatch \
  scripts/sbatch_general_code_direct_final_tillicum_h200.sbatch \
  scripts/sbatch_general_code_quorum_tillicum_h200.sbatch \
  scripts/sbatch_general_code_final_evaluation_tillicum.sbatch \
  scripts/dispatch_general_code_magicoder_lcb_tillicum.sh \
  scripts/submit_general_code_magicoder_lcb_tillicum.sh \
  scripts/status_general_code_magicoder_lcb_tillicum.sh; do
  test -s "$repo/$path" || {
    echo "Missing staged workflow file: $path" >&2
    exit 2
  }
done
du -sh "$repo" "$root/staged/bad_medical_pi_A"
REMOTE

echo "Tillicum staging complete. No Slurm job was submitted."
echo "Next (on Tillicum): scripts/submit_general_code_magicoder_lcb_tillicum.sh overnight --ack-max-cost-usd 14.40"
