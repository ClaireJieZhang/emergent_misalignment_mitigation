#!/bin/bash
# Run on Claire's Mac after the current Git branch has been pushed. This creates
# the dedicated Tillicum workspace, checks out the experiment branch, and
# stages the three small Klone-independent inputs already present on the Mac.

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
workspace_root=$(cd "$repo_root/.." && pwd)

TILLICUM_HOST="${TILLICUM_HOST:-tillicum}"
TILLICUM_ROOT="${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}"
REMOTE_REPO_URL="${REMOTE_REPO_URL:-https://github.com/ClaireJieZhang/emergent_misalignment_mitigation.git}"
REMOTE_BRANCH="${REMOTE_BRANCH:-claire/capability-quorum-secure-code}"

LOCAL_BAD_MODEL="${LOCAL_BAD_MODEL:-$workspace_root/hyak_results/em_qwen25_7b_bad_medical_vs_benign_medical/models/pi_A}"
LOCAL_BROAD_PROMPTS="${LOCAL_BROAD_PROMPTS:-$workspace_root/hyak_results/em_qwen25_7b_bad_medical_vs_benign_medical/datasets/eval/broad_prompts.json}"
LOCAL_MEDICAL_PROMPTS="${LOCAL_MEDICAL_PROMPTS:-$workspace_root/hyak_results/em_qwen25_7b_bad_medical_vs_benign_medical/majority_bad_medical_union_4bad_1good/narrow_medical_scaled_s5_a100_1gpu/medical_prompts.json}"

REMOTE_REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate
REMOTE_STAGED_ROOT=$TILLICUM_ROOT/staged

test -f "$LOCAL_BAD_MODEL/adapter_config.json" || {
  echo "Missing local bad-medical adapter: $LOCAL_BAD_MODEL" >&2
  exit 2
}
test -f "$LOCAL_BROAD_PROMPTS" || {
  echo "Missing local broad prompts: $LOCAL_BROAD_PROMPTS" >&2
  exit 2
}
test -f "$LOCAL_MEDICAL_PROMPTS" || {
  echo "Missing local medical prompts: $LOCAL_MEDICAL_PROMPTS" >&2
  exit 2
}

echo "=== Create/update the dedicated Tillicum checkout ==="
ssh "$TILLICUM_HOST" bash -s -- \
  "$TILLICUM_ROOT" "$REMOTE_REPO_URL" "$REMOTE_BRANCH" <<'REMOTE'
set -euo pipefail
root=$1
repo_url=$2
branch=$3
repo=$root/projects/subliminal-mitigate

umask 007
mkdir -p \
  "$root/projects" \
  "$root/staged/bad_medical_pi_A" \
  "$root/staged/prompts" \
  "$root/envs" \
  "$root/cache" \
  "$root/tmp" \
  "$root/outputs/logs"

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

git -C "$repo" status --short --branch
git -C "$repo" log -1 --oneline
REMOTE

echo "=== Stage the existing bad-medical LoRA adapter ==="
rsync -avP "$LOCAL_BAD_MODEL/" \
  "$TILLICUM_HOST:$REMOTE_STAGED_ROOT/bad_medical_pi_A/"

echo "=== Stage fixed broad and narrow-medical prompt banks ==="
rsync -avP "$LOCAL_BROAD_PROMPTS" \
  "$TILLICUM_HOST:$REMOTE_STAGED_ROOT/prompts/broad_prompts.json"
rsync -avP "$LOCAL_MEDICAL_PROMPTS" \
  "$TILLICUM_HOST:$REMOTE_STAGED_ROOT/prompts/medical_prompts.json"

echo "=== Verify staged inputs ==="
ssh "$TILLICUM_HOST" \
  "test -f '$REMOTE_REPO_ROOT/scripts/sbatch_em_train_bad_medical_secure_code_quorum_m4_tillicum_h200.sbatch' && test -f '$REMOTE_STAGED_ROOT/bad_medical_pi_A/adapter_config.json' && test -s '$REMOTE_STAGED_ROOT/prompts/broad_prompts.json' && test -s '$REMOTE_STAGED_ROOT/prompts/medical_prompts.json' && du -sh '$REMOTE_REPO_ROOT' '$REMOTE_STAGED_ROOT/bad_medical_pi_A' '$REMOTE_STAGED_ROOT/prompts'"

echo "Tillicum staging complete. No Slurm job was submitted."

