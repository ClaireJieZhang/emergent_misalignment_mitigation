#!/bin/bash
# Release exactly one held-first 15-minute infrastructure-recovery smoke job.

set -euo pipefail
umask 077
ulimit -c 0

[[ $# -eq 7 && $1 == smoke-recovery && $2 == --ack-source-actual-cost-usd && $3 == 0.00875 && $4 == --ack-max-recovery-cost-usd && $5 == 0.225 && $6 == --ack-actual-plus-cap-cost-usd && $7 == 0.23375 ]] || {
  echo 'Usage: scripts/submit_massive_medical_union_composition_exploratory_smoke_recovery_v1_tillicum.sh smoke-recovery --ack-source-actual-cost-usd 0.00875 --ack-max-recovery-cost-usd 0.225 --ack-actual-plus-cap-cost-usd 0.23375' >&2
  exit 2
}

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-composition-exploratory-smoke-recovery-v1
output=$root/outputs/massive_medical_union_composition_exploratory_smoke_recovery_v1
control=$output/control
env_root=$root/envs/subliminal-mitigate-py311
auditor=$repo/scripts/audit_massive_medical_union_composition_exploratory_smoke_recovery_v1.py
batch=scripts/sbatch_massive_medical_union_composition_exploratory_smoke_recovery_v1_tillicum_h200.sbatch
lock=$control/SMOKE_RECOVERY_SUBMISSION_LOCK
attempt=$control/SMOKE_RECOVERY_SUBMISSION_ATTEMPT.tsv
submitted=$control/SMOKE_RECOVERY_SUBMITTED
release_auth=$control/SMOKE_RECOVERY_RELEASE_AUTHORIZED

cd "$repo"
unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY
unset TRANSFORMERS_CACHE
module load conda/Miniforge3-25.3.1-3
conda activate "$env_root"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=$root/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

python "$auditor" audit-prep
python "$auditor" audit-preflight
python "$auditor" audit-staged
python "$auditor" assert-submit-ready

mkdir "$lock" || {
  echo 'Recovery submission is permanently locked; retry/resubmission is forbidden.' >&2
  exit 3
}
owner_tmp=$lock/owner.tmp.$$
printf 'workflow_id=massive_medical_union_composition_exploratory_workflow_smoke_recovery_v1\nstage=smoke_recovery\nrepo_commit=%s\nsource_job_id=261152\nnew_versioned_recovery_not_retry=true\n' \
  "$(git rev-parse HEAD)" > "$owner_tmp"
chmod 0400 "$owner_tmp"
mv "$owner_tmp" "$lock/owner"

raw_job=$(sbatch --parsable --hold --export=NONE --job-name=mmu_cmpx_smoke_rec_v1 "$batch")
job_id=${raw_job%%;*}
[[ $job_id =~ ^[0-9]+$ ]] || {
  echo "Invalid recovery sbatch job id: $raw_job" >&2
  exit 5
}
release_completed=false
cancel_pristine_held_on_exit() {
  code=$?
  if [[ $release_completed != true ]]; then
    state_reason=$(squeue -h -j "$job_id" -o '%T|%r' 2>/dev/null || true)
    if [[ $state_reason == 'PENDING|JobHeldUser' ]]; then
      scancel "$job_id" || true
    fi
  fi
  trap - EXIT
  exit "$code"
}
trap cancel_pristine_held_on_exit EXIT

attempt_tmp=$attempt.tmp.$$
printf 'stage\tjob_id\nsmoke_recovery\t%s\n' "$job_id" > "$attempt_tmp"
chmod 0400 "$attempt_tmp"
mv "$attempt_tmp" "$attempt"

python "$auditor" write-held-auth --job-id "$job_id"
submitted_tmp=$submitted.tmp.$$
printf 'stage=smoke_recovery\njob_id=%s\nheld_first=true\nnew_versioned_recovery_not_retry=true\n' \
  "$job_id" > "$submitted_tmp"
chmod 0400 "$submitted_tmp"
mv "$submitted_tmp" "$submitted"
python "$auditor" audit-held --job-id "$job_id"

release_tmp=$release_auth.tmp.$$
printf 'stage=smoke_recovery\njob_id=%s\nheld_audit_passed=true\nrelease_authorized=true\n' \
  "$job_id" > "$release_tmp"
chmod 0400 "$release_tmp"
mv "$release_tmp" "$release_auth"
scontrol release "$job_id"
release_completed=true
trap - EXIT

echo "SMOKE_RECOVERY_JOB_ID=$job_id"
echo 'One 15-minute H200 recovery job released; no retry, confirmation, dependency, training, or API call.'
