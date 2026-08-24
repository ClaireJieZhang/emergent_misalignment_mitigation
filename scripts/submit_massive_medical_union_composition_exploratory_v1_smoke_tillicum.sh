#!/bin/bash
# Submit exactly one held-first 15-minute smoke job, then release it after audit.

set -euo pipefail
umask 077
ulimit -c 0

[[ $# -eq 3 && $1 == smoke && $2 == --ack-max-cost-usd && $3 == 0.225 ]] || {
  echo 'Usage: scripts/submit_massive_medical_union_composition_exploratory_v1_smoke_tillicum.sh smoke --ack-max-cost-usd 0.225' >&2
  exit 2
}

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-composition-exploratory-v1
output=$root/outputs/massive_medical_union_composition_exploratory_v1
control=$output/control
logs=$root/outputs/logs
env_root=$root/envs/subliminal-mitigate-py311
auditor=$repo/scripts/audit_massive_medical_union_composition_exploratory_workflow_v1.py
batch=scripts/sbatch_massive_medical_union_composition_exploratory_v1_smoke_tillicum_h200.sbatch
lock=$control/SMOKE_SUBMISSION_LOCK
attempt=$control/SMOKE_SUBMISSION_ATTEMPT.tsv
submitted=$control/SMOKE_SUBMITTED
release_auth=$control/SMOKE_RELEASE_AUTHORIZED

cd "$repo"
test -f "$control/STAGED"
test ! -e "$control/SMOKE_JOB.json"
test ! -e "$control/SMOKE_AUTHORIZED_MAX_COST_USD_0.225.json"
test ! -e "$control/SMOKE_RESULT.json"
test ! -e "$attempt"
test ! -e "$submitted"
test ! -e "$release_auth"
test ! -e "$output/generation/smoke"
test ! -e "$output/evaluation/smoke"
if compgen -G "$logs/massive_medical_union_composition_exploratory_v1_smoke_*" >/dev/null; then
  echo 'Smoke log namespace is not fresh.' >&2
  exit 4
fi

module load conda/Miniforge3-25.3.1-3
conda activate "$env_root"
export PYTHONDONTWRITEBYTECODE=1
python "$auditor" audit-prep
python "$auditor" audit-preflight --stage smoke
python "$auditor" audit-staged

mkdir "$lock" || {
  echo 'Smoke submission is permanently locked; retry/resubmission is forbidden.' >&2
  exit 3
}
owner_tmp=$lock/owner.tmp.$$
printf 'workflow_id=massive_medical_union_composition_exploratory_workflow_v1\nstage=smoke\nrepo_commit=%s\n' \
  "$(git rev-parse HEAD)" > "$owner_tmp"
chmod 0400 "$owner_tmp"
mv "$owner_tmp" "$lock/owner"

raw_job=$(sbatch --parsable --hold --export=NONE --job-name=mmu_cmpx_smoke_v1 "$batch")
job_id=${raw_job%%;*}
[[ $job_id =~ ^[0-9]+$ ]] || {
  echo "Invalid sbatch job id: $raw_job" >&2
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
printf 'stage\tjob_id\nsmoke\t%s\n' "$job_id" > "$attempt_tmp"
chmod 0400 "$attempt_tmp"
mv "$attempt_tmp" "$attempt"

python "$auditor" write-held-auth --stage smoke --job-id "$job_id"
submitted_tmp=$submitted.tmp.$$
printf 'stage=smoke\njob_id=%s\nheld_first=true\n' "$job_id" > "$submitted_tmp"
chmod 0400 "$submitted_tmp"
mv "$submitted_tmp" "$submitted"
python "$auditor" audit-held --stage smoke --job-id "$job_id"

release_tmp=$release_auth.tmp.$$
printf 'stage=smoke\njob_id=%s\nheld_audit_passed=true\nrelease_authorized=true\n' \
  "$job_id" > "$release_tmp"
chmod 0400 "$release_tmp"
mv "$release_tmp" "$release_auth"
scontrol release "$job_id"
release_completed=true
trap - EXIT

echo "SMOKE_JOB_ID=$job_id"
echo 'One 15-minute H200 smoke job released; no retry, dependency, or API call.'
