#!/bin/bash
# Submit exactly one held-first 65-minute H200 benefit job after explicit cost acknowledgement.
set -euo pipefail
umask 077
ulimit -c 0
[[ $# -eq 7 && $1 == benefit && $2 == --ack-prior-program-actual-usd && $3 == 1.696936 && $4 == --ack-max-cost-usd && $5 == 0.975 && $6 == --ack-exact-cumulative-max-usd && $7 == 2.671936 ]] || {
  echo 'Usage: submit_..._benefit_tillicum.sh benefit --ack-prior-program-actual-usd 1.696936 --ack-max-cost-usd 0.975 --ack-exact-cumulative-max-usd 2.671936' >&2; exit 2
}
root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-composition-exploratory-sequential-confirmation-v1
output=$root/outputs/massive_medical_union_composition_exploratory_sequential_confirmation_v1
control=$output/control; logs=$root/outputs/logs
auditor=$repo/scripts/audit_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py
batch=scripts/sbatch_massive_medical_union_composition_exploratory_sequential_confirmation_v1_benefit_tillicum_h200.sbatch
lock=$control/BENEFIT_SUBMISSION_LOCK; attempt=$control/BENEFIT_SUBMISSION_ATTEMPT.tsv
submitted=$control/BENEFIT_SUBMITTED; released=$control/BENEFIT_RELEASE_AUTHORIZED
cd "$repo"
for path in "$lock" "$attempt" "$submitted" "$released" "$control/BENEFIT_JOB.json" "$control/BENEFIT_AUTHORIZATION.json" "$control/BENEFIT_RESULT.json" "$control/STOPPED_benefit" "$output/generation/benefit" "$output/evaluation/benefit"; do test ! -e "$path"; done
if compgen -G "$logs/massive_medical_union_composition_exploratory_sequential_confirmation_v1_benefit_*" >/dev/null; then echo 'Benefit log namespace is not fresh.' >&2; exit 4; fi
module load conda/Miniforge3-25.3.1-3
conda activate "$root/envs/subliminal-mitigate-py311"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=$root/tmp/mmu-seq-benefit-submit-pyc
python "$auditor" audit-staged
python "$auditor" audit-preflight --stage benefit
mkdir "$lock" || { echo 'Benefit submission is permanently locked; retry is forbidden.' >&2; exit 3; }
printf 'workflow_id=massive_medical_union_composition_exploratory_sequential_confirmation_v1\nstage=benefit\nrepo_commit=%s\n' "$(git rev-parse HEAD)" > "$lock/owner.tmp"
chmod 0400 "$lock/owner.tmp"; mv "$lock/owner.tmp" "$lock/owner"
raw_job=$(sbatch --parsable --hold --export=NONE --job-name=mmu_seq_benefit_v1 "$batch")
job_id=${raw_job%%;*}; [[ $job_id =~ ^[0-9]+$ ]] || exit 5
released_ok=false
cleanup() { code=$?; if [[ $released_ok != true ]]; then state=$(squeue -h -j "$job_id" -o '%T|%r' 2>/dev/null || true); [[ $state != 'PENDING|JobHeldUser' ]] || scancel "$job_id" || true; fi; trap - EXIT; exit "$code"; }
trap cleanup EXIT
printf 'stage\tjob_id\nbenefit\t%s\n' "$job_id" > "$attempt.tmp"; chmod 0400 "$attempt.tmp"; mv "$attempt.tmp" "$attempt"
python "$auditor" write-held-auth --stage benefit --job-id "$job_id"
printf 'stage=benefit\njob_id=%s\nheld_first=true\n' "$job_id" > "$submitted.tmp"; chmod 0400 "$submitted.tmp"; mv "$submitted.tmp" "$submitted"
python "$auditor" audit-held --stage benefit --job-id "$job_id"
printf 'stage=benefit\njob_id=%s\nheld_audit_passed=true\nrelease_authorized=true\n' "$job_id" > "$released.tmp"; chmod 0400 "$released.tmp"; mv "$released.tmp" "$released"
scontrol release "$job_id"; released_ok=true; trap - EXIT
echo "BENEFIT_JOB_ID=$job_id"
echo 'One 65-minute H200 benefit job released; no dependency, requeue, retry, medical job, or API call.'
