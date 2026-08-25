#!/bin/bash
# Submit exactly one held-first 95-minute medical job, only after sealed benefit PASS.
set -euo pipefail
umask 077
ulimit -c 0
[[ $# -eq 9 && $1 == medical && $2 == --ack-prior-program-actual-usd && $3 == 1.696936 && $4 == --ack-benefit-actual-usd && $6 == --ack-max-cost-usd && $7 == 1.425 && $8 == --ack-exact-cumulative-cap-usd && $9 == 4.096936 ]] || {
  echo 'Usage: submit_..._medical_tillicum.sh medical --ack-prior-program-actual-usd 1.696936 --ack-benefit-actual-usd <sealed-value> --ack-max-cost-usd 1.425 --ack-exact-cumulative-cap-usd 4.096936' >&2; exit 2
}
benefit_actual=$5
root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-composition-exploratory-sequential-confirmation-v1
output=$root/outputs/massive_medical_union_composition_exploratory_sequential_confirmation_v1
control=$output/control; logs=$root/outputs/logs
auditor=$repo/scripts/audit_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py
batch=scripts/sbatch_massive_medical_union_composition_exploratory_sequential_confirmation_v1_medical_tillicum_h200.sbatch
lock=$control/MEDICAL_SUBMISSION_LOCK; attempt=$control/MEDICAL_SUBMISSION_ATTEMPT.tsv
submitted=$control/MEDICAL_SUBMITTED; released=$control/MEDICAL_RELEASE_AUTHORIZED
cd "$repo"
module load conda/Miniforge3-25.3.1-3
conda activate "$root/envs/subliminal-mitigate-py311"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=$root/tmp/mmu-seq-medical-submit-pyc
# Eligibility is asserted before any lock or sbatch side effect.
python "$auditor" assert-medical-release --ack-benefit-actual-usd "$benefit_actual"
python "$auditor" audit-preflight --stage medical
for path in "$lock" "$attempt" "$submitted" "$released" "$control/MEDICAL_JOB.json" "$control/MEDICAL_AUTHORIZATION.json" "$control/MEDICAL_RESULT.json" "$control/STOPPED_medical" "$output/generation/medical" "$output/evaluation/medical"; do test ! -e "$path"; done
if compgen -G "$logs/massive_medical_union_composition_exploratory_sequential_confirmation_v1_medical_*" >/dev/null; then echo 'Medical log namespace is not fresh.' >&2; exit 4; fi
mkdir "$lock" || { echo 'Medical submission is permanently locked; retry is forbidden.' >&2; exit 3; }
printf 'workflow_id=massive_medical_union_composition_exploratory_sequential_confirmation_v1\nstage=medical\nrepo_commit=%s\n' "$(git rev-parse HEAD)" > "$lock/owner.tmp"
chmod 0400 "$lock/owner.tmp"; mv "$lock/owner.tmp" "$lock/owner"
raw_job=$(sbatch --parsable --hold --export=NONE --job-name=mmu_seq_medical_v1 "$batch")
job_id=${raw_job%%;*}; [[ $job_id =~ ^[0-9]+$ ]] || exit 5
released_ok=false
cleanup() { code=$?; if [[ $released_ok != true ]]; then state=$(squeue -h -j "$job_id" -o '%T|%r' 2>/dev/null || true); [[ $state != 'PENDING|JobHeldUser' ]] || scancel "$job_id" || true; fi; trap - EXIT; exit "$code"; }
trap cleanup EXIT
printf 'stage\tjob_id\nmedical\t%s\n' "$job_id" > "$attempt.tmp"; chmod 0400 "$attempt.tmp"; mv "$attempt.tmp" "$attempt"
python "$auditor" write-held-auth --stage medical --job-id "$job_id"
printf 'stage=medical\njob_id=%s\nheld_first=true\n' "$job_id" > "$submitted.tmp"; chmod 0400 "$submitted.tmp"; mv "$submitted.tmp" "$submitted"
python "$auditor" audit-held --stage medical --job-id "$job_id"
printf 'stage=medical\njob_id=%s\nheld_audit_passed=true\nrelease_authorized=true\n' "$job_id" > "$released.tmp"; chmod 0400 "$released.tmp"; mv "$released.tmp" "$released"
scontrol release "$job_id"; released_ok=true; trap - EXIT
echo "MEDICAL_JOB_ID=$job_id"
echo 'One 95-minute H200 medical job released; no dependency, requeue, retry, or API call.'
