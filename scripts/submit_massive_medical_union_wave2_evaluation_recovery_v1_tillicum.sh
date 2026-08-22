#!/bin/bash
# Exact-once held-first dispatch of one 15-minute Wave-2 evaluation recovery.

set -euo pipefail
umask 077

usage() {
  echo 'Usage: scripts/submit_massive_medical_union_wave2_evaluation_recovery_v1_tillicum.sh evaluation-recovery-v1 --ack-max-cost-usd 0.225' >&2
  exit 2
}
[[ $# -eq 3 && "$1" == evaluation-recovery-v1 && "$2" == --ack-max-cost-usd && "$3" == 0.225 ]] || usage

TILLICUM_ROOT=/gpfs/projects/stf/claizhan/subliminal-mitigate
REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate-mmu-wave2-eval-recovery-v1
ENV_ROOT=$TILLICUM_ROOT/envs/subliminal-mitigate-py311
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/massive_medical_union_pilot_v1
CONTROL_ROOT=$OUTPUT_ROOT/control/wave2_eval_recovery_v1
RECOVERED_MANIFEST_ROOT=$CONTROL_ROOT/models
EVAL_ROOT=$OUTPUT_ROOT/evaluation/wave2_eval_recovery_v1
PREP_FILE=$CONTROL_ROOT/PREP.json
JOBS_FILE=$CONTROL_ROOT/jobs.tsv
AUTH_FILE=$CONTROL_ROOT/AUTHORIZED_MAX_COST_USD_0.225.json
LOCK_DIR=$CONTROL_ROOT/SUBMISSION_LOCK
ATTEMPT_FILE=$CONTROL_ROOT/SUBMISSION_ATTEMPT.tsv
SBATCH_FILE=scripts/sbatch_massive_medical_union_wave2_evaluation_recovery_v1_tillicum_h200.sbatch
AUDITOR=scripts/audit_massive_medical_union_wave2_evaluation_recovery_v1.py

cd "$REPO_ROOT"
test -s "$PREP_FILE"
test -s "$CONTROL_ROOT/STAGED"
test -s "$RECOVERED_MANIFEST_ROOT/pi_B2/MODEL_MANIFEST.json"
test -s "$RECOVERED_MANIFEST_ROOT/pi_B3/MODEL_MANIFEST.json"
test ! -e "$EVAL_ROOT"
for path in "$JOBS_FILE" "$AUTH_FILE" "$LOCK_DIR" "$ATTEMPT_FILE" \
  "$CONTROL_ROOT/SUBMITTED" "$CONTROL_ROOT/RELEASED" \
  "$CONTROL_ROOT/STOPPED_submission"; do
  test ! -e "$path"
done

module load conda/Miniforge3-25.3.1-3
conda activate "$ENV_ROOT"
export PYTHONPYCACHEPREFIX=$TILLICUM_ROOT/tmp/mmu-wave2-eval-recovery-v1-submit-pyc

mkdir "$LOCK_DIR" || {
  echo 'Permanent evaluation-recovery submission lock exists; refusing another dispatch.' >&2
  exit 3
}
printf 'created_at=%s\nowner_pid=%s\nrepo_commit=%s\nheld_first=true\njob_cap=1\nmax_h200_minutes=15\nmax_gpu_cost_usd=0.225\nno_dependency=true\nno_retry_or_reserve=true\nretraining=false\nexternal_api_calls=0\nwave3_submitted_or_released=false\n' \
  "$(date --iso-8601=seconds)" "$$" "$(git rev-parse HEAD)" > "$LOCK_DIR/owner"
chmod 0400 "$LOCK_DIR/owner"

job_id=
release_started=false
write_attempt() {
  temporary=$CONTROL_ROOT/.submission-attempt-$$
  printf 'stage\tjob_id\n' > "$temporary"
  if [[ -n "$job_id" ]]; then
    printf 'evaluate_recovery_v1\t%s\n' "$job_id" >> "$temporary"
  fi
  chmod 0400 "$temporary"
  mv "$temporary" "$ATTEMPT_FILE"
}
record_failure() {
  status=$?
  if (( status != 0 )); then
    if [[ -n "$job_id" ]]; then
      scontrol hold "$job_id" >/dev/null 2>&1 || true
      write_attempt
    fi
    temporary=$CONTROL_ROOT/.stopped-submission-$$
    printf 'stage=submission\nexit_status=%s\nstopped_at=%s\njob_id=%s\nrelease_started=%s\nhold_requested_on_failure=true\nno_retry_authorized=true\nretraining=false\nexternal_api_calls=0\nwave3_submitted_or_released=false\n' \
      "$status" "$(date --iso-8601=seconds)" "${job_id:-NONE}" \
      "$release_started" > "$temporary"
    chmod 0400 "$temporary"
    mv "$temporary" "$CONTROL_ROOT/STOPPED_submission"
  fi
  exit "$status"
}
trap record_failure EXIT

raw=$(sbatch --parsable --hold --export=NONE --job-name=mmu_w2_evalrec_v1 "$SBATCH_FILE")
job_id=${raw%%;*}
[[ "$job_id" =~ ^[0-9]+$ ]]
write_attempt

jobs_build=$CONTROL_ROOT/.jobs-$$
{
  printf 'stage\tjob_id\tmax_minutes\treleased\n'
  printf 'evaluate_recovery_v1\t%s\t15\ttrue\n' "$job_id"
} > "$jobs_build"
chmod 0400 "$jobs_build"
mv "$jobs_build" "$JOBS_FILE"

python "$AUDITOR" write-auth

submitted_build=$CONTROL_ROOT/.submitted-$$
printf 'submitted_at=%s\nrepo_commit=%s\njob_id=%s\nheld_first=true\njob_cap=1\nmax_h200_minutes=15\nmax_gpu_cost_usd=0.225\nno_dependency=true\nno_retry_or_reserve=true\nretraining=false\nexternal_api_calls=0\nwave3_submitted_or_released=false\n' \
  "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)" "$job_id" > "$submitted_build"
chmod 0400 "$submitted_build"
mv "$submitted_build" "$CONTROL_ROOT/SUBMITTED"

python "$AUDITOR" audit-held

release_started=true
scontrol release "$job_id"

released_build=$CONTROL_ROOT/.released-$$
printf 'released_at=%s\njob_id=%s\nrelease_order=%s\nmax_h200_minutes=15\nmax_gpu_cost_usd=0.225\nno_retry_or_reserve=true\nretraining=false\nexternal_api_calls=0\nwave3_submitted_or_released=false\n' \
  "$(date --iso-8601=seconds)" "$job_id" "$job_id" > "$released_build"
chmod 0400 "$released_build"
mv "$released_build" "$CONTROL_ROOT/RELEASED"

trap - EXIT
echo "Released one exact-once evaluation-only recovery job: $job_id"
echo 'New ceiling: 15 H200-minutes / $0.225; cumulative Wave-2 released ceiling: 90 minutes / $1.35.'
echo 'No retraining, dependency, external API call, or Wave-3 allocation was submitted.'
