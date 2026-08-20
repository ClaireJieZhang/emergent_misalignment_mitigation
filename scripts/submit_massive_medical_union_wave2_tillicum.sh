#!/bin/bash
# Exact-once held-first dispatch of Wave 2 only: B2, B3, then direct evaluation.

set -euo pipefail
umask 077

usage() {
  echo 'Usage: scripts/submit_massive_medical_union_wave2_tillicum.sh wave2 --ack-max-cost-usd 1.125' >&2
  exit 2
}
[[ $# -eq 3 && "$1" == wave2 && "$2" == --ack-max-cost-usd && "$3" == 1.125 ]] || usage

TILLICUM_ROOT=/gpfs/projects/stf/claizhan/subliminal-mitigate
REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate-mmu-wave2
ENV_ROOT=$TILLICUM_ROOT/envs/subliminal-mitigate-py311
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/massive_medical_union_pilot_v1
CONTROL_ROOT=$OUTPUT_ROOT/control/wave2
EVAL_ROOT=$OUTPUT_ROOT/evaluation/wave2
MODEL_ROOT=$OUTPUT_ROOT/models
PREP_FILE=$CONTROL_ROOT/PREP.json
JOBS_FILE=$CONTROL_ROOT/jobs.tsv
AUTH_FILE=$CONTROL_ROOT/AUTHORIZED_MAX_COST_USD_1.125.json
LOCK_DIR=$CONTROL_ROOT/SUBMISSION_LOCK
ATTEMPT_FILE=$CONTROL_ROOT/SUBMISSION_ATTEMPT.tsv
TRAIN_SBATCH=scripts/sbatch_massive_medical_union_wave2_train_tillicum_h200.sbatch
EVAL_SBATCH=scripts/sbatch_massive_medical_union_wave2_evaluate_tillicum_h200.sbatch

cd "$REPO_ROOT"
test -s "$PREP_FILE"
test ! -e "$EVAL_ROOT"
test ! -e "$MODEL_ROOT/pi_B2"
test ! -e "$MODEL_ROOT/pi_B3"
for path in "$JOBS_FILE" "$AUTH_FILE" "$LOCK_DIR" "$ATTEMPT_FILE" \
  "$CONTROL_ROOT/SUBMITTED" "$CONTROL_ROOT/RELEASED" \
  "$CONTROL_ROOT/STOPPED_submission"; do
  test ! -e "$path"
done

module load conda/Miniforge3-25.3.1-3
conda activate "$ENV_ROOT"
export PYTHONPYCACHEPREFIX=$TILLICUM_ROOT/tmp/mmu-wave2-submit-pyc

mkdir "$LOCK_DIR" || {
  echo 'Permanent Wave-2 submission lock exists; refusing another dispatch.' >&2
  exit 3
}
printf 'created_at=%s\nowner_pid=%s\nrepo_commit=%s\nheld_first=true\njob_cap=3\nmax_h200_minutes=75\nmax_gpu_cost_usd=1.125\nnew_external_judge_cap_usd=0.50\nno_retry_or_reserve=true\nwave3_submitted_or_released=false\n' \
  "$(date --iso-8601=seconds)" "$$" "$(git rev-parse HEAD)" > "$LOCK_DIR/owner"
chmod 0400 "$LOCK_DIR/owner"

declare -a submitted_stages=()
declare -a submitted_ids=()
release_started=false
write_attempt() {
  temporary=$CONTROL_ROOT/.submission-attempt-$$
  printf 'stage\tjob_id\n' > "$temporary"
  for index in "${!submitted_ids[@]}"; do
    printf '%s\t%s\n' "${submitted_stages[$index]}" "${submitted_ids[$index]}" >> "$temporary"
  done
  chmod 0400 "$temporary"
  mv "$temporary" "$ATTEMPT_FILE"
}
record_failure() {
  status=$?
  if (( status != 0 )); then
    for job_id in "${submitted_ids[@]}"; do
      scontrol hold "$job_id" >/dev/null 2>&1 || true
    done
    write_attempt
    temporary=$CONTROL_ROOT/.stopped-submission-$$
    printf 'stage=submission\nexit_status=%s\nstopped_at=%s\nrecorded_jobs=%s\nrelease_started=%s\nhold_requested_on_all_recorded_jobs=true\nno_retry_or_reserve_authorized=true\nwave3_submitted_or_released=false\n' \
      "$status" "$(date --iso-8601=seconds)" "${#submitted_ids[@]}" \
      "$release_started" > "$temporary"
    chmod 0400 "$temporary"
    mv "$temporary" "$CONTROL_ROOT/STOPPED_submission"
  fi
  exit "$status"
}
trap record_failure EXIT

submit_held() {
  stage=$1
  shift
  raw=$(sbatch --parsable --hold --export=NONE "$@")
  job_id=${raw%%;*}
  [[ "$job_id" =~ ^[0-9]+$ ]] || {
    echo "Invalid Slurm ID for $stage: $raw" >&2
    return 4
  }
  submitted_stages+=("$stage")
  submitted_ids+=("$job_id")
  write_attempt
  SUBMITTED_JOB_ID=$job_id
}

SUBMITTED_JOB_ID=
submit_held train_B2 --job-name=mmu_w2_B2 "$TRAIN_SBATCH"
b2_job=$SUBMITTED_JOB_ID
submit_held train_B3 --job-name=mmu_w2_B3 "$TRAIN_SBATCH"
b3_job=$SUBMITTED_JOB_ID
submit_held evaluate --job-name=mmu_w2_eval \
  --dependency="afterok:${b2_job}:${b3_job}" --kill-on-invalid-dep=yes \
  "$EVAL_SBATCH"
eval_job=$SUBMITTED_JOB_ID

jobs_tmp=$CONTROL_ROOT/.jobs-$$
{
  printf 'stage\tjob_id\tmax_minutes\treleased\n'
  printf 'train_B2\t%s\t30\ttrue\n' "$b2_job"
  printf 'train_B3\t%s\t30\ttrue\n' "$b3_job"
  printf 'evaluate\t%s\t15\ttrue\n' "$eval_job"
} > "$jobs_tmp"
chmod 0400 "$jobs_tmp"
mv "$jobs_tmp" "$JOBS_FILE"

# Bind held requests, exact submit lines, dependencies, and Slurm-spooled bytes.
python scripts/audit_massive_medical_union_wave2.py write-auth

submitted_tmp=$CONTROL_ROOT/.submitted-$$
printf 'submitted_at=%s\nrepo_commit=%s\ntrain_B2_job=%s\ntrain_B3_job=%s\nevaluate_job=%s\nheld_first=true\nmax_h200_minutes=75\nmax_gpu_cost_usd=1.125\nno_retry_or_reserve=true\nwave3_submitted_or_released=false\n' \
  "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)" "$b2_job" \
  "$b3_job" "$eval_job" > "$submitted_tmp"
chmod 0400 "$submitted_tmp"
mv "$submitted_tmp" "$CONTROL_ROOT/SUBMITTED"

python scripts/audit_massive_medical_union_wave2.py audit-held

# Release downstream first; afterok still blocks evaluation until both replicas succeed.
release_started=true
scontrol release "$eval_job"
scontrol release "$b3_job"
scontrol release "$b2_job"

released_tmp=$CONTROL_ROOT/.released-$$
printf 'released_at=%s\nrelease_order=%s,%s,%s\nmax_h200_minutes=75\nmax_gpu_cost_usd=1.125\nno_retry_or_reserve=true\nwave3_submitted_or_released=false\n' \
  "$(date --iso-8601=seconds)" "$eval_job" "$b3_job" "$b2_job" > "$released_tmp"
chmod 0400 "$released_tmp"
mv "$released_tmp" "$CONTROL_ROOT/RELEASED"

trap - EXIT
echo "Released Wave 2 exactly once: B2=$b2_job B3=$b3_job eval=$eval_job"
echo 'Immutable GPU maximum: 75 H200-minutes / $1.125. No Wave-3 job exists.'
