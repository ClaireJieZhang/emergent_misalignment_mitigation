#!/bin/bash
# Exact-once held-first release of Wave 1 only. This file has no Wave-2 path.

set -euo pipefail
umask 077

usage() {
  echo 'Usage: scripts/submit_massive_medical_union_wave1_tillicum.sh wave1 --ack-max-cost-usd 1.20' >&2
  exit 2
}

[[ $# -eq 3 && "$1" == wave1 && "$2" == --ack-max-cost-usd && "$3" == 1.20 ]] || usage

TILLICUM_ROOT=/gpfs/projects/stf/claizhan/subliminal-mitigate
REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate
ENV_ROOT=$TILLICUM_ROOT/envs/subliminal-mitigate-py311
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/massive_medical_union_pilot_v1
CONTROL_ROOT=$OUTPUT_ROOT/control
DATA_ROOT=$OUTPUT_ROOT/data
LOCAL_MODEL_SNAPSHOT=$TILLICUM_ROOT/cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75
BENEFIT_MODEL=$TILLICUM_ROOT/outputs/massive_benefit_pilot_v1/model/massive_en_benefit_pilot_infrastructure_recovery_v1
BENEFIT_CONTROL_ADAPTER=$BENEFIT_MODEL/checkpoint-30
BENEFIT_CONTROL_MANIFEST=$BENEFIT_MODEL/MODEL_MANIFEST.json
PREP_FILE=$CONTROL_ROOT/PREP_COMPLETE.json
JOBS_FILE=$CONTROL_ROOT/wave1_jobs.tsv
AUTH_FILE=$CONTROL_ROOT/AUTHORIZED_WAVE1_MAX_COST_USD_1.20.json
LOCK_DIR=$CONTROL_ROOT/WAVE1_SUBMISSION_LOCK
ATTEMPT_FILE=$CONTROL_ROOT/WAVE1_SUBMISSION_ATTEMPT.tsv

cd "$REPO_ROOT"
module load conda/Miniforge3-25.3.1-3
conda activate "$ENV_ROOT"
test -s "$CONTROL_ROOT/STAGED"
test -s "$PREP_FILE"
test ! -e "$JOBS_FILE"
test ! -e "$AUTH_FILE"
test ! -e "$CONTROL_ROOT/WAVE1_SUBMITTED"
test ! -e "$CONTROL_ROOT/WAVE1_RELEASED"
test ! -e "$CONTROL_ROOT/STOPPED_submission"
mkdir "$LOCK_DIR" || {
  echo "Permanent Wave-1 submission lock already exists; refusing a second dispatch." >&2
  exit 3
}
printf 'created_at=%s\nowner_pid=%s\nrepo_commit=%s\nhard_max_h200_minutes=80\nhard_max_cost_usd=1.20\n' \
  "$(date --iso-8601=seconds)" "$$" "$(git rev-parse HEAD)" > "$LOCK_DIR/owner"

declare -a submitted_stages=()
declare -a submitted_ids=()
release_started=false
record_attempt() {
  temporary=$CONTROL_ROOT/.wave1-submission-attempt-$$
  printf 'stage\tjob_id\n' > "$temporary"
  for index in "${!submitted_ids[@]}"; do
    printf '%s\t%s\n' "${submitted_stages[$index]}" "${submitted_ids[$index]}" >> "$temporary"
  done
  mv "$temporary" "$ATTEMPT_FILE"
}
record_failure() {
  status=$?
  if (( status != 0 )); then
    if [[ "$release_started" == true ]]; then
      for job_id in "${submitted_ids[@]}"; do
        scontrol hold "$job_id" >/dev/null 2>&1 || true
      done
    fi
    record_attempt
    temporary=$CONTROL_ROOT/.stopped-submission-$$
    printf 'stage=submission\nexit_status=%s\nstopped_at=%s\nrecorded_jobs=%s\nrelease_started=%s\nhold_requested_on_failure=true\nno_retry_authorized=true\n' \
      "$status" "$(date --iso-8601=seconds)" "${#submitted_ids[@]}" \
      "$release_started" > "$temporary"
    mv "$temporary" "$CONTROL_ROOT/STOPPED_submission"
  fi
  exit "$status"
}
trap record_failure EXIT

submit_held() {
  local stage=$1
  shift
  local raw job_id
  raw=$(sbatch --parsable --hold --export=NONE "$@")
  job_id=${raw%%;*}
  [[ "$job_id" =~ ^[0-9]+$ ]] || {
    echo "Invalid Slurm job ID for $stage: $raw" >&2
    return 4
  }
  submitted_stages+=("$stage")
  submitted_ids+=("$job_id")
  record_attempt
  SUBMITTED_JOB_ID=$job_id
}

SUBMITTED_JOB_ID=
submit_held train_A --job-name=mmu_train_A \
  scripts/sbatch_massive_medical_union_train_tillicum_h200.sbatch
train_a_job=$SUBMITTED_JOB_ID
submit_held train_B1 --job-name=mmu_train_B1 \
  scripts/sbatch_massive_medical_union_train_tillicum_h200.sbatch
train_b1_job=$SUBMITTED_JOB_ID
submit_held evaluate --job-name=mmu_w1_eval \
  --dependency="afterok:${train_a_job}:${train_b1_job}" --kill-on-invalid-dep=yes \
  scripts/sbatch_massive_medical_union_wave1_evaluate_tillicum_h200.sbatch
evaluate_job=$SUBMITTED_JOB_ID

audit_held_job() {
  local stage=$1 job_id=$2 expected_minutes=$3 expected_dependency=$4
  local record state reason requeue account partition time_limit nodes tasks dependency requested_tres
  record=$(scontrol show job "$job_id" -o | tr ' ' '\n')
  state=$(awk -F= '$1=="JobState" {print $2; exit}' <<< "$record")
  reason=$(awk -F= '$1=="Reason" {print $2; exit}' <<< "$record")
  requeue=$(awk -F= '$1=="Requeue" {print $2; exit}' <<< "$record")
  account=$(awk -F= '$1=="Account" {print $2; exit}' <<< "$record")
  partition=$(awk -F= '$1=="Partition" {print $2; exit}' <<< "$record")
  time_limit=$(awk -F= '$1=="TimeLimit" {print $2; exit}' <<< "$record")
  nodes=$(awk -F= '$1=="NumNodes" {print $2; exit}' <<< "$record")
  tasks=$(awk -F= '$1=="NumTasks" {print $2; exit}' <<< "$record")
  dependency=$(awk -F= '$1=="Dependency" {print $2; exit}' <<< "$record")
  requested_tres=$(sed -n 's/^ReqTRES=//p' <<< "$record")
  [[ "$state" == PENDING && "$reason" == JobHeldUser && "$requeue" == 0 ]]
  [[ "$account" == stf && "$partition" == gpu-h200 && "$nodes" == 1 && "$tasks" == 1 ]]
  [[ "$time_limit" == "00:${expected_minutes}:00" ]]
  [[ "$(tr ',' '\n' <<< "$requested_tres" | awk -F= '$1=="gres/gpu:h200" {print $2}')" == 1 ]]
  [[ "$(tr ',' '\n' <<< "$requested_tres" | awk -F= '$1=="gres/gpu" {print $2}')" == 1 ]]
  if [[ -n "$expected_dependency" ]]; then
    [[ "$dependency" == "$expected_dependency" ]]
  else
    [[ -z "$dependency" || "$dependency" == "(null)" ]]
  fi
  printf 'Audited held %s job %s (%sm).\n' "$stage" "$job_id" "$expected_minutes"
}

audit_held_job train_A "$train_a_job" 30 ""
audit_held_job train_B1 "$train_b1_job" 30 ""
audit_held_job evaluate "$evaluate_job" 20 "afterok:${train_a_job}:${train_b1_job}"

jobs_build=$CONTROL_ROOT/.wave1-jobs-$$
{
  printf 'stage\tjob_id\tmax_minutes\treleased\n'
  printf 'train_A\t%s\t30\ttrue\n' "$train_a_job"
  printf 'train_B1\t%s\t30\ttrue\n' "$train_b1_job"
  printf 'evaluate\t%s\t20\ttrue\n' "$evaluate_job"
} > "$jobs_build"
chmod 0400 "$jobs_build"
mv "$jobs_build" "$JOBS_FILE"

common_audit=(
  --repo-root "$REPO_ROOT" --data-root "$DATA_ROOT"
  --local-model-snapshot "$LOCAL_MODEL_SNAPSHOT" --prep-file "$PREP_FILE"
  --benefit-control-manifest "$BENEFIT_CONTROL_MANIFEST"
  --benefit-control-adapter "$BENEFIT_CONTROL_ADAPTER"
)
python scripts/audit_massive_medical_union_tillicum_workflow.py write-auth \
  "${common_audit[@]}" --jobs-file "$JOBS_FILE" --output-file "$AUTH_FILE"

submitted_build=$CONTROL_ROOT/.wave1-submitted-$$
printf 'submitted_at=%s\nrepo_commit=%s\ntrain_A_job=%s\ntrain_B1_job=%s\nevaluate_job=%s\nheld_first=true\nhard_max_h200_minutes=80\nhard_max_cost_usd=1.20\nwave2_jobs_submitted=false\nquorum_jobs_submitted=false\n' \
  "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)" "$train_a_job" \
  "$train_b1_job" "$evaluate_job" > "$submitted_build"
mv "$submitted_build" "$CONTROL_ROOT/WAVE1_SUBMITTED"

# Release downstream first. Dependencies still prevent evaluation from running.
release_started=true
scontrol release "$evaluate_job"
scontrol release "$train_b1_job"
scontrol release "$train_a_job"

released_build=$CONTROL_ROOT/.wave1-released-$$
printf 'released_at=%s\nrelease_order=%s,%s,%s\nhard_max_h200_minutes=80\nhard_max_cost_usd=1.20\nno_retry_authorized=true\nwave2_jobs_submitted=false\nquorum_jobs_submitted=false\n' \
  "$(date --iso-8601=seconds)" "$evaluate_job" "$train_b1_job" "$train_a_job" \
  > "$released_build"
mv "$released_build" "$CONTROL_ROOT/WAVE1_RELEASED"

trap - EXIT
echo "Released Wave 1 exactly once: A=$train_a_job B1=$train_b1_job eval=$evaluate_job"
echo 'Immutable maximum: 80 H200-minutes / $1.20. No Wave-2 or quorum job exists.'
