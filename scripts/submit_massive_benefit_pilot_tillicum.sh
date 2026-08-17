#!/bin/bash
# Held-first, capped submission of the benefit-only MASSIVE DAG.

set -euo pipefail
umask 077

if [[ $# -ne 3 || "$1" != pilot || "$2" != --ack-max-cost-usd || "$3" != 2.93 ]]; then
  echo "Usage: $0 pilot --ack-max-cost-usd 2.93" >&2
  exit 2
fi

TILLICUM_ROOT=/gpfs/projects/stf/claizhan/subliminal-mitigate
REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/massive_benefit_pilot_v1
CONTROL_ROOT=$OUTPUT_ROOT/control
DATA_ROOT=$OUTPUT_ROOT/data
TRAINING_CONFIG=$REPO_ROOT/configs/training_qwen25_7b_massive_benefit_pilot.yaml
PREP_FILE=$CONTROL_ROOT/PREP_COMPLETE.json
AUTH_FILE=$CONTROL_ROOT/AUTHORIZED_MAX_COST_USD_2.93.json
JOBS_FILE=$CONTROL_ROOT/jobs.tsv
ATTEMPT_FILE=$CONTROL_ROOT/dispatch_attempt.tsv
SUBMITTED_FILE=$CONTROL_ROOT/SUBMITTED
RELEASED_FILE=$CONTROL_ROOT/RELEASED
LOCK_DIR=$CONTROL_ROOT/.submission-lock
ENV_ROOT=$TILLICUM_ROOT/envs/subliminal-mitigate-py311

cd "$REPO_ROOT"
test -z "$(git status --porcelain)"
test -s "$PREP_FILE"
test ! -e "$SUBMITTED_FILE"
test ! -e "$RELEASED_FILE"
test ! -e "$JOBS_FILE"
test ! -e "$ATTEMPT_FILE"
test ! -e "$AUTH_FILE"
mkdir "$LOCK_DIR" || {
  echo "Submission lock already exists; refusing duplicate dispatch." >&2
  exit 3
}
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

"$ENV_ROOT/bin/python" scripts/audit_massive_benefit_tillicum_workflow.py write-prep \
  --repo-root "$REPO_ROOT" --data-root "$DATA_ROOT" \
  --training-config "$TRAINING_CONFIG" --output-file "$PREP_FILE"

attempt_stages=()
attempt_ids=()
attempt_minutes=()
write_attempt() {
  local temporary=$CONTROL_ROOT/.dispatch-attempt-$$
  printf 'stage\tjob_id\tmax_minutes\theld\n' > "$temporary"
  local index
  for ((index=0; index<${#attempt_stages[@]}; index++)); do
    printf '%s\t%s\t%s\ttrue\n' \
      "${attempt_stages[$index]}" "${attempt_ids[$index]}" \
      "${attempt_minutes[$index]}" >> "$temporary"
  done
  mv "$temporary" "$ATTEMPT_FILE"
}

released=false
dispatch_failure() {
  local status=$?
  if (( status != 0 )) && [[ "$released" != true ]]; then
    echo "Dispatch stopped before release; recorded jobs must remain held:" >&2
    [[ -s "$ATTEMPT_FILE" ]] && cat "$ATTEMPT_FILE" >&2
  fi
  exit "$status"
}
trap dispatch_failure EXIT

base_job=$(sbatch --parsable --hold --export=NONE \
  scripts/sbatch_massive_benefit_base_dev_tillicum_h200.sbatch)
base_job=${base_job%%;*}
[[ "$base_job" =~ ^[0-9]+$ ]]
attempt_stages+=(base_dev); attempt_ids+=("$base_job"); attempt_minutes+=(30)
write_attempt

train_job=$(sbatch --parsable --hold --export=NONE --kill-on-invalid-dep=yes \
  --dependency="afterok:$base_job" \
  scripts/sbatch_massive_benefit_train_tillicum_h200.sbatch)
train_job=${train_job%%;*}
[[ "$train_job" =~ ^[0-9]+$ ]]
attempt_stages+=(train); attempt_ids+=("$train_job"); attempt_minutes+=(90)
write_attempt

evaluate_job=$(sbatch --parsable --hold --export=NONE --kill-on-invalid-dep=yes \
  --dependency="afterok:$train_job" \
  scripts/sbatch_massive_benefit_evaluate_tillicum_h200.sbatch)
evaluate_job=${evaluate_job%%;*}
[[ "$evaluate_job" =~ ^[0-9]+$ ]]
attempt_stages+=(evaluate); attempt_ids+=("$evaluate_job"); attempt_minutes+=(75)
write_attempt

audit_held_job() {
  local stage=$1 job_id=$2 minutes=$3 expected_dependency=$4
  local record dependency expected_limit node_range requested_tres
  record=$(scontrol show job "$job_id" -o | tr ' ' '\n')
  test "$(awk -F= '$1=="JobState" {print $2; exit}' <<< "$record")" = PENDING
  test "$(awk -F= '$1=="Reason" {print $2; exit}' <<< "$record")" = JobHeldUser
  test "$(awk -F= '$1=="Requeue" {print $2; exit}' <<< "$record")" = 0
  test "$(awk -F= '$1=="Account" {print $2; exit}' <<< "$record")" = stf
  test "$(awk -F= '$1=="Partition" {print $2; exit}' <<< "$record")" = gpu-h200
  case "$minutes" in
    30) expected_limit=00:30:00 ;;
    90) expected_limit=01:30:00 ;;
    75) expected_limit=01:15:00 ;;
    *) return 2 ;;
  esac
  test "$(awk -F= '$1=="TimeLimit" {print $2; exit}' <<< "$record")" = "$expected_limit"
  node_range=$(awk -F= '$1=="NumNodes" {print $2; exit}' <<< "$record")
  [[ "$node_range" = 1 || "$node_range" = 1-1 ]]
  test "$(awk -F= '$1=="NumTasks" {print $2; exit}' <<< "$record")" = 1
  requested_tres=$(sed -n 's/^ReqTRES=//p' <<< "$record")
  test "$(tr ',' '\n' <<< "$requested_tres" | awk -F= '$1=="gres/gpu:h200" {print $2}')" = 1
  test "$(tr ',' '\n' <<< "$requested_tres" | awk -F= '$1=="gres/gpu" {print $2}')" = 1
  if [[ -n "$expected_dependency" ]]; then
    dependency=$(awk -F= '$1=="Dependency" {print $2; exit}' <<< "$record")
    [[ "$dependency" = "afterok:$expected_dependency"* ]]
  fi
  echo "Audited held $stage job $job_id"
}
audit_held_job base_dev "$base_job" 30 ""
audit_held_job train "$train_job" 90 "$base_job"
audit_held_job evaluate "$evaluate_job" 75 "$train_job"

jobs_build=$CONTROL_ROOT/.jobs-$$
printf 'stage\tjob_id\tmax_minutes\treleased\n' > "$jobs_build"
printf 'base_dev\t%s\t30\ttrue\n' "$base_job" >> "$jobs_build"
printf 'train\t%s\t90\ttrue\n' "$train_job" >> "$jobs_build"
printf 'evaluate\t%s\t75\ttrue\n' "$evaluate_job" >> "$jobs_build"
mv "$jobs_build" "$JOBS_FILE"
chmod 0400 "$JOBS_FILE"

"$ENV_ROOT/bin/python" scripts/audit_massive_benefit_tillicum_workflow.py write-auth \
  --repo-root "$REPO_ROOT" --data-root "$DATA_ROOT" \
  --training-config "$TRAINING_CONFIG" --prep-file "$PREP_FILE" \
  --jobs-file "$JOBS_FILE" --output-file "$AUTH_FILE"

submitted_build=$CONTROL_ROOT/.submitted-$$
printf 'submitted_at=%s\nrepo_commit=%s\nbase_dev_job=%s\ntrain_job=%s\nevaluate_job=%s\nmaximum_h200_minutes=195\nmaximum_cost_usd_exact=2.925\nmaximum_cost_usd_display=2.93\nno_retries_or_reserve=true\nautomatic_medical_union_or_quorum=false\n' \
  "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)" "$base_job" \
  "$train_job" "$evaluate_job" > "$submitted_build"
mv "$submitted_build" "$SUBMITTED_FILE"

# Release downstream first; dependencies keep it blocked until upstream GO.
scontrol release "$evaluate_job"
scontrol release "$train_job"
scontrol release "$base_job"
released=true

released_build=$CONTROL_ROOT/.released-$$
printf 'released_at=%s\nbase_dev_job=%s\ntrain_job=%s\nevaluate_job=%s\nrelease_order=evaluate,train,base_dev\n' \
  "$(date --iso-8601=seconds)" "$base_job" "$train_job" "$evaluate_job" \
  > "$released_build"
mv "$released_build" "$RELEASED_FILE"
rm -f "$ATTEMPT_FILE"
rmdir "$LOCK_DIR"
trap - EXIT

echo "Submitted MASSIVE benefit pilot: base=$base_job train=$train_job eval=$evaluate_job"
echo 'Immutable maximum: 195 H200-minutes = $2.925 ($2.93 displayed).'
echo "No retry, reserve, union, extra adapter, quorum, or continuation was submitted."
