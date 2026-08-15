#!/bin/bash
# Submit the held-first, capped two-job K&K reasoning-benefit DAG.

set -euo pipefail
umask 077

if [[ "$#" -ne 3 || "$1" != pilot || "$2" != --ack-max-cost-usd || "$3" != 3.60 ]]; then
  cat >&2 <<'EOF'
Usage:
  scripts/submit_knights_knaves_reasoning_pilot_tillicum.sh pilot --ack-max-cost-usd 3.60

The immutable cumulative ceiling is 240 H200-minutes at $0.90/hour ($3.60).
This command initially releases only:
  one completion-only training job:        75 H200-minutes
  dev selection + conditional final job:   75 H200-minutes
Initial released maximum: 150 H200-minutes ($2.25).

The 90-minute remainder is a repair reserve and is NOT submitted here.  Both
initial jobs are held until the full afterok DAG and provenance are recorded,
use --no-requeue, and cannot launch arrays, further adapters, medical unions,
or quorum experiments.
EOF
  exit 2
fi

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY

TILLICUM_ROOT=/gpfs/projects/stf/claizhan/subliminal-mitigate
REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate
ENV_ROOT=$TILLICUM_ROOT/envs/subliminal-mitigate-py311
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/knights_knaves_reasoning_pilot_v1
CONTROL_ROOT=$OUTPUT_ROOT/control
DATA_ROOT=$OUTPUT_ROOT/data
TRAINING_CONFIG=$REPO_ROOT/configs/training_qwen25_7b_kk_reasoning_pilot.yaml
PREP_FILE=$CONTROL_ROOT/PREP_COMPLETE
AUTH_FILE=$CONTROL_ROOT/AUTHORIZED_MAX_COST_USD_3.60
JOBS_FILE=$CONTROL_ROOT/jobs.tsv
ATTEMPT_FILE=$CONTROL_ROOT/dispatch_attempt.tsv
SUBMITTED_FILE=$CONTROL_ROOT/SUBMITTED
RELEASED_FILE=$CONTROL_ROOT/RELEASED

cd "$REPO_ROOT"
mkdir -p "$CONTROL_ROOT" "$TILLICUM_ROOT/outputs/logs"
test -f "$ENV_ROOT/.ready" || {
  echo "Missing ready Tillicum environment: $ENV_ROOT" >&2
  exit 2
}
test -z "$(git status --porcelain)" || {
  echo "Refusing to submit from a dirty Tillicum checkout" >&2
  git status --short >&2
  exit 2
}
for path in \
  configs/training_qwen25_7b_kk_reasoning_pilot.yaml \
  scripts/audit_knights_knaves_tillicum_workflow.py \
  scripts/prepare_knights_knaves_pilot_data.py \
  scripts/sample_knights_knaves_generations.py \
  scripts/evaluate_knights_knaves_generations.py \
  scripts/summarize_knights_knaves_pilot.py \
  scripts/sbatch_knights_knaves_reasoning_pilot_train_tillicum_h200.sbatch \
  scripts/sbatch_knights_knaves_reasoning_pilot_evaluate_tillicum_h200.sbatch; do
  test -s "$path" || { echo "Missing workflow file: $path" >&2; exit 2; }
done

python scripts/audit_knights_knaves_tillicum_workflow.py verify-prep \
  --repo-root "$REPO_ROOT" --data-root "$DATA_ROOT" \
  --training-config "$TRAINING_CONFIG" --prep-file "$PREP_FILE"

# This persistent atomic lock is retained even after a preflight/dispatch
# failure.  It prevents two simultaneous acknowledgements from releasing two
# copies of the capped DAG.
SUBMISSION_LOCK=$CONTROL_ROOT/SUBMISSION_LOCK
if ! mkdir "$SUBMISSION_LOCK" 2>/dev/null; then
  echo "A K&K pilot submission attempt already owns $SUBMISSION_LOCK" >&2
  echo "Inspect status and held jobs before any manual recovery." >&2
  exit 3
fi
owner_build=$SUBMISSION_LOCK/.owner-$$
printf 'created_at=%s\nrepo_commit=%s\n' \
  "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)" > "$owner_build"
mv "$owner_build" "$SUBMISSION_LOCK/owner"

if [[ -e "$AUTH_FILE" || -e "$JOBS_FILE" || -e "$SUBMITTED_FILE" \
      || -e "$RELEASED_FILE" || -e "$ATTEMPT_FILE" ]]; then
  echo "This output root already contains submission state; refusing duplicates." >&2
  exit 3
fi

python scripts/audit_knights_knaves_tillicum_workflow.py write-authorization \
  --repo-root "$REPO_ROOT" --data-root "$DATA_ROOT" \
  --training-config "$TRAINING_CONFIG" --prep-file "$PREP_FILE" \
  --ack-max-cost-usd 3.60 --output-file "$AUTH_FILE"
chmod 0400 "$AUTH_FILE"

declare -a attempt_stages=()
declare -a attempt_ids=()
declare -a attempt_minutes=()
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
    echo "Dispatch stopped before release. Any recorded jobs remain held:" >&2
    if [[ -s "$ATTEMPT_FILE" ]]; then cat "$ATTEMPT_FILE" >&2; fi
    echo "Do not resubmit; inspect and recover the recorded held jobs." >&2
  fi
  exit "$status"
}
trap dispatch_failure EXIT

train_job=$(sbatch --parsable --hold --export=NONE \
  scripts/sbatch_knights_knaves_reasoning_pilot_train_tillicum_h200.sbatch)
train_job=${train_job%%;*}
[[ "$train_job" =~ ^[0-9]+$ ]]
attempt_stages+=(train)
attempt_ids+=("$train_job")
attempt_minutes+=(75)
write_attempt

evaluate_job=$(sbatch --parsable --hold --export=NONE --kill-on-invalid-dep=yes \
  --dependency="afterok:$train_job" \
  scripts/sbatch_knights_knaves_reasoning_pilot_evaluate_tillicum_h200.sbatch)
evaluate_job=${evaluate_job%%;*}
[[ "$evaluate_job" =~ ^[0-9]+$ ]]
attempt_stages+=(evaluate)
attempt_ids+=("$evaluate_job")
attempt_minutes+=(75)
write_attempt

audit_held_job() {
  local stage=$1 job_id=$2 expected_dependency=$3
  local record dependency
  record=$(scontrol show job "$job_id" -o | tr ' ' '\n')
  test "$(awk -F= '$1=="JobState" {print $2; exit}' <<< "$record")" = PENDING
  test "$(awk -F= '$1=="Reason" {print $2; exit}' <<< "$record")" = JobHeldUser
  test "$(awk -F= '$1=="Requeue" {print $2; exit}' <<< "$record")" = 0
  test "$(awk -F= '$1=="Account" {print $2; exit}' <<< "$record")" = stf
  test "$(awk -F= '$1=="Partition" {print $2; exit}' <<< "$record")" = gpu-h200
  test "$(awk -F= '$1=="TimeLimit" {print $2; exit}' <<< "$record")" = 01:15:00
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
audit_held_job train "$train_job" ""
audit_held_job evaluate "$evaluate_job" "$train_job"

# Record the entire authorized DAG atomically before either held job is
# released.  `released=true` records the authorized release state that the
# following downstream-first scontrol operations enact.
jobs_build=$CONTROL_ROOT/.jobs-$$
printf 'stage\tjob_id\tmax_minutes\treleased\n' > "$jobs_build"
printf 'train\t%s\t75\ttrue\n' "$train_job" >> "$jobs_build"
printf 'evaluate\t%s\t75\ttrue\n' "$evaluate_job" >> "$jobs_build"
mv "$jobs_build" "$JOBS_FILE"
chmod 0400 "$JOBS_FILE"

submitted_build=$CONTROL_ROOT/.submitted-$$
printf 'submitted_at=%s\nrepo_commit=%s\ntrain_job=%s\nevaluate_job=%s\ninitial_released_h200_minutes=150\nmax_h200_minutes=240\nreserve_h200_minutes=90\nreserve_submitted=false\nautomatic_medical_union_or_quorum=false\n' \
  "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)" "$train_job" \
  "$evaluate_job" > "$submitted_build"
mv "$submitted_build" "$SUBMITTED_FILE"

# Release downstream first: it remains dependency-blocked.  Only after that
# succeeds is the upstream training job released.
scontrol release "$evaluate_job"
scontrol release "$train_job"
released=true

released_build=$CONTROL_ROOT/.released-$$
printf 'released_at=%s\ntrain_job=%s\nevaluate_job=%s\nrelease_order=evaluate,train\n' \
  "$(date --iso-8601=seconds)" "$train_job" "$evaluate_job" > "$released_build"
mv "$released_build" "$RELEASED_FILE"
trap - EXIT

echo "Submitted and released K&K pilot: train=$train_job evaluate=$evaluate_job"
echo "Initial hard maximum: 150 H200-minutes = \$2.25."
echo "Cumulative ceiling: 240 H200-minutes = \$3.60; 90-minute reserve not submitted."
echo "No medical union, extra adapter, quorum, retry, or continuation job exists."
echo "Output root: $OUTPUT_ROOT"
