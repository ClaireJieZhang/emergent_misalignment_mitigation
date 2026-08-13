#!/bin/bash
# Called by the completed gate job. It allocates no continuation GPU unless the
# audited gate sentinel is GO. A one-shot directory lock prevents duplicate
# submissions after partial dispatcher failure.

set -euo pipefail
umask 077

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY

gate_job_id=${1:?Usage: dispatch_general_code_magicoder_lcb_tillicum.sh GATE_JOB_ID}
case "$gate_job_id" in
  *[!0-9]*|'')
    echo "Invalid gate job ID: $gate_job_id" >&2
    exit 2
    ;;
esac
if [[ "${SLURM_JOB_ID:-}" != "$gate_job_id" ]]; then
  echo "Dispatcher must be called by gate job $gate_job_id, not ${SLURM_JOB_ID:-outside-slurm}" >&2
  exit 2
fi

TILLICUM_ROOT=/gpfs/projects/stf/claizhan/subliminal-mitigate
REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/general_code_magicoder_lcb_q3_m4
CONTROL_ROOT=$OUTPUT_ROOT/control
GATE_ROOT=$CONTROL_ROOT/gate
JOBS_FILE=$CONTROL_ROOT/jobs.tsv
AUTH_FILE=$CONTROL_ROOT/AUTHORIZED_MAX_COST_USD_14.40
LOCK_DIR=$CONTROL_ROOT/dispatch.lock

cd "$REPO_ROOT"
test -s "$AUTH_FILE" || {
  echo "Missing exact USD 14.40 user authorization sentinel: $AUTH_FILE" >&2
  exit 2
}
grep -Fx 'ack_max_cost_usd=14.40' "$AUTH_FILE" >/dev/null
test -s "$OUTPUT_ROOT/gate/GATE_EVALUATION_COMPLETE"
authorized_commit=$(awk -F= '$1=="repo_commit" {print $2}' "$AUTH_FILE")
test -n "$authorized_commit"
test "$(git rev-parse HEAD)" = "$authorized_commit"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  if [[ -s "$CONTROL_ROOT/DISPATCHED" || -s "$CONTROL_ROOT/STOPPED_NO_GO" ]]; then
    echo "Dispatcher already reached a terminal state; no jobs resubmitted."
    exit 0
  fi
  echo "Dispatcher lock exists without a terminal sentinel. Refusing any possibly duplicate GPU submission: $LOCK_DIR" >&2
  exit 3
fi
printf 'gate_job_id=%s\nstarted_at=%s\n' "$gate_job_id" "$(date --iso-8601=seconds)" > "$LOCK_DIR/owner"

if [[ -s "$GATE_ROOT/NO_GO" ]]; then
  test ! -e "$GATE_ROOT/GO"
  stopped_build=$CONTROL_ROOT/.stopped-no-go-$gate_job_id
  printf 'gate_job_id=%s\nstopped_at=%s\nreason=%s\n' \
    "$gate_job_id" "$(date --iso-8601=seconds)" "$(tr '\n' ' ' < "$GATE_ROOT/NO_GO")" \
    > "$stopped_build"
  mv "$stopped_build" "$CONTROL_ROOT/STOPPED_NO_GO"
  echo "Pilot decision is NO_GO. No post-gate H200 job was submitted."
  echo 'Maximum realized H200 allocation ceiling: 3 H200-hours = $2.70.'
  exit 0
fi
test -s "$GATE_ROOT/GO" || {
  echo "Gate has neither GO nor NO_GO sentinel" >&2
  exit 3
}
test ! -e "$GATE_ROOT/NO_GO"

append_job() {
  local stage=$1
  local job_id=$2
  printf '%s\t%s\t%s\n' "$stage" "$job_id" "$(date --iso-8601=seconds)" >> "$JOBS_FILE"
}

echo "Pilot decision is GO. Submitting the authorized post-gate DAG."
train_1=$(sbatch --parsable --export=NONE --dependency="afterok:$gate_job_id" \
  scripts/sbatch_general_code_train_tillicum_h200.sbatch 1)
train_1=${train_1%%;*}
append_job train_pi_good_1 "$train_1"
train_2=$(sbatch --parsable --export=NONE --dependency="afterok:$gate_job_id" \
  scripts/sbatch_general_code_train_tillicum_h200.sbatch 2)
train_2=${train_2%%;*}
append_job train_pi_good_2 "$train_2"

model_dependency=afterok:$train_1:$train_2
direct_job=$(sbatch --parsable --export=NONE --dependency="$model_dependency" \
  scripts/sbatch_general_code_direct_final_tillicum_h200.sbatch)
direct_job=${direct_job%%;*}
append_job direct_final "$direct_job"
quorum_job=$(sbatch --parsable --export=NONE --dependency="$model_dependency" \
  scripts/sbatch_general_code_quorum_tillicum_h200.sbatch quorum)
quorum_job=${quorum_job%%;*}
append_job quorum_q3_m4_array "$quorum_job"
delta_job=$(sbatch --parsable --export=NONE --dependency="$model_dependency" \
  scripts/sbatch_general_code_quorum_tillicum_h200.sbatch pi_quorum_delta)
delta_job=${delta_job%%;*}
append_job pi_quorum_delta_q3_m4_array "$delta_job"

final_dependency=afterok:$direct_job:$quorum_job:$delta_job
final_job=$(sbatch --parsable --export=NONE --dependency="$final_dependency" \
  scripts/sbatch_general_code_final_evaluation_tillicum.sbatch)
final_job=${final_job%%;*}
append_job final_sandbox_evaluation "$final_job"

dispatched_build=$CONTROL_ROOT/.dispatched-$gate_job_id
printf 'gate_job_id=%s\ndispatched_at=%s\ntrain_1=%s\ntrain_2=%s\ndirect=%s\nquorum_array=%s\ndelta_array=%s\nfinal=%s\n' \
  "$gate_job_id" "$(date --iso-8601=seconds)" "$train_1" "$train_2" \
  "$direct_job" "$quorum_job" "$delta_job" "$final_job" > "$dispatched_build"
mv "$dispatched_build" "$CONTROL_ROOT/DISPATCHED"

echo "Post-gate jobs: train1=$train_1 train2=$train_2 direct=$direct_job quorum=$quorum_job delta=$delta_job final=$final_job"
echo 'Remaining authorized maximum: 13 H200-hours = $11.70.'
echo 'Whole-workflow maximum remains 16 H200-hours = $14.40.'
