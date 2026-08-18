#!/bin/bash
# Read-only status for the one-time MASSIVE offline-loader recovery.

set -euo pipefail

TILLICUM_ROOT=${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}
REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate
ENV_ROOT=$TILLICUM_ROOT/envs/subliminal-mitigate-py311
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/massive_benefit_pilot_v1
LOGS_ROOT=$TILLICUM_ROOT/outputs/logs
CONTROL_ROOT=$OUTPUT_ROOT/control
RECOVERY_ROOT=$CONTROL_ROOT/infrastructure_recovery_v1
RECOVERY_LOCK=$CONTROL_ROOT/INFRASTRUCTURE_RECOVERY_V1_SUBMISSION_LOCK
RECOVERY_JOBS=$RECOVERY_ROOT/jobs.tsv
RECOVERY_ATTEMPT=$RECOVERY_ROOT/dispatch_attempt.tsv
RECOVERY_ADDENDUM=$RECOVERY_ROOT/AUTHORIZED_INFRASTRUCTURE_RECOVERY_WITHIN_ORIGINAL_CAP.json
RECOVERY_MODEL=$OUTPUT_ROOT/model/massive_en_benefit_pilot_infrastructure_recovery_v1
RECOVERY_EVAL=$OUTPUT_ROOT/evaluation/infrastructure_recovery_v1
LOCAL_MODEL_PATH=$TILLICUM_ROOT/cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75

echo "Output root: $OUTPUT_ROOT"
echo 'Original MASSIVE authorization: 195 H200-minutes / $2.925.'
echo 'Prior conservative usage: base 2m + failed train 1m + cancelled eval 0m = 3m.'
echo 'Recovery cap: train 90m + evaluation 75m; cumulative maximum 168m / $2.520.'
echo 'Remaining below original ceiling: 27 H200-minutes / $0.405.'
echo 'Base development rerun, requeue, further retry, union, extra adapter, and quorum: disabled.'

echo
echo "=== Immutable original evidence ==="
for record in \
  "$CONTROL_ROOT/PREP_COMPLETE.json" \
  "$CONTROL_ROOT/AUTHORIZED_MAX_COST_USD_2.93.json" \
  "$CONTROL_ROOT/jobs.tsv" \
  "$CONTROL_ROOT/SUBMITTED" \
  "$CONTROL_ROOT/RELEASED" \
  "$OUTPUT_ROOT/evaluation/scores/massive_en_dev__pi_base.json" \
  "$OUTPUT_ROOT/evaluation/base_development/summary.json" \
  "$CONTROL_ROOT/GO_MASSIVE_BASE_DEV"; do
  if [[ -s "$record" ]]; then
    printf 'PRESENT  %s\n' "${record#$OUTPUT_ROOT/}"
  else
    printf 'MISSING  %s\n' "${record#$OUTPUT_ROOT/}"
  fi
done
sacct -X --starttime 2026-08-17 --jobs 237935,237936,237937 \
  --format='JobID,JobName%24,State,Elapsed,ElapsedRaw,Timelimit,AllocTRES%42,ExitCode' \
  --units=G || true

echo
echo "=== Versioned recovery control ==="
for record in \
  "$RECOVERY_LOCK/owner" \
  "$RECOVERY_ATTEMPT" \
  "$RECOVERY_JOBS" \
  "$RECOVERY_ADDENDUM" \
  "$RECOVERY_ROOT/SUBMITTED" \
  "$RECOVERY_ROOT/RELEASED" \
  "$RECOVERY_ROOT/TRAIN_STARTED" \
  "$RECOVERY_MODEL/TRAIN_COMPLETE" \
  "$RECOVERY_MODEL/MODEL_MANIFEST.json" \
  "$RECOVERY_ROOT/EVALUATE_STARTED" \
  "$RECOVERY_EVAL/selection/summary.json" \
  "$RECOVERY_ROOT/GO_MASSIVE_SEALED_TEST" \
  "$RECOVERY_EVAL/sealed_final/summary.json" \
  "$RECOVERY_ROOT/GO_MASSIVE_BENEFIT_ONLY"; do
  if [[ -s "$record" ]]; then
    printf 'DONE     %s\n' "${record#$OUTPUT_ROOT/}"
  else
    printf 'PENDING  %s\n' "${record#$OUTPUT_ROOT/}"
  fi
done

if [[ -s "$RECOVERY_ADDENDUM" && -s "$RECOVERY_JOBS" ]]; then
  echo
  echo "=== Recovery seal audit ==="
  set +e
  PYTHONDONTWRITEBYTECODE=1 "$ENV_ROOT/bin/python" \
    "$REPO_ROOT/scripts/audit_massive_benefit_infrastructure_recovery_v1.py" \
    verify-control \
    --repo-root "$REPO_ROOT" \
    --tillicum-root "$TILLICUM_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --logs-root "$LOGS_ROOT" \
    --jobs-file "$RECOVERY_JOBS" \
    --local-model-path "$LOCAL_MODEL_PATH" \
    --addendum-file "$RECOVERY_ADDENDUM"
  audit_status=$?
  set -e
  if (( audit_status != 0 )); then
    echo "RECOVERY_CONTROL_AUDIT_FAILED"
  fi
fi

job_record=$RECOVERY_JOBS
if [[ ! -s "$job_record" && -s "$RECOVERY_ATTEMPT" ]]; then
  job_record=$RECOVERY_ATTEMPT
fi
mapfile -t job_ids < <(
  if [[ -s "$job_record" ]]; then
    awk -F'\t' 'NR>1 && $2 ~ /^[0-9]+$/ {print $2}' "$job_record"
  fi
)
train_job_id=""
evaluate_job_id=""
train_state=""
evaluate_state=""
train_exit=""
evaluate_exit=""
if (( ${#job_ids[@]} > 0 )); then
  id_csv=$(IFS=,; echo "${job_ids[*]}")
  echo
  echo "=== Recovery jobs ==="
  column -t -s $'\t' "$job_record" 2>/dev/null || cat "$job_record"
  echo
  echo "=== Active queue ==="
  squeue --jobs "$id_csv" \
    --format='%.18i %.26j %.2t %.10M %.10l %.4D %R' || true
  echo
  echo "=== Recovery accounting ==="
  sacct -X --jobs "$id_csv" \
    --format='JobID,JobName%26,State,Elapsed,ElapsedRaw,Timelimit,AllocTRES%42,ExitCode' \
    --units=G || true
  train_job_id=$(awk -F'\t' '$1=="train" {print $2; exit}' "$job_record")
  evaluate_job_id=$(awk -F'\t' '$1=="evaluate" {print $2; exit}' "$job_record")
  if [[ -n "$train_job_id" ]]; then
    IFS='|' read -r train_state train_exit < <(
      sacct -X -n -P --jobs "$train_job_id" --format=State,ExitCode \
        | awk -F'|' 'NF >= 2 {print $1 "|" $2; exit}'
    ) || true
  fi
  if [[ -n "$evaluate_job_id" ]]; then
    IFS='|' read -r evaluate_state evaluate_exit < <(
      sacct -X -n -P --jobs "$evaluate_job_id" --format=State,ExitCode \
        | awk -F'|' 'NF >= 2 {print $1 "|" $2; exit}'
    ) || true
  fi
fi

for report in \
  "$OUTPUT_ROOT/evaluation/base_development/summary.md" \
  "$RECOVERY_EVAL/selection/summary.md" \
  "$RECOVERY_EVAL/sealed_final/summary.md"; do
  if [[ -s "$report" ]]; then
    echo
    echo "=== ${report#$OUTPUT_ROOT/evaluation/} ==="
    cat "$report"
  fi
done

is_terminal_failure() {
  case "$1" in
    FAILED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*|CANCELLED*) return 0 ;;
    *) return 1 ;;
  esac
}

echo
echo "=== Recovery decision ==="
if [[ -s "$RECOVERY_ROOT/GO_MASSIVE_BENEFIT_ONLY" ]]; then
  echo "FINAL_EVALUATION_COMPLETE: GO_MASSIVE_BENEFIT_ONLY"
  echo "The task-specific MASSIVE benefit gate passed; no union or quorum ran."
elif [[ -s "$RECOVERY_ROOT/STOPPED_MASSIVE_FINAL" ]]; then
  echo "FINAL_EVALUATION_COMPLETE: STOPPED_MASSIVE_FINAL"
elif [[ -s "$RECOVERY_ROOT/STOPPED_MASSIVE_SELECTION" ]]; then
  echo "STOPPED_MASSIVE_SELECTION: development gate failed; sealed test was not scored."
elif [[ -s "$RECOVERY_ROOT/GO_MASSIVE_SEALED_TEST" ]]; then
  echo "Development checkpoint selected; paired sealed-test scoring is pending."
elif is_terminal_failure "$train_state" && [[ ! -s "$RECOVERY_MODEL/TRAIN_COMPLETE" ]]; then
  echo "STOPPED_RECOVERY_TRAIN_JOB: job $train_job_id state=$train_state exit=$train_exit"
elif is_terminal_failure "$evaluate_state"; then
  echo "STOPPED_RECOVERY_EVALUATE_JOB: job $evaluate_job_id state=$evaluate_state exit=$evaluate_exit"
elif [[ -s "$RECOVERY_MODEL/TRAIN_COMPLETE" ]]; then
  echo "Recovered training complete; evaluation is pending or running."
elif [[ -s "$RECOVERY_ROOT/TRAIN_STARTED" ]]; then
  echo "Recovered training is pending or running."
elif [[ -s "$RECOVERY_ROOT/RELEASED" ]]; then
  echo "Two-job recovery DAG released; training is pending or running."
elif [[ -s "$RECOVERY_ATTEMPT" ]]; then
  echo "Recovery dispatch incomplete; recorded jobs must remain held."
elif [[ -s "$RECOVERY_LOCK/owner" ]]; then
  echo "Recovery exact-once lock exists; dispatch did not reach a recorded job."
else
  echo "One-time infrastructure recovery has not been submitted."
fi

echo
echo "Logs: $LOGS_ROOT/massive_benefit_infrastructure_recovery_v1_*"
