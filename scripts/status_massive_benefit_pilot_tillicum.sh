#!/bin/bash
# Read-only status for the capped, benefit-only MASSIVE English pilot.

set -euo pipefail

TILLICUM_ROOT=${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/massive_benefit_pilot_v1
CONTROL_ROOT=$OUTPUT_ROOT/control
MODEL_DIR=$OUTPUT_ROOT/model/massive_en_benefit_pilot
EVAL_ROOT=$OUTPUT_ROOT/evaluation
JOBS_FILE=$CONTROL_ROOT/jobs.tsv
ATTEMPT_FILE=$CONTROL_ROOT/dispatch_attempt.tsv

echo "Output root: $OUTPUT_ROOT"
echo 'MASSIVE-only hard maximum: 195 H200-minutes = $2.925 at $0.90/hour.'
echo 'Released stages: base development 30m; training 90m; evaluation 75m.'
echo 'Retries, reserve, medical unions, extra adapters, and quorum: disabled.'

echo
echo "=== Durable stages ==="
for record in \
  "$CONTROL_ROOT/STAGED" \
  "$CONTROL_ROOT/PREP_COMPLETE.json" \
  "$CONTROL_ROOT/AUTHORIZED_MAX_COST_USD_2.93.json" \
  "$CONTROL_ROOT/SUBMITTED" \
  "$CONTROL_ROOT/RELEASED" \
  "$EVAL_ROOT/base_development/summary.json" \
  "$CONTROL_ROOT/GO_MASSIVE_BASE_DEV" \
  "$MODEL_DIR/TRAIN_COMPLETE" \
  "$MODEL_DIR/MODEL_MANIFEST.json" \
  "$EVAL_ROOT/selection/summary.json" \
  "$CONTROL_ROOT/GO_MASSIVE_SEALED_TEST" \
  "$EVAL_ROOT/sealed_final/summary.json" \
  "$CONTROL_ROOT/GO_MASSIVE_BENEFIT_ONLY"; do
  if [[ -s "$record" ]]; then
    printf 'DONE     %s\n' "${record#$OUTPUT_ROOT/}"
  else
    printf 'PENDING  %s\n' "${record#$OUTPUT_ROOT/}"
  fi
done

echo
echo "=== Recorded Slurm jobs ==="
if [[ -s "$JOBS_FILE" ]]; then
  column -t -s $'\t' "$JOBS_FILE" 2>/dev/null || cat "$JOBS_FILE"
elif [[ -s "$ATTEMPT_FILE" ]]; then
  echo "Incomplete held-first dispatch; recorded jobs should remain held:"
  column -t -s $'\t' "$ATTEMPT_FILE" 2>/dev/null || cat "$ATTEMPT_FILE"
else
  echo "No jobs recorded; the GPU workflow has not been submitted."
fi

mapfile -t job_ids < <(
  for record in "$JOBS_FILE" "$ATTEMPT_FILE"; do
    if [[ -s "$record" ]]; then
      awk 'NR>1 && $2 ~ /^[0-9]+$/ {print $2}' "$record"
    fi
  done | sort -u
)
if (( ${#job_ids[@]} > 0 )); then
  id_csv=$(IFS=,; echo "${job_ids[*]}")
  echo
  echo "=== Active queue ==="
  squeue --jobs "$id_csv" \
    --format='%.18i %.24j %.2t %.10M %.10l %.4D %R' || true
  echo
  echo "=== Accounting ==="
  sacct --jobs "$id_csv" \
    --format='JobID,JobName%24,State,Elapsed,ElapsedRaw,Timelimit,AllocTRES%42,ExitCode' \
    --units=G || true
fi

for report in \
  "$EVAL_ROOT/base_development/summary.md" \
  "$EVAL_ROOT/selection/summary.md" \
  "$EVAL_ROOT/sealed_final/summary.md"; do
  if [[ -s "$report" ]]; then
    echo
    echo "=== ${report#$EVAL_ROOT/} ==="
    cat "$report"
  fi
done

echo
echo "=== Workflow decision ==="
if [[ -s "$CONTROL_ROOT/GO_MASSIVE_BENEFIT_ONLY" ]]; then
  echo "FINAL_EVALUATION_COMPLETE: GO_MASSIVE_BENEFIT_ONLY"
  echo "The task-specific MASSIVE benefit gate passed; no union or quorum ran."
elif [[ -s "$CONTROL_ROOT/STOPPED_MASSIVE_FINAL" ]]; then
  echo "FINAL_EVALUATION_COMPLETE: STOPPED_MASSIVE_FINAL"
elif [[ -s "$CONTROL_ROOT/STOPPED_MASSIVE_SELECTION" ]]; then
  echo "STOPPED_MASSIVE_SELECTION: development gate failed; test was not scored."
elif [[ -s "$CONTROL_ROOT/STOPPED_MASSIVE_BASE" ]]; then
  echo "STOPPED_MASSIVE_BASE: insufficient preregistered improvement headroom."
elif [[ -s "$CONTROL_ROOT/GO_MASSIVE_SEALED_TEST" ]]; then
  echo "Development checkpoint selected; cleaned test-subset scoring is pending."
elif [[ -s "$MODEL_DIR/TRAIN_COMPLETE" ]]; then
  echo "Training complete; development checkpoint evaluation is pending."
elif [[ -s "$CONTROL_ROOT/GO_MASSIVE_BASE_DEV" ]]; then
  echo "Base-development gate passed; training is pending or running."
elif [[ -s "$CONTROL_ROOT/RELEASED" ]]; then
  echo "Capped DAG released; base-development evaluation is pending or running."
elif [[ -s "$ATTEMPT_FILE" ]]; then
  echo "Dispatch incomplete; jobs are intended to remain held for manual audit."
elif [[ -s "$CONTROL_ROOT/STAGED" ]]; then
  echo "Non-GPU staging complete; no GPU jobs have been submitted."
else
  echo "Workflow has not been staged."
fi

echo
echo "Logs: $TILLICUM_ROOT/outputs/logs/massive_benefit_*"
