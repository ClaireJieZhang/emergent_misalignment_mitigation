#!/bin/bash
# Read-only status for the capped K&K reasoning-benefit pilot.

set -euo pipefail

TILLICUM_ROOT=${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}
ENV_ROOT=$TILLICUM_ROOT/envs/subliminal-mitigate-py311
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/knights_knaves_reasoning_pilot_v1
CONTROL_ROOT=$OUTPUT_ROOT/control
MODEL_DIR=$OUTPUT_ROOT/model/kk_reasoning_n5_pilot
EVAL_ROOT=$OUTPUT_ROOT/evaluation
JOBS_FILE=$CONTROL_ROOT/jobs.tsv
ATTEMPT_FILE=$CONTROL_ROOT/dispatch_attempt.tsv

echo "Output root: $OUTPUT_ROOT"
echo 'Initial released maximum: 150 H200-minutes = $2.25 at $0.90/hour.'
echo 'Immutable cumulative ceiling: 240 H200-minutes = $3.60.'
echo 'Repair reserve: 90 H200-minutes; not automatically submitted.'
echo 'Automatic extra adapters, medical unions, and quorum: disabled.'

echo
echo "=== Durable stages ==="
for record in \
  "$CONTROL_ROOT/PREP_COMPLETE" \
  "$CONTROL_ROOT/AUTHORIZED_MAX_COST_USD_3.60" \
  "$CONTROL_ROOT/SUBMITTED" \
  "$CONTROL_ROOT/RELEASED" \
  "$MODEL_DIR/TRAIN_COMPLETE" \
  "$EVAL_ROOT/EVALUATION_PROVENANCE.json" \
  "$EVAL_ROOT/selection/summary.json" \
  "$CONTROL_ROOT/GO_KK_SEALED_FINAL" \
  "$EVAL_ROOT/sealed_final/summary.json" \
  "$CONTROL_ROOT/GO_KK_BENEFIT_UNIONS" \
  "$CONTROL_ROOT/STOPPED_NO_GO"; do
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
  squeue --jobs "$id_csv" --format='%.18i %.24j %.2t %.10M %.10l %.4D %R' || true
  echo
  echo "=== Accounting ==="
  sacct --jobs "$id_csv" \
    --format='JobID,JobName%24,State,Elapsed,ElapsedRaw,Timelimit,AllocTRES%42,ExitCode' \
    --units=G || true
fi

if [[ -s "$EVAL_ROOT/selection/summary.md" ]]; then
  echo
  echo "=== Development checkpoint-selection result ==="
  cat "$EVAL_ROOT/selection/summary.md"
fi

if [[ -s "$EVAL_ROOT/sealed_final/summary.md" ]]; then
  echo
  echo "=== Sealed-final result ==="
  cat "$EVAL_ROOT/sealed_final/summary.md"
fi

if [[ -s "$EVAL_ROOT/truncation_report.md" ]]; then
  echo
  echo "=== Generation truncation and parse coverage ==="
  cat "$EVAL_ROOT/truncation_report.md"
fi

echo
echo "=== Workflow decision ==="
if [[ -s "$CONTROL_ROOT/GO_KK_BENEFIT_UNIONS" ]]; then
  echo "FINAL_EVALUATION_COMPLETE: GO_KK_BENEFIT_UNIONS"
  echo "The benefit gate passed. This run did not submit unions or quorum."
elif [[ -s "$CONTROL_ROOT/STOPPED_NO_GO" ]]; then
  decision_phase=$("$ENV_ROOT/bin/python" - "$CONTROL_ROOT/STOPPED_NO_GO" <<'PY'
import json, os, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
summary = value.get("summary_file", "")
print("sealed-final" if "/sealed_final/" in summary else "development-selection")
PY
)
  echo "STOPPED_NO_GO at $decision_phase."
  echo "No matched union or quorum continuation is authorized."
elif [[ -s "$CONTROL_ROOT/GO_KK_SEALED_FINAL" ]]; then
  echo "Development GO recorded; sealed-final evaluation is pending or running."
elif [[ -s "$MODEL_DIR/TRAIN_COMPLETE" ]]; then
  echo "Training complete; development evaluation is pending or running."
elif [[ -s "$CONTROL_ROOT/RELEASED" ]]; then
  echo "Initial capped DAG released; training is pending or running."
elif [[ -s "$ATTEMPT_FILE" && ! -s "$CONTROL_ROOT/RELEASED" ]]; then
  echo "Dispatch incomplete; jobs are intended to remain held for manual audit."
else
  echo "No terminal decision yet."
fi

echo
echo "Logs: $TILLICUM_ROOT/outputs/logs/knights_knaves_reasoning_*"
