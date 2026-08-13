#!/bin/bash
set -euo pipefail

TILLICUM_ROOT=/gpfs/projects/stf/claizhan/subliminal-mitigate
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/general_code_evalplus_base_vs_pilot_v1
CONTROL_ROOT=$OUTPUT_ROOT/control
JOBS_FILE=$CONTROL_ROOT/jobs.tsv

echo "Output root: $OUTPUT_ROOT"
echo 'Authorized ceiling: 1 H200-hour = $0.90; no requeue or continuation.'
echo
if [[ -s "$OUTPUT_ROOT/DIAGNOSTIC_COMPLETE" ]]; then
  echo "DIAGNOSTIC_COMPLETE"
  cat "$OUTPUT_ROOT/DIAGNOSTIC_COMPLETE"
elif [[ -s "$CONTROL_ROOT/SUBMITTED" ]]; then
  echo "SUBMITTED / IN PROGRESS"
  cat "$CONTROL_ROOT/SUBMITTED"
  if [[ -s "$CONTROL_ROOT/RESUMED" ]]; then
    echo "RESUMED WITHIN ORIGINAL CAP"
    cat "$CONTROL_ROOT/RESUMED"
  fi
else
  echo "NOT_SUBMITTED"
fi

echo
echo "=== Recorded jobs ==="
if [[ -s "$JOBS_FILE" ]]; then
  column -t -s $'\t' "$JOBS_FILE"
else
  echo "No jobs recorded."
fi

job_ids=()
if [[ -s "$JOBS_FILE" ]]; then
  while IFS=$'\t' read -r _stage job_id _submitted; do
    [[ "$job_id" =~ ^[0-9]+$ ]] && job_ids+=("$job_id")
  done < <(tail -n +2 "$JOBS_FILE")
fi
if ((${#job_ids[@]})); then
  joined=$(IFS=,; echo "${job_ids[*]}")
  echo
  echo "=== Active queue ==="
  squeue -j "$joined" -o '%.18i %.24j %.2t %.10M %.10l %.8N %R' || true
  echo
  echo "=== Accounting ==="
  sacct -j "$joined" --format=JobID,JobName,State,Elapsed,Timelimit,AllocTRES,ExitCode || true
fi

echo
echo "=== Durable artifacts ==="
for path in \
  assets/ASSETS_READY \
  data/data_manifest.json \
  generations/pi_base.json \
  generations/pi_good_0.json \
  evaluations/humaneval/pi_base.json \
  evaluations/humaneval/pi_good_0.json \
  evaluations/mbpp/pi_base.json \
  evaluations/mbpp/pi_good_0.json \
  summary.json \
  DIAGNOSTIC_COMPLETE; do
  if [[ -s "$OUTPUT_ROOT/$path" ]]; then
    printf 'DONE     %s\n' "$path"
  else
    printf 'PENDING  %s\n' "$path"
  fi
done

if [[ -s "$OUTPUT_ROOT/summary.md" ]]; then
  echo
  echo "=== Diagnostic summary ==="
  cat "$OUTPUT_ROOT/summary.md"
fi

echo
echo "Logs: $TILLICUM_ROOT/outputs/logs/general_code_evalplus_diagnostic_*"
