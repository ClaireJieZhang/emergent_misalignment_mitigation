#!/bin/bash
# Compact, read-only status report for the unattended Magicoder/LCB DAG.

set -euo pipefail

TILLICUM_ROOT=${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/general_code_magicoder_lcb_q3_m4
CONTROL_ROOT=$OUTPUT_ROOT/control
JOBS_FILE=$CONTROL_ROOT/jobs.tsv

echo "Output root: $OUTPUT_ROOT"
echo 'Authorized ceiling: 16 H200-hours = $14.40 at $0.90/H200-hour.'

echo
echo "=== Workflow decision ==="
if [[ -s "$CONTROL_ROOT/STOPPED_NO_GO" ]]; then
  echo "NO_GO: post-gate GPU jobs were not submitted."
  cat "$CONTROL_ROOT/STOPPED_NO_GO"
  echo 'Maximum allocation ceiling for this stopped path: 3 H200-hours = $2.70.'
elif [[ -s "$CONTROL_ROOT/DISPATCHED" ]]; then
  echo "GO: post-gate DAG was submitted."
  cat "$CONTROL_ROOT/DISPATCHED"
elif [[ -s "$CONTROL_ROOT/gate/GO" ]]; then
  echo "GO sentinel exists; dispatcher has not yet reached its terminal sentinel."
elif [[ -s "$CONTROL_ROOT/gate/NO_GO" ]]; then
  echo "NO_GO sentinel exists; dispatcher has not yet recorded STOPPED_NO_GO."
else
  echo "Pilot gate has not completed."
fi

echo
echo "=== Durable stage sentinels ==="
for sentinel in \
  "$OUTPUT_ROOT/PREP_COMPLETE" \
  "$OUTPUT_ROOT/models/pi_good_0/GENERAL_CODE_TRAIN_COMPLETE" \
  "$OUTPUT_ROOT/gate/GATE_EVALUATION_COMPLETE" \
  "$OUTPUT_ROOT/models/pi_good_1/GENERAL_CODE_TRAIN_COMPLETE" \
  "$OUTPUT_ROOT/models/pi_good_2/GENERAL_CODE_TRAIN_COMPLETE" \
  "$OUTPUT_ROOT/final/DIRECT_COMPLETE" \
  "$OUTPUT_ROOT/final/generations/quorum_q3_m4/chunk_0/JOB_COMPLETE" \
  "$OUTPUT_ROOT/final/generations/quorum_q3_m4/chunk_1/JOB_COMPLETE" \
  "$OUTPUT_ROOT/final/generations/pi_quorum_delta_q3_m4/chunk_0/JOB_COMPLETE" \
  "$OUTPUT_ROOT/final/generations/pi_quorum_delta_q3_m4/chunk_1/JOB_COMPLETE" \
  "$OUTPUT_ROOT/final/FINAL_EVALUATION_COMPLETE"; do
  if [[ -s "$sentinel" ]]; then
    printf 'DONE     %s\n' "${sentinel#$OUTPUT_ROOT/}"
  else
    printf 'PENDING  %s\n' "${sentinel#$OUTPUT_ROOT/}"
  fi
done

echo
echo "=== Recorded Slurm jobs ==="
if [[ -s "$JOBS_FILE" ]]; then
  column -t -s $'\t' "$JOBS_FILE" 2>/dev/null || cat "$JOBS_FILE"
  mapfile -t job_ids < <(awk 'NR>1 && $2 ~ /^[0-9]+$/ {print $2}' "$JOBS_FILE" | sort -u)
else
  echo "No jobs.tsv exists; the workflow has not been submitted."
  job_ids=()
fi

if (( ${#job_ids[@]} > 0 )); then
  id_csv=$(IFS=,; echo "${job_ids[*]}")
  echo
  echo "=== Active queue ==="
  squeue --jobs "$id_csv" \
    --format='%.18i %.24j %.2t %.10M %.10l %.4D %R' || true
  echo
  echo "=== Accounting (parents and array tasks) ==="
  sacct --jobs "$id_csv" \
    --format='JobID,JobName%24,State,Elapsed,Timelimit,AllocTRES%42,ExitCode' \
    --units=G || true
fi

if [[ -s "$OUTPUT_ROOT/gate/summary.md" ]]; then
  echo
  echo "=== Pilot summary ==="
  cat "$OUTPUT_ROOT/gate/summary.md"
fi
if [[ -s "$OUTPUT_ROOT/final/summary.md" ]]; then
  echo
  echo "=== Final summary ==="
  cat "$OUTPUT_ROOT/final/summary.md"
fi

echo
echo "Logs: $TILLICUM_ROOT/outputs/logs/general_code_*"
