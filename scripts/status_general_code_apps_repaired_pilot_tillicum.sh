#!/bin/bash
# Read-only status for the capped repaired APPS pilot.

set -euo pipefail

TILLICUM_ROOT=${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/general_code_apps_repaired_pilot_v1
CONTROL_ROOT=$OUTPUT_ROOT/control
JOBS_FILE=$CONTROL_ROOT/jobs.tsv

echo "Output root: $OUTPUT_ROOT"
echo 'Authorized ceiling: 120 H200-minutes = $1.80 at $0.90/H200-hour.'
echo 'Automatic continuation: disabled (one adapter only; no quorum).'

echo
echo "=== Durable stages ==="
for sentinel in \
  "$OUTPUT_ROOT/PREP_COMPLETE" \
  "$OUTPUT_ROOT/model/apps_repaired_pilot/TRAIN_COMPLETE" \
  "$OUTPUT_ROOT/evaluation/apps_validation/SELECTED_CHECKPOINT.json" \
  "$OUTPUT_ROOT/evaluation/FINAL_EVALUATION_COMPLETE"; do
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
  squeue --jobs "$id_csv" --format='%.18i %.24j %.2t %.10M %.10l %.4D %R' || true
  echo
  echo "=== Accounting ==="
  sacct --jobs "$id_csv" \
    --format='JobID,JobName%24,State,Elapsed,ElapsedRaw,Timelimit,AllocTRES%42,ExitCode' \
    --units=G || true
fi

if [[ -s "$OUTPUT_ROOT/evaluation/apps_validation/SELECTED_CHECKPOINT.json" ]]; then
  echo
  echo "=== Frozen APPS selection ==="
  python3 - "$OUTPUT_ROOT/evaluation/apps_validation/SELECTED_CHECKPOINT.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
print("selected_checkpoint=" + value["selected_checkpoint"])
print("selected_step=" + str(value["selected_step"]))
print("selection_suite=" + value["selection_suite"])
PY
fi
if [[ -s "$OUTPUT_ROOT/evaluation/summary.md" ]]; then
  echo
  echo "=== Final report ==="
  cat "$OUTPUT_ROOT/evaluation/summary.md"
fi

echo
echo "Logs: $TILLICUM_ROOT/outputs/logs/general_code_apps_repaired_*"
