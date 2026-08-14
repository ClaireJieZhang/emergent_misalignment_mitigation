#!/bin/bash
# Read-only status for the capped repaired APPS pilot.

set -euo pipefail

TILLICUM_ROOT=${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/general_code_apps_repaired_pilot_v1
CONTROL_ROOT=$OUTPUT_ROOT/control
JOBS_FILE=$CONTROL_ROOT/jobs.tsv
RESUME_JOBS_FILE=$CONTROL_ROOT/resume_227440/jobs.tsv
RESUME_ATTEMPT_FILE=$CONTROL_ROOT/resume_227440/dispatch_attempt.tsv
COMPAT_JOBS_FILE=$CONTROL_ROOT/resume_227440_compat2/jobs.tsv
COMPAT_ATTEMPT_FILE=$CONTROL_ROOT/resume_227440_compat2/dispatch_attempt.tsv
REQTRES_JOBS_FILE=$CONTROL_ROOT/resume_227440_compat3/jobs.tsv
REQTRES_ATTEMPT_FILE=$CONTROL_ROOT/resume_227440_compat3/dispatch_attempt.tsv
IOSCHEMA_JOBS_FILE=$CONTROL_ROOT/resume_227440_compat4/jobs.tsv
IOSCHEMA_ATTEMPT_FILE=$CONTROL_ROOT/resume_227440_compat4/dispatch_attempt.tsv

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
  echo "Original submission:"
  column -t -s $'\t' "$JOBS_FILE" 2>/dev/null || cat "$JOBS_FILE"
else
  echo "No jobs.tsv exists; the workflow has not been submitted."
fi
if [[ -s "$RESUME_JOBS_FILE" ]]; then
  echo
  echo "Parser-repair resume (remaining 119-minute cap):"
  column -t -s $'\t' "$RESUME_JOBS_FILE" 2>/dev/null || cat "$RESUME_JOBS_FILE"
elif [[ -s "$RESUME_ATTEMPT_FILE" ]]; then
  echo
  echo "Incomplete held resume dispatch (no released preparation job):"
  column -t -s $'\t' "$RESUME_ATTEMPT_FILE" 2>/dev/null || cat "$RESUME_ATTEMPT_FILE"
fi
if [[ -s "$COMPAT_JOBS_FILE" ]]; then
  echo
  echo "Scheduler-compatibility dispatch (same remaining 119-minute cap):"
  column -t -s $'\t' "$COMPAT_JOBS_FILE" 2>/dev/null || cat "$COMPAT_JOBS_FILE"
elif [[ -s "$COMPAT_ATTEMPT_FILE" ]]; then
  echo
  echo "Incomplete held compatibility dispatch (no released preparation job):"
  column -t -s $'\t' "$COMPAT_ATTEMPT_FILE" 2>/dev/null || cat "$COMPAT_ATTEMPT_FILE"
fi
if [[ -s "$REQTRES_JOBS_FILE" ]]; then
  echo
  echo "ReqTRES-parser dispatch (same remaining 119-minute cap):"
  column -t -s $'\t' "$REQTRES_JOBS_FILE" 2>/dev/null || cat "$REQTRES_JOBS_FILE"
elif [[ -s "$REQTRES_ATTEMPT_FILE" ]]; then
  echo
  echo "Incomplete held ReqTRES dispatch (no released preparation job):"
  column -t -s $'\t' "$REQTRES_ATTEMPT_FILE" 2>/dev/null || cat "$REQTRES_ATTEMPT_FILE"
fi
if [[ -s "$IOSCHEMA_JOBS_FILE" ]]; then
  echo
  echo "APPS I/O-schema repair (118-minute remaining cap):"
  column -t -s $'\t' "$IOSCHEMA_JOBS_FILE" 2>/dev/null || cat "$IOSCHEMA_JOBS_FILE"
elif [[ -s "$IOSCHEMA_ATTEMPT_FILE" ]]; then
  echo
  echo "Incomplete held I/O-schema dispatch (no released preparation job):"
  column -t -s $'\t' "$IOSCHEMA_ATTEMPT_FILE" 2>/dev/null || cat "$IOSCHEMA_ATTEMPT_FILE"
fi
mapfile -t job_ids < <(
  for path in "$JOBS_FILE" "$RESUME_JOBS_FILE" "$RESUME_ATTEMPT_FILE" \
    "$COMPAT_JOBS_FILE" "$COMPAT_ATTEMPT_FILE" "$REQTRES_JOBS_FILE" \
    "$REQTRES_ATTEMPT_FILE" "$IOSCHEMA_JOBS_FILE" \
    "$IOSCHEMA_ATTEMPT_FILE"; do
    if [[ -s "$path" ]]; then
      awk 'NR>1 && $2 ~ /^[0-9]+$/ {print $2}' "$path"
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
