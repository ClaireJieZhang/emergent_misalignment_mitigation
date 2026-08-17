#!/bin/bash
# Read-only status for the K&K v3 direct robustness confirmation.

set -euo pipefail
ROOT=/gpfs/projects/stf/claizhan/subliminal-mitigate
V3=$ROOT/outputs/knights_knaves_reasoning_confirmation_v3
CONTROL=$V3/control

echo "Output root: $V3"
echo 'New hard cap: 30 H200-minutes = $0.45.'
echo 'Cumulative K&K released maximum including v1+v2: 210 H200-minutes = $3.15.'
echo 'V2 STOP is preserved; V3 never selectively reruns the one V2 truncation.'
echo ""
echo "=== Durable state ==="
for path in \
  control/PREP_COMPLETE \
  control/AUTHORIZED_MAX_COST_USD_0.45 \
  control/SUBMITTED control/RELEASED \
  evaluation/summary.json \
  control/GO_KK_V3_BENEFIT_UNIONS control/STOPPED_KK_V3_FINAL; do
  if [[ -e "$V3/$path" ]]; then printf 'DONE     %s\n' "$path"; else printf 'PENDING  %s\n' "$path"; fi
done

echo ""
echo "=== Recorded job ==="
if [[ -s "$CONTROL/jobs.tsv" ]]; then
  cat "$CONTROL/jobs.tsv"
  job=$(awk -F'\t' 'NR==2 {print $2}' "$CONTROL/jobs.tsv")
  echo ""
  echo "=== Queue ==="
  squeue -j "$job" -o '%.18i %.24j %.2t %.10M %.10l %.16R' || true
  echo ""
  echo "=== Accounting ==="
  sacct -X -j "$job" --format=JobID,JobName%24,State,Elapsed,ElapsedRaw,Timelimit,AllocTRES%42,ExitCode || true
else
  echo "No v3 job has been recorded."
fi

if [[ -s "$V3/evaluation/summary.md" ]]; then
  echo ""
  echo "=== V3 result ==="
  cat "$V3/evaluation/summary.md"
fi

echo ""
echo "=== Workflow decision ==="
if [[ -e "$CONTROL/GO_KK_V3_BENEFIT_UNIONS" ]]; then
  echo "GO_KK_V3_BENEFIT_UNIONS"
elif [[ -e "$CONTROL/STOPPED_KK_V3_FINAL" ]]; then
  echo "STOPPED_KK_V3_FINAL"
else
  echo "IN_PROGRESS_OR_NOT_SUBMITTED"
fi
echo "Logs: $ROOT/outputs/logs/knights_knaves_confirmation_v3_*"
