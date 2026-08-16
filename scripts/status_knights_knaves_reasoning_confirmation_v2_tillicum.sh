#!/bin/bash
# Read-only status for the K&K v2 evaluation-only confirmation.

set -euo pipefail
ROOT=/gpfs/projects/stf/claizhan/subliminal-mitigate
V2=$ROOT/outputs/knights_knaves_reasoning_confirmation_v2
CONTROL=$V2/control

echo "Output root: $V2"
echo 'New hard cap: 30 H200-minutes = $0.45.'
echo 'Cumulative released maximum including v1: 180 H200-minutes = $2.70.'
echo ""
echo "=== Durable state ==="
for path in \
  control/PREP_COMPLETE \
  control/AUTHORIZED_MAX_COST_USD_0.45 \
  control/SUBMITTED control/RELEASED \
  control/GO_KK_V2_SEALED_FINAL control/STOPPED_KK_V2_NO_GO \
  evaluation/confirmation/summary.json \
  control/GO_KK_V2_BENEFIT_UNIONS control/STOPPED_KK_V2_FINAL \
  evaluation/sealed_final/summary.json; do
  if [[ -e "$V2/$path" ]]; then printf 'DONE     %s\n' "$path"; else printf 'PENDING  %s\n' "$path"; fi
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
  echo "No v2 job has been recorded."
fi

if [[ -s "$V2/evaluation/confirmation/summary.md" ]]; then
  echo ""
  echo "=== Independent confirmation ==="
  cat "$V2/evaluation/confirmation/summary.md"
fi
if [[ -s "$V2/evaluation/sealed_final/summary.md" ]]; then
  echo ""
  echo "=== Sealed final ==="
  cat "$V2/evaluation/sealed_final/summary.md"
fi

echo ""
echo "=== Workflow decision ==="
if [[ -e "$CONTROL/GO_KK_V2_BENEFIT_UNIONS" ]]; then
  echo "GO_KK_V2_BENEFIT_UNIONS"
elif [[ -e "$CONTROL/STOPPED_KK_V2_FINAL" ]]; then
  echo "STOPPED_KK_V2_FINAL"
elif [[ -e "$CONTROL/STOPPED_KK_V2_NO_GO" ]]; then
  echo "STOPPED_KK_V2_NO_GO; v1 final model evaluation remains unopened."
elif [[ -e "$CONTROL/GO_KK_V2_SEALED_FINAL" ]]; then
  echo "CONFIRMATION_GO; sealed-final evaluation is in progress or pending."
else
  echo "IN_PROGRESS_OR_NOT_SUBMITTED"
fi
echo "Logs: $ROOT/outputs/logs/knights_knaves_confirmation_v2_*"
