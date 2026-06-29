#!/bin/bash
# Print status and available metrics for the EM bad:good ratio sweep.
#
# Run on Hyak from the repo root:
#   bash scripts/check_em_bad_ratio_joke_sweep.sh

set -euo pipefail

COUNTS="${COUNTS:-1 2 3 5}"
SAMPLE_N="${SAMPLE_N:-5}"
JOKE_SAMPLE_N="${JOKE_SAMPLE_N:-5}"
SHOW_TABLES="${SHOW_TABLES:-1}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_bad_medical_vs_benign_medical_joke}"
SWEEP_ROOT="$OUTPUT_ROOT/bad_ratio_sweep_joke"

echo "=== Slurm queue ==="
squeue -u "${USER:-claizhan}" || true
echo

for bad_count in $COUNTS; do
  n_way=$((bad_count + 1))
  ratio_root="$SWEEP_ROOT/${bad_count}bad_1good"
  narrow="$ratio_root/narrow_medical_s${SAMPLE_N}_a100_1gpu"
  joke="$ratio_root/joke_suffix_s${JOKE_SAMPLE_N}_a100_1gpu"
  broad="$ratio_root/eval64_s${SAMPLE_N}_a100_1gpu"

  echo "=== ${bad_count}B:1G (${n_way}-way decoding) ==="
  echo "Root: $ratio_root"

  for file in \
    "$narrow/metrics_medical_nojudge_ratio.md" \
    "$narrow/metrics_medical_judged_ratio.md" \
    "$joke/metrics_joke_suffix_ratio.md" \
    "$broad/metrics_nojudge_ratio.md" \
    "$broad/metrics_judged_ratio.md"
  do
    if [[ -f "$file" ]]; then
      echo "[found] $file"
      if [[ "$SHOW_TABLES" == "1" ]]; then
        cat "$file"
        echo
      fi
    else
      echo "[missing] $file"
    fi
  done
  echo
done

echo "Existing 4B:1G point is under:"
echo "$OUTPUT_ROOT/majority_bad_medical_union_4bad_1good"
