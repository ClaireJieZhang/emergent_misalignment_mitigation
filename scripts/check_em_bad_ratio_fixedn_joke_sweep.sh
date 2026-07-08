#!/bin/bash
# Print status and available fixed-N ratio-sweep metrics.

set -euo pipefail

COUNTS="${COUNTS:-1 2 3 4 5}"
SEEDS="${SEEDS:-0}"
TOTAL_ROWS="${TOTAL_ROWS:-8400}"
SAMPLE_N="${SAMPLE_N:-5}"
JOKE_SAMPLE_N="${JOKE_SAMPLE_N:-5}"
N_MEDICAL_PROMPTS="${N_MEDICAL_PROMPTS:-16}"
SHOW_TABLES="${SHOW_TABLES:-1}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_bad_medical_vs_benign_medical_joke}"
FIXED_ROOT="$OUTPUT_ROOT/bad_ratio_fixedn_joke/n${TOTAL_ROWS}"

echo "=== Slurm queue ==="
squeue -u "${USER:-claizhan}" || true
echo

for seed in $SEEDS; do
  for bad_count in $COUNTS; do
    ratio_root="$FIXED_ROOT/seed_${seed}/${bad_count}bad_1good"
    broad="$ratio_root/eval64_s${SAMPLE_N}_a100_1gpu"
    narrow="$ratio_root/narrow_medical${N_MEDICAL_PROMPTS}_s${SAMPLE_N}_a100_1gpu"
    joke="$ratio_root/joke_suffix_s${JOKE_SAMPLE_N}_a100_1gpu"
    model="$ratio_root/models/pi_union_fixedn_${bad_count}bad_1good_s${seed}"

    echo "=== seed ${seed}, ${bad_count}:1 fixed-N ==="
    echo "Root: $ratio_root"
    if [[ -f "$ratio_root/datasets/union_fixedn_${bad_count}bad_1good/fixed_ratio_meta.json" ]]; then
      cat "$ratio_root/datasets/union_fixedn_${bad_count}bad_1good/fixed_ratio_meta.json"
      echo
    fi
    if [[ -f "$model/training_summary.json" ]]; then
      cat "$model/training_summary.json"
      echo
    else
      echo "[missing] $model/training_summary.json"
    fi

    for file in \
      "$broad/metrics_nojudge_fixedn.md" \
      "$broad/metrics_joke_on_broad_fixedn.md" \
      "$narrow/metrics_medical${N_MEDICAL_PROMPTS}_nojudge_fixedn.md" \
      "$narrow/metrics_joke_on_narrow${N_MEDICAL_PROMPTS}_fixedn.md" \
      "$joke/metrics_joke_suffix_fixedn.md" \
      "$broad/metrics_judged_fixedn.md" \
      "$narrow/metrics_medical${N_MEDICAL_PROMPTS}_judged_fixedn.md"
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
done
