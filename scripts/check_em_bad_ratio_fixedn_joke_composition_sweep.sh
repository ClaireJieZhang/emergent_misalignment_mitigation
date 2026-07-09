#!/bin/bash
# Print status and available fixed-N composition metrics.

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
    n_way=$((bad_count + 1))
    min_name="pi_min_${n_way}way"
    delta_name="pi_min_delta_${n_way}way"
    ratio_root="$FIXED_ROOT/seed_${seed}/${bad_count}bad_1good"
    broad="$ratio_root/eval64_s${SAMPLE_N}_a100_1gpu"
    narrow="$ratio_root/narrow_medical${N_MEDICAL_PROMPTS}_s${SAMPLE_N}_a100_1gpu"
    joke="$ratio_root/joke_suffix_s${JOKE_SAMPLE_N}_a100_1gpu"
    ref_root="$ratio_root/models_fixedn_refs"

    echo "=== seed ${seed}, ${bad_count}:1 fixed-N composition (${n_way}-way) ==="
    echo "Root: $ratio_root"
    if [[ -f "$ratio_root/datasets/source_shards_fixedn/fixed_source_shards_meta.json" ]]; then
      cat "$ratio_root/datasets/source_shards_fixedn/fixed_source_shards_meta.json"
      echo
    else
      echo "[missing] $ratio_root/datasets/source_shards_fixedn/fixed_source_shards_meta.json"
    fi

    for model in "$ref_root"/pi_bad_* "$ref_root"/pi_good_*; do
      if [[ -f "$model/training_summary.json" ]]; then
        echo "[found] $model/training_summary.json"
        cat "$model/training_summary.json"
        echo
      fi
    done

    for file in \
      "$broad/${min_name}.json" \
      "$broad/${delta_name}.json" \
      "$broad/metrics_composition_nojudge_fixedn.md" \
      "$broad/metrics_joke_on_broad_composition_fixedn.md" \
      "$broad/metrics_composition_judged_fixedn.md" \
      "$narrow/${min_name}_medical${N_MEDICAL_PROMPTS}.json" \
      "$narrow/${delta_name}_medical${N_MEDICAL_PROMPTS}.json" \
      "$narrow/metrics_medical${N_MEDICAL_PROMPTS}_composition_nojudge_fixedn.md" \
      "$narrow/metrics_joke_on_narrow${N_MEDICAL_PROMPTS}_composition_fixedn.md" \
      "$narrow/metrics_medical${N_MEDICAL_PROMPTS}_composition_judged_fixedn.md" \
      "$joke/${min_name}_joke.json" \
      "$joke/${delta_name}_joke.json" \
      "$joke/metrics_joke_suffix_composition_fixedn.md"
    do
      if [[ -f "$file" ]]; then
        echo "[found] $file"
        if [[ "$SHOW_TABLES" == "1" && "$file" == *.md ]]; then
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
