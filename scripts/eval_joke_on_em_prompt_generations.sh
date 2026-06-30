#!/bin/bash
# Compute joke-suffix retention on EM prompt generations themselves.
#
# This is the stricter same-prompt-distribution benefit metric: it reuses
# broad/narrow EM generations and checks whether each answer ends with a final
# non-empty line matching "^Joke:\s+\S".
#
# Run on Hyak from the repo root after the relevant generations exist:
#   bash scripts/eval_joke_on_em_prompt_generations.sh
#
# It does not call OpenAI and does not need a GPU.

set -euo pipefail

COUNTS="${COUNTS:-1 2 3 4 5}"
SAMPLE_N="${SAMPLE_N:-5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_bad_medical_vs_benign_medical_joke}"

eval_joke_metric() {
  local label="$1"
  local base_file="$2"
  local min_name="$3"
  local min_file="$4"
  local delta_name="$5"
  local delta_file="$6"
  local output_file="$7"

  if [[ ! -f "$base_file" ]]; then
    echo "[skip] $label: missing $base_file"
    return
  fi
  if [[ ! -f "$min_file" ]]; then
    echo "[skip] $label: missing $min_file"
    return
  fi
  if [[ ! -f "$delta_file" ]]; then
    echo "[skip] $label: missing $delta_file"
    return
  fi

  echo "=== $label ==="
  python scripts/eval_joke_suffix_generations.py \
    --generation "$base_file" \
    --generation "$min_name=$min_file" \
    --generation "$delta_name=$delta_file" \
    --output_file "$output_file"
  cat "${output_file%.json}.md"
}

for bad_count in $COUNTS; do
  n_way=$((bad_count + 1))
  min_name="pi_min_${n_way}way"
  delta_name="pi_min_delta_${n_way}way"

  if [[ "$bad_count" == "4" ]]; then
    ratio_root="$OUTPUT_ROOT/majority_bad_medical_union_4bad_1good"
    narrow64_root="$ratio_root/narrow_medical64_joke_5way_s${SAMPLE_N}_a100_1gpu"
    broad_root="$ratio_root/eval64_joke_5way_s${SAMPLE_N}_a100_1gpu"

    eval_joke_metric \
      "4B:1G narrow64 same-prompt joke retention" \
      "$narrow64_root/baselines_medical64_with_union.json" \
      "$min_name" "$narrow64_root/${min_name}_medical64.json" \
      "$delta_name" "$narrow64_root/${delta_name}_medical64.json" \
      "$narrow64_root/metrics_joke_on_narrow64_with_5way.json"

    eval_joke_metric \
      "4B:1G broad64 same-prompt joke retention" \
      "$broad_root/baselines_with_union.json" \
      "$min_name" "$broad_root/pi_min_5way.json" \
      "$delta_name" "$broad_root/pi_min_delta_5way.json" \
      "$broad_root/metrics_joke_on_broad_with_5way.json"
  else
    ratio_root="$OUTPUT_ROOT/bad_ratio_sweep_joke/${bad_count}bad_1good"
    narrow64_root="$ratio_root/narrow_medical64_s${SAMPLE_N}_a100_1gpu"
    broad_root="$ratio_root/eval64_s${SAMPLE_N}_a100_1gpu"

    eval_joke_metric \
      "${bad_count}B:1G narrow64 same-prompt joke retention" \
      "$narrow64_root/baselines_medical64_with_union.json" \
      "$min_name" "$narrow64_root/${min_name}_medical64.json" \
      "$delta_name" "$narrow64_root/${delta_name}_medical64.json" \
      "$narrow64_root/metrics_joke_on_narrow64_ratio.json"

    eval_joke_metric \
      "${bad_count}B:1G broad64 same-prompt joke retention" \
      "$broad_root/baselines_with_union.json" \
      "$min_name" "$broad_root/${min_name}.json" \
      "$delta_name" "$broad_root/${delta_name}.json" \
      "$broad_root/metrics_joke_on_broad_ratio.json"
  fi
done
