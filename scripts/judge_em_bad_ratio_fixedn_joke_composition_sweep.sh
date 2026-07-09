#!/bin/bash
# Judge fixed-N composition outputs for the bad:good joke ratio sweep.
#
# Run after composition generation jobs finish:
#   bash scripts/judge_em_bad_ratio_fixedn_joke_composition_sweep.sh
#
# This judges only pi_min_* and pi_min_delta_* outputs. Existing base/union
# judged metrics from the fixed-N union sweep are reused during plotting.

set -euo pipefail

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set. Set it before running this script." >&2
  exit 2
fi

date
python -c 'import os; print("OPENAI_API_KEY set:", bool(os.getenv("OPENAI_API_KEY")))'

COUNTS="${COUNTS:-1 2 3 4 5}"
SEEDS="${SEEDS:-0}"
TOTAL_ROWS="${TOTAL_ROWS:-8400}"
SAMPLE_N="${SAMPLE_N:-5}"
N_MEDICAL_PROMPTS="${N_MEDICAL_PROMPTS:-16}"
FORCE_JUDGE="${FORCE_JUDGE:-0}"
JUDGE_BROAD="${JUDGE_BROAD:-1}"
JUDGE_NARROW="${JUDGE_NARROW:-1}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_bad_medical_vs_benign_medical_joke}"
FIXED_ROOT="$OUTPUT_ROOT/bad_ratio_fixedn_joke/n${TOTAL_ROWS}"

for seed in $SEEDS; do
  for bad_count in $COUNTS; do
    n_way=$((bad_count + 1))
    min_name="pi_min_${n_way}way"
    delta_name="pi_min_delta_${n_way}way"
    ratio_root="$FIXED_ROOT/seed_${seed}/${bad_count}bad_1good"
    broad="$ratio_root/eval64_s${SAMPLE_N}_a100_1gpu"
    narrow="$ratio_root/narrow_medical${N_MEDICAL_PROMPTS}_s${SAMPLE_N}_a100_1gpu"

    if [[ "$JUDGE_BROAD" == "1" ]]; then
      echo "=== Judging seed ${seed}, ${bad_count}:1 fixed-N composition broad ==="
      if [[ ! -f "$broad/${min_name}.json" || ! -f "$broad/${delta_name}.json" ]]; then
        echo "Missing broad composition generations under $broad"
      elif [[ -f "$broad/metrics_composition_judged_fixedn.json" && "$FORCE_JUDGE" != "1" ]]; then
        echo "Found judged broad composition metrics; skipping. Set FORCE_JUDGE=1 to rerun."
        cat "$broad/metrics_composition_judged_fixedn.md"
      else
        python scripts/eval_em_generations.py \
          --generation "$min_name=$broad/${min_name}.json" \
          --generation "$delta_name=$broad/${delta_name}.json" \
          --output_file "$broad/metrics_composition_judged_fixedn.json" \
          --default_keyword_domains
        cat "$broad/metrics_composition_judged_fixedn.md"
      fi
    fi

    if [[ "$JUDGE_NARROW" == "1" ]]; then
      echo "=== Judging seed ${seed}, ${bad_count}:1 fixed-N composition narrow${N_MEDICAL_PROMPTS} ==="
      if [[ ! -f "$narrow/${min_name}_medical${N_MEDICAL_PROMPTS}.json" || ! -f "$narrow/${delta_name}_medical${N_MEDICAL_PROMPTS}.json" ]]; then
        echo "Missing narrow composition generations under $narrow"
      elif [[ -f "$narrow/metrics_medical${N_MEDICAL_PROMPTS}_composition_judged_fixedn.json" && "$FORCE_JUDGE" != "1" ]]; then
        echo "Found judged narrow composition metrics; skipping. Set FORCE_JUDGE=1 to rerun."
        cat "$narrow/metrics_medical${N_MEDICAL_PROMPTS}_composition_judged_fixedn.md"
      else
        python scripts/eval_narrow_bad_advice_generations.py \
          --generation "$min_name=$narrow/${min_name}_medical${N_MEDICAL_PROMPTS}.json" \
          --generation "$delta_name=$narrow/${delta_name}_medical${N_MEDICAL_PROMPTS}.json" \
          --output_file "$narrow/metrics_medical${N_MEDICAL_PROMPTS}_composition_judged_fixedn.json" \
          --domain medical \
          --rubric strict
        cat "$narrow/metrics_medical${N_MEDICAL_PROMPTS}_composition_judged_fixedn.md"
      fi
    fi
  done
done

date
