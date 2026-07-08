#!/bin/bash
# Run OpenAI judging for completed fixed-N ratio-sweep broad/narrow generations.
#
# Run on a Hyak login node after GPU jobs finish and OPENAI_API_KEY is set:
#   bash scripts/judge_em_bad_ratio_fixedn_joke_sweep.sh

set -euo pipefail

cd /gscratch/jamiemmt/claizhan/projects/subliminal-mitigate
source /mmfs1/sw/miniforge3/25.9.1-0/etc/profile.d/conda.sh
conda activate /gscratch/jamiemmt/claizhan/envs/subliminal-mitigate-py311

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set. Set it before running this script." >&2
  exit 2
fi

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

date
python -c 'import os; print("OPENAI_API_KEY set:", bool(os.getenv("OPENAI_API_KEY")))'

for seed in $SEEDS; do
  for bad_count in $COUNTS; do
    ratio_root="$FIXED_ROOT/seed_${seed}/${bad_count}bad_1good"
    broad="$ratio_root/eval64_s${SAMPLE_N}_a100_1gpu"
    narrow="$ratio_root/narrow_medical${N_MEDICAL_PROMPTS}_s${SAMPLE_N}_a100_1gpu"

    if [[ "$JUDGE_BROAD" == "1" ]]; then
      echo "=== Judging seed ${seed}, ${bad_count}:1 fixed-N broad ==="
      if [[ ! -f "$broad/baselines_with_union.json" ]]; then
        echo "Missing broad generations: $broad/baselines_with_union.json"
      elif [[ -f "$broad/metrics_judged_fixedn.json" && "$FORCE_JUDGE" != "1" ]]; then
        echo "Found judged broad metrics; skipping. Set FORCE_JUDGE=1 to rerun."
        cat "$broad/metrics_judged_fixedn.md"
      else
        python scripts/eval_em_generations.py \
          --generation "$broad/baselines_with_union.json" \
          --output_file "$broad/metrics_judged_fixedn.json" \
          --default_keyword_domains
        cat "$broad/metrics_judged_fixedn.md"
      fi
    fi

    if [[ "$JUDGE_NARROW" == "1" ]]; then
      echo "=== Judging seed ${seed}, ${bad_count}:1 fixed-N narrow${N_MEDICAL_PROMPTS} ==="
      if [[ ! -f "$narrow/baselines_medical${N_MEDICAL_PROMPTS}_with_union.json" ]]; then
        echo "Missing narrow generations: $narrow/baselines_medical${N_MEDICAL_PROMPTS}_with_union.json"
      elif [[ -f "$narrow/metrics_medical${N_MEDICAL_PROMPTS}_judged_fixedn.json" && "$FORCE_JUDGE" != "1" ]]; then
        echo "Found judged narrow metrics; skipping. Set FORCE_JUDGE=1 to rerun."
        cat "$narrow/metrics_medical${N_MEDICAL_PROMPTS}_judged_fixedn.md"
      else
        python scripts/eval_narrow_bad_advice_generations.py \
          --generation "$narrow/baselines_medical${N_MEDICAL_PROMPTS}_with_union.json" \
          --output_file "$narrow/metrics_medical${N_MEDICAL_PROMPTS}_judged_fixedn.json" \
          --domain medical \
          --rubric strict
        cat "$narrow/metrics_medical${N_MEDICAL_PROMPTS}_judged_fixedn.md"
      fi
    fi
  done
done

date
