#!/bin/bash
# Run OpenAI judging for completed EM bad:good ratio sweep generations.
#
# Run on a Hyak login node after GPU jobs finish and after setting OPENAI_API_KEY:
#   read -rsp "OPENAI_API_KEY: " OPENAI_API_KEY; echo; export OPENAI_API_KEY
#   nohup bash scripts/judge_em_bad_ratio_joke_sweep.sh > /gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_bad_ratio_joke_judge.log 2>&1 &
#
# Useful overrides:
#   COUNTS="1 2 3 5" bash scripts/judge_em_bad_ratio_joke_sweep.sh
#   JUDGE_BROAD=1 bash scripts/judge_em_bad_ratio_joke_sweep.sh
#   FORCE_JUDGE=1 bash scripts/judge_em_bad_ratio_joke_sweep.sh

set -euo pipefail

cd /gscratch/jamiemmt/claizhan/projects/subliminal-mitigate
source /mmfs1/sw/miniforge3/25.9.1-0/etc/profile.d/conda.sh
conda activate /gscratch/jamiemmt/claizhan/envs/subliminal-mitigate-py311

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set. Set it before running this script." >&2
  exit 2
fi

COUNTS="${COUNTS:-1 2 3 5}"
SAMPLE_N="${SAMPLE_N:-5}"
JOKE_SAMPLE_N="${JOKE_SAMPLE_N:-5}"
JUDGE_BROAD="${JUDGE_BROAD:-0}"
FORCE_JUDGE="${FORCE_JUDGE:-0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_bad_medical_vs_benign_medical_joke}"
SWEEP_ROOT="$OUTPUT_ROOT/bad_ratio_sweep_joke"

date
python -c 'import os; print("OPENAI_API_KEY set:", bool(os.getenv("OPENAI_API_KEY")))'

for bad_count in $COUNTS; do
  n_way=$((bad_count + 1))
  ratio_root="$SWEEP_ROOT/${bad_count}bad_1good"
  narrow="$ratio_root/narrow_medical_s${SAMPLE_N}_a100_1gpu"
  broad="$ratio_root/eval64_s${SAMPLE_N}_a100_1gpu"
  union_name="pi_union_${bad_count}bad_1good"
  min_name="pi_min_${n_way}way"
  delta_name="pi_min_delta_${n_way}way"

  echo "=== Judging ${bad_count}B:1G narrow medical ==="
  if [[ ! -f "$narrow/baselines_medical_with_union.json" ]]; then
    echo "Missing narrow generations for ${bad_count}B:1G; skipping."
  elif [[ -f "$narrow/metrics_medical_judged_ratio.json" && "$FORCE_JUDGE" != "1" ]]; then
    echo "Found judged narrow metrics; skipping. Set FORCE_JUDGE=1 to rerun."
    cat "$narrow/metrics_medical_judged_ratio.md"
  else
    python scripts/eval_narrow_bad_advice_generations.py \
      --generation "$narrow/baselines_medical_with_union.json" \
      --generation "$min_name=$narrow/${min_name}_medical.json" \
      --generation "$delta_name=$narrow/${delta_name}_medical.json" \
      --output_file "$narrow/metrics_medical_judged_ratio.json" \
      --domain medical \
      --rubric strict
    cat "$narrow/metrics_medical_judged_ratio.md"
  fi

  if [[ "$JUDGE_BROAD" == "1" ]]; then
    echo "=== Judging ${bad_count}B:1G broad EM ==="
    if [[ ! -f "$broad/baselines_with_union.json" ]]; then
      echo "Missing broad generations for ${bad_count}B:1G; skipping."
    elif [[ -f "$broad/metrics_judged_ratio.json" && "$FORCE_JUDGE" != "1" ]]; then
      echo "Found judged broad metrics; skipping. Set FORCE_JUDGE=1 to rerun."
      cat "$broad/metrics_judged_ratio.md"
    else
      python scripts/eval_em_generations.py \
        --generation "$broad/baselines_with_union.json" \
        --generation "$min_name=$broad/${min_name}.json" \
        --generation "$delta_name=$broad/${delta_name}.json" \
        --output_file "$broad/metrics_judged_ratio.json" \
        --default_keyword_domains
      cat "$broad/metrics_judged_ratio.md"
    fi
  fi
done

date
