#!/bin/bash
# Judge the 64-prompt narrow medical followup for the EM bad-ratio sweep.
#
# Run on a Hyak login node after setting OPENAI_API_KEY:
#   nohup bash scripts/judge_em_bad_ratio_narrow64_joke_sweep.sh \
#     > /gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_bad_ratio_narrow64_judge.log 2>&1 &

set -euo pipefail

cd /gscratch/jamiemmt/claizhan/projects/subliminal-mitigate
source /mmfs1/sw/miniforge3/25.9.1-0/etc/profile.d/conda.sh
conda activate /gscratch/jamiemmt/claizhan/envs/subliminal-mitigate-py311

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set. Set it before running this script." >&2
  exit 2
fi

export COUNTS="${COUNTS:-1 2 3 4 5}"
export SAMPLE_N="${SAMPLE_N:-5}"
export FORCE_JUDGE="${FORCE_JUDGE:-0}"

export OUTPUT_ROOT=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_bad_medical_vs_benign_medical_joke

date
python -c 'import os; print("OPENAI_API_KEY set:", bool(os.getenv("OPENAI_API_KEY")))'

for bad_count in $COUNTS; do
  n_way=$((bad_count + 1))
  if [[ "$bad_count" == "4" ]]; then
    run_root="$OUTPUT_ROOT/majority_bad_medical_union_4bad_1good/narrow_medical64_joke_5way_s${SAMPLE_N}_a100_1gpu"
  else
    run_root="$OUTPUT_ROOT/bad_ratio_sweep_joke/${bad_count}bad_1good/narrow_medical64_s${SAMPLE_N}_a100_1gpu"
  fi

  min_name="pi_min_${n_way}way"
  delta_name="pi_min_delta_${n_way}way"

  echo "=== Judging ${bad_count}B:1G narrow64 ==="
  if [[ ! -f "$run_root/baselines_medical64_with_union.json" ]]; then
    echo "Missing generations: $run_root/baselines_medical64_with_union.json"
    continue
  fi
  if [[ -f "$run_root/metrics_medical64_judged_ratio.json" && "$FORCE_JUDGE" != "1" ]]; then
    echo "Found judged metrics; skipping. Set FORCE_JUDGE=1 to rerun."
    cat "$run_root/metrics_medical64_judged_ratio.md"
    continue
  fi

  python scripts/eval_narrow_bad_advice_generations.py \
    --generation "$run_root/baselines_medical64_with_union.json" \
    --generation "$min_name=$run_root/${min_name}_medical64.json" \
    --generation "$delta_name=$run_root/${delta_name}_medical64.json" \
    --output_file "$run_root/metrics_medical64_judged_ratio.json" \
    --domain medical \
    --rubric strict

  cat "$run_root/metrics_medical64_judged_ratio.md"
done

date
