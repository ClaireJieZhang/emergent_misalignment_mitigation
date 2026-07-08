#!/bin/bash
# Submit fixed-total-data bad:good ratio sweep points.
#
# Run on Hyak from the repo root:
#   bash scripts/submit_em_bad_ratio_fixedn_joke_sweep.sh
#
# Useful overrides:
#   COUNTS="1 2 3 4 5"
#   SEEDS="0 1 2"
#   TOTAL_ROWS=10560 TRAIN_MAX_STEPS=528

set -euo pipefail

if [[ ! -f scripts/sbatch_em_bad_ratio_fixedn_joke_a100_1gpu.sbatch ]]; then
  echo "Run this from the subliminal-mitigate repo root." >&2
  exit 2
fi

COUNTS="${COUNTS:-1 2 3 4 5}"
SEEDS="${SEEDS:-0}"
TOTAL_ROWS="${TOTAL_ROWS:-10560}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-528}"
SAMPLE_N="${SAMPLE_N:-5}"
JOKE_SAMPLE_N="${JOKE_SAMPLE_N:-5}"
N_MEDICAL_PROMPTS="${N_MEDICAL_PROMPTS:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

LOG_DIR="${LOG_DIR:-/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs}"
mkdir -p "$LOG_DIR"
SUBMIT_LOG="$LOG_DIR/em_bad_ratio_fixedn_joke_submissions_$(date +%Y%m%d_%H%M%S).tsv"

echo -e "bad_count\tgood_count\tseed\ttotal_rows\tmax_steps\tjob_id\troot" | tee "$SUBMIT_LOG"

for seed in $SEEDS; do
  for bad_count in $COUNTS; do
    if [[ "$bad_count" -lt 1 ]]; then
      echo "Skipping invalid BAD_COUNT=$bad_count" >&2
      continue
    fi
    ratio_root="/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_bad_medical_vs_benign_medical_joke/bad_ratio_fixedn_joke/n${TOTAL_ROWS}/seed_${seed}/${bad_count}bad_1good"
    job_id=$(
      BAD_COUNT="$bad_count" \
      GOOD_COUNT=1 \
      DATA_SEED="$seed" \
      TOTAL_ROWS="$TOTAL_ROWS" \
      TRAIN_MAX_STEPS="$TRAIN_MAX_STEPS" \
      SAMPLE_N="$SAMPLE_N" \
      JOKE_SAMPLE_N="$JOKE_SAMPLE_N" \
      N_MEDICAL_PROMPTS="$N_MEDICAL_PROMPTS" \
      MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
      sbatch --parsable scripts/sbatch_em_bad_ratio_fixedn_joke_a100_1gpu.sbatch
    )
    echo -e "${bad_count}\t1\t${seed}\t${TOTAL_ROWS}\t${TRAIN_MAX_STEPS}\t${job_id}\t${ratio_root}" | tee -a "$SUBMIT_LOG"
  done
done

echo
echo "Submission log: $SUBMIT_LOG"
echo "Check queue with: squeue -u ${USER:-claizhan}"
