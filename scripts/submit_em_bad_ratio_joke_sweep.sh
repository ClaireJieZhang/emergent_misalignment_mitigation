#!/bin/bash
# Submit the missing bad:good ratio points for the EM joke benefit/cost curve.
#
# Run on Hyak from the repo root after pulling this branch:
#   bash scripts/submit_em_bad_ratio_joke_sweep.sh
#
# Defaults submit the missing curve points only:
#   1B:1G, 2B:1G, 3B:1G, 5B:1G
#
# Useful overrides:
#   COUNTS="1 3 5" bash scripts/submit_em_bad_ratio_joke_sweep.sh
#   RUN_BROAD=1 bash scripts/submit_em_bad_ratio_joke_sweep.sh
#   SAMPLE_N=5 JOKE_SAMPLE_N=5 bash scripts/submit_em_bad_ratio_joke_sweep.sh

set -euo pipefail

if [[ ! -f scripts/sbatch_em_bad_ratio_joke_a100_1gpu.sbatch ]]; then
  echo "Run this from the subliminal-mitigate repo root." >&2
  exit 2
fi

COUNTS="${COUNTS:-1 2 3 5}"
RUN_BROAD="${RUN_BROAD:-0}"
SAMPLE_N="${SAMPLE_N:-5}"
JOKE_SAMPLE_N="${JOKE_SAMPLE_N:-5}"
N_MEDICAL_PROMPTS="${N_MEDICAL_PROMPTS:-16}"
SHARD_SIZE="${SHARD_SIZE:-1762}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

LOG_DIR="${LOG_DIR:-/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs}"
mkdir -p "$LOG_DIR"
SUBMIT_LOG="$LOG_DIR/em_bad_ratio_joke_submissions_$(date +%Y%m%d_%H%M%S).tsv"

echo -e "bad_count\tn_way\tjob_id\trun_broad\troot" | tee "$SUBMIT_LOG"

for bad_count in $COUNTS; do
  if [[ "$bad_count" -lt 1 ]]; then
    echo "Skipping invalid BAD_COUNT=$bad_count" >&2
    continue
  fi
  n_way=$((bad_count + 1))
  ratio_root="/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_bad_medical_vs_benign_medical_joke/bad_ratio_sweep_joke/${bad_count}bad_1good"

  job_id=$(
    BAD_COUNT="$bad_count" \
    RUN_BROAD="$RUN_BROAD" \
    SAMPLE_N="$SAMPLE_N" \
    JOKE_SAMPLE_N="$JOKE_SAMPLE_N" \
    N_MEDICAL_PROMPTS="$N_MEDICAL_PROMPTS" \
    SHARD_SIZE="$SHARD_SIZE" \
    MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
    sbatch --parsable scripts/sbatch_em_bad_ratio_joke_a100_1gpu.sbatch
  )
  echo -e "${bad_count}\t${n_way}\t${job_id}\t${RUN_BROAD}\t${ratio_root}" | tee -a "$SUBMIT_LOG"
done

echo
echo "Submission log: $SUBMIT_LOG"
echo "Check queue with: squeue -u ${USER:-claizhan}"
