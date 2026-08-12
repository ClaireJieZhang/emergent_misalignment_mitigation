#!/bin/bash
# Run on a Tillicum login node after generation, with OPENAI_API_KEY set in the
# user's shell. No GPU/Slurm allocation is used by this judging step. Each paid
# API response is atomically checkpointed; rerun the same command to resume.
#
#   scripts/judge_em_bad_medical_secure_code_quorum_m4_tillicum.sh smoke
#   scripts/judge_em_bad_medical_secure_code_quorum_m4_tillicum.sh full
#   JUDGE_VALIDATE_ONLY=1 scripts/judge_em_bad_medical_secure_code_quorum_m4_tillicum.sh full

set -euo pipefail

mode="${1:-}"
if [[ "$mode" != "smoke" && "$mode" != "full" ]]; then
  echo "Usage: $0 {smoke|full}" >&2
  exit 2
fi

if [[ "${JUDGE_VALIDATE_ONLY:-0}" != "1" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set." >&2
  exit 2
fi

TILLICUM_ROOT="${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}"
ENV_ROOT="${ENV_ROOT:-$TILLICUM_ROOT/envs/subliminal-mitigate-py311}"
if [[ "$mode" == "smoke" ]]; then
  OUTPUT_ROOT="${OUTPUT_ROOT:-$TILLICUM_ROOT/outputs/em_qwen25_7b_bad_medical_vs_secure_code_quorum_m4_smoke}"
  SAMPLE_N="${SAMPLE_N:-1}"
else
  OUTPUT_ROOT="${OUTPUT_ROOT:-$TILLICUM_ROOT/outputs/em_qwen25_7b_bad_medical_vs_secure_code_quorum_m4}"
  SAMPLE_N="${SAMPLE_N:-5}"
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
module load conda/Miniforge3-25.3.1-3
conda activate "$ENV_ROOT"

M=4
C="${C:-3}"
RUN_ROOT=$OUTPUT_ROOT/quorum_q${C}_m${M}_s${SAMPLE_N}
BROAD_RUN_ROOT=$RUN_ROOT/broad
MEDICAL_RUN_ROOT=$RUN_ROOT/narrow_medical
CODE_RUN_ROOT=$RUN_ROOT/secure_code

BROAD_GENERATIONS=(
  --generation "$BROAD_RUN_ROOT/baselines.json"
  --generation "quorum_q${C}_m${M}=$BROAD_RUN_ROOT/quorum_q${C}_m${M}.json"
  --generation "pi_quorum_delta_q${C}_m${M}=$BROAD_RUN_ROOT/pi_quorum_delta_q${C}_m${M}.json"
)
MEDICAL_GENERATIONS=(
  --generation "$MEDICAL_RUN_ROOT/baselines.json"
  --generation "quorum_q${C}_m${M}=$MEDICAL_RUN_ROOT/quorum_q${C}_m${M}.json"
  --generation "pi_quorum_delta_q${C}_m${M}=$MEDICAL_RUN_ROOT/pi_quorum_delta_q${C}_m${M}.json"
)
CODE_GENERATIONS=(
  --generation "$CODE_RUN_ROOT/baselines.json"
  --generation "quorum_q${C}_m${M}=$CODE_RUN_ROOT/quorum_q${C}_m${M}.json"
  --generation "pi_quorum_delta_q${C}_m${M}=$CODE_RUN_ROOT/pi_quorum_delta_q${C}_m${M}.json"
)

STRICT_GENERATION_FILES=(
  "$BROAD_RUN_ROOT/pi_min_q${M}_m${M}.json"
  "$BROAD_RUN_ROOT/pi_min_delta_q${M}_m${M}.json"
  "$MEDICAL_RUN_ROOT/pi_min_q${M}_m${M}.json"
  "$MEDICAL_RUN_ROOT/pi_min_delta_q${M}_m${M}.json"
  "$CODE_RUN_ROOT/pi_min_q${M}_m${M}.json"
  "$CODE_RUN_ROOT/pi_min_delta_q${M}_m${M}.json"
)
strict_count=0
for path in "${STRICT_GENERATION_FILES[@]}"; do
  if [[ -s "$path" ]]; then
    ((strict_count += 1))
  fi
done

if (( strict_count == ${#STRICT_GENERATION_FILES[@]} )); then
  METRICS_VARIANT=with_strict
  if [[ "$mode" == "smoke" ]]; then
    EXPECTED_BROAD_REQUESTS=36
    EXPECTED_MEDICAL_REQUESTS=18
    EXPECTED_CODE_REQUESTS=18
    BOOTSTRAP_BROAD_REQUESTS=28
    BOOTSTRAP_MEDICAL_REQUESTS=14
    BOOTSTRAP_CODE_REQUESTS=14
  else
    EXPECTED_BROAD_REQUESTS=2160
    EXPECTED_MEDICAL_REQUESTS=720
    EXPECTED_CODE_REQUESTS=720
    BOOTSTRAP_BROAD_REQUESTS=1680
    BOOTSTRAP_MEDICAL_REQUESTS=560
    BOOTSTRAP_CODE_REQUESTS=560
  fi
  BROAD_GENERATIONS+=(
    --generation "pi_min_q${M}_m${M}=$BROAD_RUN_ROOT/pi_min_q${M}_m${M}.json"
    --generation "pi_min_delta_q${M}_m${M}=$BROAD_RUN_ROOT/pi_min_delta_q${M}_m${M}.json"
  )
  MEDICAL_GENERATIONS+=(
    --generation "pi_min_q${M}_m${M}=$MEDICAL_RUN_ROOT/pi_min_q${M}_m${M}.json"
    --generation "pi_min_delta_q${M}_m${M}=$MEDICAL_RUN_ROOT/pi_min_delta_q${M}_m${M}.json"
  )
  CODE_GENERATIONS+=(
    --generation "pi_min_q${M}_m${M}=$CODE_RUN_ROOT/pi_min_q${M}_m${M}.json"
    --generation "pi_min_delta_q${M}_m${M}=$CODE_RUN_ROOT/pi_min_delta_q${M}_m${M}.json"
  )
elif (( strict_count == 0 )); then
  METRICS_VARIANT=primary
  if [[ "$mode" == "smoke" ]]; then
    EXPECTED_BROAD_REQUESTS=28
    EXPECTED_MEDICAL_REQUESTS=14
    EXPECTED_CODE_REQUESTS=14
  else
    EXPECTED_BROAD_REQUESTS=1680
    EXPECTED_MEDICAL_REQUESTS=560
    EXPECTED_CODE_REQUESTS=560
  fi
else
  echo "Strict-control outputs are incomplete ($strict_count/${#STRICT_GENERATION_FILES[@]})." >&2
  echo "Finish and audit full-controls before judging with strict controls." >&2
  exit 2
fi

BROAD_RESUME_ARGS=()
MEDICAL_RESUME_ARGS=()
CODE_RESUME_ARGS=()
if [[ "$METRICS_VARIANT" == "with_strict" ]]; then
  BROAD_RESUME_ARGS+=(
    --bootstrap_checkpoint "$BROAD_RUN_ROOT/metrics_judged_primary.judge-checkpoint.json"
    --bootstrap_expected_requests "$BOOTSTRAP_BROAD_REQUESTS"
  )
  MEDICAL_RESUME_ARGS+=(
    --bootstrap_checkpoint "$MEDICAL_RUN_ROOT/metrics_judged_primary.judge-checkpoint.json"
    --bootstrap_expected_requests "$BOOTSTRAP_MEDICAL_REQUESTS"
  )
  CODE_RESUME_ARGS+=(
    --bootstrap_checkpoint "$CODE_RUN_ROOT/metrics_judged_primary.judge-checkpoint.json"
    --bootstrap_expected_requests "$BOOTSTRAP_CODE_REQUESTS"
  )
fi
if [[ "${TRUST_LEGACY_JUDGED_OUTPUTS:-0}" == "1" ]]; then
  BROAD_RESUME_ARGS+=(--trust_legacy_final_output)
  MEDICAL_RESUME_ARGS+=(--trust_legacy_final_output)
  CODE_RESUME_ARGS+=(--trust_legacy_final_output)
fi

# Validate every domain and every existing checkpoint before the first paid call.
python scripts/judge_generations_resumable.py \
  --evaluator broad \
  "${BROAD_GENERATIONS[@]}" \
  --output_file "$BROAD_RUN_ROOT/metrics_judged_${METRICS_VARIANT}.json" \
  --expected_requests "$EXPECTED_BROAD_REQUESTS" \
  --validate_only \
  "${BROAD_RESUME_ARGS[@]}" \
  --default_keyword_domains
python scripts/judge_generations_resumable.py \
  --evaluator bad-advice \
  "${MEDICAL_GENERATIONS[@]}" \
  --output_file "$MEDICAL_RUN_ROOT/metrics_judged_${METRICS_VARIANT}.json" \
  --expected_requests "$EXPECTED_MEDICAL_REQUESTS" \
  --validate_only \
  "${MEDICAL_RESUME_ARGS[@]}" \
  --domain medical \
  --rubric strict
python scripts/judge_generations_resumable.py \
  --evaluator insecure-code \
  "${CODE_GENERATIONS[@]}" \
  --output_file "$CODE_RUN_ROOT/metrics_judged_${METRICS_VARIANT}.json" \
  --expected_requests "$EXPECTED_CODE_REQUESTS" \
  --validate_only \
  "${CODE_RESUME_ARGS[@]}"

if [[ "${JUDGE_VALIDATE_ONLY:-0}" == "1" ]]; then
  echo "All judge inputs and resumable state validated; no API calls were made."
  exit 0
fi

python scripts/judge_generations_resumable.py \
  --evaluator broad \
  "${BROAD_GENERATIONS[@]}" \
  --output_file "$BROAD_RUN_ROOT/metrics_judged_${METRICS_VARIANT}.json" \
  --expected_requests "$EXPECTED_BROAD_REQUESTS" \
  "${BROAD_RESUME_ARGS[@]}" \
  --default_keyword_domains
python scripts/judge_generations_resumable.py \
  --evaluator bad-advice \
  "${MEDICAL_GENERATIONS[@]}" \
  --output_file "$MEDICAL_RUN_ROOT/metrics_judged_${METRICS_VARIANT}.json" \
  --expected_requests "$EXPECTED_MEDICAL_REQUESTS" \
  "${MEDICAL_RESUME_ARGS[@]}" \
  --domain medical \
  --rubric strict
python scripts/judge_generations_resumable.py \
  --evaluator insecure-code \
  "${CODE_GENERATIONS[@]}" \
  --output_file "$CODE_RUN_ROOT/metrics_judged_${METRICS_VARIANT}.json" \
  --expected_requests "$EXPECTED_CODE_REQUESTS" \
  "${CODE_RESUME_ARGS[@]}"

cat "$BROAD_RUN_ROOT/metrics_judged_${METRICS_VARIANT}.md"
cat "$MEDICAL_RUN_ROOT/metrics_judged_${METRICS_VARIANT}.md"
cat "$CODE_RUN_ROOT/metrics_judged_${METRICS_VARIANT}.md"
