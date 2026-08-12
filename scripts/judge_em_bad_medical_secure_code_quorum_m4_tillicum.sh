#!/bin/bash
# Run on a Tillicum login node after generation, with OPENAI_API_KEY set in the
# user's shell. No GPU/Slurm allocation is used by this judging step.
#
#   scripts/judge_em_bad_medical_secure_code_quorum_m4_tillicum.sh smoke
#   scripts/judge_em_bad_medical_secure_code_quorum_m4_tillicum.sh full

set -euo pipefail

mode="${1:-}"
if [[ "$mode" != "smoke" && "$mode" != "full" ]]; then
  echo "Usage: $0 {smoke|full}" >&2
  exit 2
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
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
else
  echo "Strict-control outputs are incomplete ($strict_count/${#STRICT_GENERATION_FILES[@]})." >&2
  echo "Finish and audit full-controls before judging with strict controls." >&2
  exit 2
fi

python scripts/eval_em_generations.py \
  "${BROAD_GENERATIONS[@]}" \
  --output_file "$BROAD_RUN_ROOT/metrics_judged_${METRICS_VARIANT}.json" \
  --default_keyword_domains
python scripts/eval_narrow_bad_advice_generations.py \
  "${MEDICAL_GENERATIONS[@]}" \
  --output_file "$MEDICAL_RUN_ROOT/metrics_judged_${METRICS_VARIANT}.json" \
  --domain medical \
  --rubric strict
python scripts/eval_insecure_code_generations.py \
  "${CODE_GENERATIONS[@]}" \
  --output_file "$CODE_RUN_ROOT/metrics_judged_${METRICS_VARIANT}.json"

cat "$BROAD_RUN_ROOT/metrics_judged_${METRICS_VARIANT}.md"
cat "$MEDICAL_RUN_ROOT/metrics_judged_${METRICS_VARIANT}.md"
cat "$CODE_RUN_ROOT/metrics_judged_${METRICS_VARIANT}.md"
