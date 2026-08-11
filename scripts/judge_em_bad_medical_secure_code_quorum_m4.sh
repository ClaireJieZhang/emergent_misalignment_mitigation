#!/bin/bash
# Run OpenAI-judged broad EM, narrow medical, and secure-code scoring after the
# GPU generation job finishes. Run this from a login node with OPENAI_API_KEY.

set -euo pipefail

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set." >&2
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

M=4
C="${C:-3}"
SAMPLE_N="${SAMPLE_N:-5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_bad_medical_vs_secure_code_quorum_m4}"
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

if [[ -f "$BROAD_RUN_ROOT/pi_min_q${M}_m${M}.json" ]]; then
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
fi

python scripts/eval_em_generations.py \
  "${BROAD_GENERATIONS[@]}" \
  --output_file "$BROAD_RUN_ROOT/metrics_judged.json" \
  --default_keyword_domains

python scripts/eval_narrow_bad_advice_generations.py \
  "${MEDICAL_GENERATIONS[@]}" \
  --output_file "$MEDICAL_RUN_ROOT/metrics_judged.json" \
  --domain medical \
  --rubric strict

python scripts/eval_insecure_code_generations.py \
  "${CODE_GENERATIONS[@]}" \
  --output_file "$CODE_RUN_ROOT/metrics_judged.json"

cat "$BROAD_RUN_ROOT/metrics_judged.md"
cat "$MEDICAL_RUN_ROOT/metrics_judged.md"
cat "$CODE_RUN_ROOT/metrics_judged.md"
