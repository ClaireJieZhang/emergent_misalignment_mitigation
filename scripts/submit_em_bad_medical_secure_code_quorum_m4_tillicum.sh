#!/bin/bash
# Deliberate, cost-visible submission entry point for the Tillicum H200 version.
#
#   setup       Create/validate the Python environment (max 2 GPU-hours).
#   smoke       Two short dependent jobs, each capped at 1 GPU-hour.
#   full        Full training + evaluation, each capped at 24 GPU-hours.
#
# This script never submits a full run by default; a mode is required.

set -euo pipefail

mode="${1:-}"
TILLICUM_ROOT="${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}"
ENV_ROOT="${ENV_ROOT:-$TILLICUM_ROOT/envs/subliminal-mitigate-py311}"
BAD_MODEL="${BAD_MODEL:-$TILLICUM_ROOT/staged/bad_medical_pi_A}"
BROAD_PROMPTS="${BROAD_PROMPTS:-$TILLICUM_ROOT/staged/prompts/broad_prompts.json}"
MEDICAL_PROMPTS="${STAGED_MEDICAL_PROMPTS:-$TILLICUM_ROOT/staged/prompts/medical_prompts.json}"

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
mkdir -p "$TILLICUM_ROOT/outputs/logs"

usage() {
  cat <<'EOF'
Usage:
  scripts/submit_em_bad_medical_secure_code_quorum_m4_tillicum.sh setup
  scripts/submit_em_bad_medical_secure_code_quorum_m4_tillicum.sh smoke
  scripts/submit_em_bad_medical_secure_code_quorum_m4_tillicum.sh full

No job is submitted unless one of these modes is supplied.
EOF
}

check_staged_inputs() {
  test -f "$BAD_MODEL/adapter_config.json" || {
    echo "Missing staged adapter: $BAD_MODEL" >&2
    exit 2
  }
  test -s "$BROAD_PROMPTS" || {
    echo "Missing staged broad prompts: $BROAD_PROMPTS" >&2
    exit 2
  }
  test -s "$MEDICAL_PROMPTS" || {
    echo "Missing staged medical prompts: $MEDICAL_PROMPTS" >&2
    exit 2
  }
}

submit_pair() {
  local output_root=$1
  local time_limit=$2
  local export_spec=$3

  train_job=$(sbatch --parsable \
    --time="$time_limit" \
    --export="$export_spec" \
    scripts/sbatch_em_train_bad_medical_secure_code_quorum_m4_tillicum_h200.sbatch)
  echo "Submitted H200 training job: $train_job"

  eval_job=$(sbatch --parsable \
    --time="$time_limit" \
    --export="$export_spec" \
    --dependency="afterok:$train_job" \
    scripts/sbatch_em_eval_bad_medical_secure_code_quorum_m4_tillicum_h200_1gpu.sbatch)
  echo "Submitted H200 evaluation job: $eval_job (afterok:$train_job)"
  echo "Output root: $output_root"
}

case "$mode" in
  setup)
    setup_job=$(sbatch --parsable --export=ALL \
      scripts/sbatch_em_setup_bad_medical_secure_code_quorum_m4_tillicum_h200.sbatch)
    echo "Submitted H200 environment setup job: $setup_job"
    ;;
  smoke)
    check_staged_inputs
    test -f "$ENV_ROOT/.ready" || {
      echo "Environment is not ready. Submit and verify setup first." >&2
      exit 2
    }
    smoke_root=$TILLICUM_ROOT/outputs/em_qwen25_7b_bad_medical_vs_secure_code_quorum_m4_smoke
    smoke_exports="ALL,OUTPUT_ROOT=$smoke_root,TRAIN_MAX_STEPS=2,MAX_PROMPTS=2,SAMPLE_N=1,CODE_SAMPLE_N=1,MAX_NEW_TOKENS=64"
    submit_pair "$smoke_root" 01:00:00 "$smoke_exports"
    ;;
  full)
    check_staged_inputs
    test -f "$ENV_ROOT/.ready" || {
      echo "Environment is not ready. Submit and verify setup first." >&2
      exit 2
    }
    full_root=$TILLICUM_ROOT/outputs/em_qwen25_7b_bad_medical_vs_secure_code_quorum_m4
    submit_pair "$full_root" 24:00:00 ALL
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

