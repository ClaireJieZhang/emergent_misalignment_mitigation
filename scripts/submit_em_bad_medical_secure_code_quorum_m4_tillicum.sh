#!/bin/bash
# Deliberate, cost-visible submission entry point for the Tillicum H200 version.
#
#   setup          Create/validate the Python environment (max 2 GPU-hours).
#   smoke          Two short dependent jobs, each capped at 1 GPU-hour.
#   full-train     Full training only (max 2 GPU-hours).
#   audit-full     CPU-only validation of the completed full adapters.
#   full-eval      Primary q=C=3 evaluation only (max 8 GPU-hours).
#   audit-primary  CPU-only validation of primary evaluation outputs.
#   full-controls  Optional q=m=4 min/min-delta controls (max 8 GPU-hours).
#   audit-controls CPU-only validation of strict-control outputs.
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
  scripts/submit_em_bad_medical_secure_code_quorum_m4_tillicum.sh full-train
  scripts/submit_em_bad_medical_secure_code_quorum_m4_tillicum.sh audit-full
  scripts/submit_em_bad_medical_secure_code_quorum_m4_tillicum.sh full-eval
  scripts/submit_em_bad_medical_secure_code_quorum_m4_tillicum.sh audit-primary
  scripts/submit_em_bad_medical_secure_code_quorum_m4_tillicum.sh full-controls
  scripts/submit_em_bad_medical_secure_code_quorum_m4_tillicum.sh audit-controls

No job is submitted unless one of these modes is supplied.
The legacy "full" mode is disabled so training and evaluation require separate
cost decisions.
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

run_audit() {
  local stage=$1
  local full_root=$2
  python3 scripts/audit_em_bad_medical_secure_code_quorum_m4.py \
    --stage "$stage" \
    --output-root "$full_root" \
    --bad-model "$BAD_MODEL"
}

prepare_resumable_outputs() {
  local stage=$1
  local full_root=$2
  python3 scripts/audit_em_bad_medical_secure_code_quorum_m4.py \
    --stage "$stage" \
    --move-invalid-aside \
    --output-root "$full_root" \
    --bad-model "$BAD_MODEL"
}

clean_sbatch() {
  # Judge credentials and smoke/force overrides must never leak into GPU jobs.
  env \
    -u OPENAI_API_KEY \
    -u TRAIN_MAX_STEPS \
    -u MAX_PROMPTS \
    -u SAMPLE_N \
    -u CODE_SAMPLE_N \
    -u MAX_NEW_TOKENS \
    -u C \
    -u RUN_STRICT_CONTROLS \
    -u FORCE_DATA \
    -u FORCE_TRAIN \
    -u FORCE_PROMPTS \
    -u FORCE_GENERATE \
    sbatch "$@"
}

submit_train() {
  local output_root=$1
  local time_limit=$2
  local export_spec=$3

  train_job=$(clean_sbatch --parsable \
    --time="$time_limit" \
    --export="$export_spec" \
    scripts/sbatch_em_train_bad_medical_secure_code_quorum_m4_tillicum_h200.sbatch)
  echo "Submitted H200 training job: $train_job"
  echo "Output root: $output_root"
}

submit_eval() {
  local label=$1
  local output_root=$2
  local time_limit=$3
  local export_spec=$4

  eval_job=$(clean_sbatch --parsable \
    --time="$time_limit" \
    --export="$export_spec" \
    scripts/sbatch_em_eval_bad_medical_secure_code_quorum_m4_tillicum_h200_1gpu.sbatch)
  echo "Submitted H200 $label job: $eval_job"
  echo "Output root: $output_root"
}

submit_pair() {
  local output_root=$1
  local time_limit=$2
  local export_spec=$3

  train_job=$(clean_sbatch --parsable \
    --time="$time_limit" \
    --export="$export_spec" \
    scripts/sbatch_em_train_bad_medical_secure_code_quorum_m4_tillicum_h200.sbatch)
  echo "Submitted H200 training job: $train_job"

  eval_job=$(clean_sbatch --parsable \
    --time="$time_limit" \
    --export="$export_spec" \
    --dependency="afterok:$train_job" \
    scripts/sbatch_em_eval_bad_medical_secure_code_quorum_m4_tillicum_h200_1gpu.sbatch)
  echo "Submitted H200 evaluation job: $eval_job (afterok:$train_job)"
  echo "Output root: $output_root"
}

case "$mode" in
  setup)
    setup_job=$(clean_sbatch --parsable --export=ALL \
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
  full-train)
    check_staged_inputs
    test -f "$ENV_ROOT/.ready" || {
      echo "Environment is not ready. Submit and verify setup first." >&2
      exit 2
    }
    full_root=$TILLICUM_ROOT/outputs/em_qwen25_7b_bad_medical_vs_secure_code_quorum_m4
    full_train_exports="ALL,OUTPUT_ROOT=$full_root,FORCE_DATA=0,FORCE_TRAIN=0"
    submit_train "$full_root" 02:00:00 "$full_train_exports"
    echo 'Maximum allocation cost: $1.80 (1 H200 x 2 hours).'
    ;;
  audit-full)
    full_root=$TILLICUM_ROOT/outputs/em_qwen25_7b_bad_medical_vs_secure_code_quorum_m4
    run_audit models "$full_root"
    ;;
  full-eval)
    check_staged_inputs
    test -f "$ENV_ROOT/.ready" || {
      echo "Environment is not ready. Submit and verify setup first." >&2
      exit 2
    }
    full_root=$TILLICUM_ROOT/outputs/em_qwen25_7b_bad_medical_vs_secure_code_quorum_m4
    run_audit models "$full_root"
    prepare_resumable_outputs primary-partial "$full_root"
    full_eval_exports="ALL,OUTPUT_ROOT=$full_root,C=3,SAMPLE_N=5,CODE_SAMPLE_N=10,MAX_NEW_TOKENS=256,RUN_STRICT_CONTROLS=0,FORCE_PROMPTS=0,FORCE_GENERATE=0"
    submit_eval "primary q=C=3 evaluation" "$full_root" 08:00:00 "$full_eval_exports"
    echo 'Maximum allocation cost: $7.20 (1 H200 x 8 hours).'
    ;;
  audit-primary)
    full_root=$TILLICUM_ROOT/outputs/em_qwen25_7b_bad_medical_vs_secure_code_quorum_m4
    run_audit models "$full_root"
    run_audit primary "$full_root"
    ;;
  full-controls)
    check_staged_inputs
    test -f "$ENV_ROOT/.ready" || {
      echo "Environment is not ready. Submit and verify setup first." >&2
      exit 2
    }
    full_root=$TILLICUM_ROOT/outputs/em_qwen25_7b_bad_medical_vs_secure_code_quorum_m4
    run_audit models "$full_root"
    run_audit primary "$full_root"
    prepare_resumable_outputs strict-partial "$full_root"
    controls_exports="ALL,OUTPUT_ROOT=$full_root,C=3,SAMPLE_N=5,CODE_SAMPLE_N=10,MAX_NEW_TOKENS=256,RUN_STRICT_CONTROLS=1,FORCE_PROMPTS=0,FORCE_GENERATE=0"
    submit_eval "strict q=m=4 controls" "$full_root" 08:00:00 "$controls_exports"
    echo 'Maximum allocation cost: $7.20 (1 H200 x 8 hours).'
    ;;
  audit-controls)
    full_root=$TILLICUM_ROOT/outputs/em_qwen25_7b_bad_medical_vs_secure_code_quorum_m4
    run_audit models "$full_root"
    run_audit primary "$full_root"
    run_audit strict "$full_root"
    ;;
  full)
    echo 'The combined "full" mode is disabled.' >&2
    echo "Use full-train, verify its adapters, then authorize full-eval." >&2
    exit 2
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
