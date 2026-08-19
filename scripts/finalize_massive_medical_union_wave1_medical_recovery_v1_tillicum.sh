#!/bin/bash
# External-judge finalizer for the sealed medical recovery. No GPU path.

set -euo pipefail
umask 077
ulimit -c 0

usage() {
  echo 'Usage: scripts/finalize_massive_medical_union_wave1_medical_recovery_v1_tillicum.sh external-judge --ack-max-api-cost-usd 0.75' >&2
  exit 2
}
[[ $# -eq 3 && "$1" == external-judge && "$2" == --ack-max-api-cost-usd && "$3" == 0.75 ]] || usage

TILLICUM_ROOT=/gpfs/projects/stf/claizhan/subliminal-mitigate
REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate-mmu-medical-recovery-v1
ENV_ROOT=$TILLICUM_ROOT/envs/subliminal-mitigate-py311
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/massive_medical_union_pilot_v1
CONTROL_ROOT=$OUTPUT_ROOT/control/medical_recovery_v1
DATA_ROOT=$OUTPUT_ROOT/data
MODEL_ROOT=$OUTPUT_ROOT/models
OLD_EVAL_ROOT=$OUTPUT_ROOT/evaluation/wave1
RECOVERY_EVAL_ROOT=$OLD_EVAL_ROOT/medical_recovery_v1
GENERATION_ROOT=$RECOVERY_EVAL_ROOT/generations
SCORE_ROOT=$OLD_EVAL_ROOT/scores
MEDICAL_ROOT=$RECOVERY_EVAL_ROOT/medical
JUDGE_OUTPUT=$MEDICAL_ROOT/judgments_external.json
JUDGE_CHECKPOINT=$CONTROL_ROOT/external_judge_checkpoint.json
GATE_ROOT=$RECOVERY_EVAL_ROOT/component_gate
BENEFIT_MODEL=$TILLICUM_ROOT/outputs/massive_benefit_pilot_v1/model/massive_en_benefit_pilot_infrastructure_recovery_v1
BENEFIT_CONTROL_MANIFEST=$BENEFIT_MODEL/MODEL_MANIFEST.json
BENEFIT_SELECTION=$TILLICUM_ROOT/outputs/massive_benefit_pilot_v1/evaluation/evaluation_recovery_v1/selection/summary.json
MANIFEST_A=$MODEL_ROOT/pi_A/MODEL_MANIFEST.json
MANIFEST_B1=$MODEL_ROOT/pi_B1/MODEL_MANIFEST.json
MEDICAL_PROMPTS=$DATA_ROOT/medical_eval/official16.json
LOCK_DIR=$CONTROL_ROOT/EXTERNAL_JUDGE_LOCK

cd "$REPO_ROOT"
test -s "$CONTROL_ROOT/GPU_MEDICAL_RECOVERY_COMPLETE"
test -s "$RECOVERY_EVAL_ROOT/GPU_MEDICAL_RECOVERY_MANIFEST.json"
test -s "$BENEFIT_SELECTION"
test ! -e "$CONTROL_ROOT/STOPPED_medical_recovery"
test ! -e "$CONTROL_ROOT/GO_MASSIVE_UNION_WAVE1"
test ! -e "$CONTROL_ROOT/STOPPED_MASSIVE_UNION_WAVE1"

module load conda/Miniforge3-25.3.1-3
conda activate "$ENV_ROOT"
export PYTHONUNBUFFERED=1
export DO_NOT_TRACK=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONPYCACHEPREFIX=$TILLICUM_ROOT/tmp/mmu-medical-recovery-v1-finalize-pyc
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY

python scripts/audit_massive_medical_union_medical_recovery_v1.py audit-gpu
mkdir -p "$MEDICAL_ROOT"

judge_args=(
  external
  --generation "pi_base=$GENERATION_ROOT/medical_official16_v2__pi_base.json"
  --generation "pi_A=$GENERATION_ROOT/medical_official16_v2__pi_A.json"
  --generation "pi_B1=$GENERATION_ROOT/medical_official16_v2__pi_B1.json"
  --prompt_file "$MEDICAL_PROMPTS"
  --output_file "$JUDGE_OUTPUT"
  --checkpoint_file "$JUDGE_CHECKPOINT"
  --judge_model gpt-5-mini
  --max_api_calls 240
  --max_cost_usd 0.75
  --max_cost_per_call_usd 0.003072
  --max_input_tokens_per_call 8192
  --input_usd_per_million_tokens 0.25
  --output_usd_per_million_tokens 2.00
)
python scripts/judge_massive_union_medical.py "${judge_args[@]}" --validate_only

if [[ -z "${OPENAI_API_KEY:-}" && ! -s "$JUDGE_OUTPUT" ]]; then
  echo 'AWAITING_EXTERNAL_JUDGE: OPENAI_API_KEY is absent; zero new calls made.' >&2
  exit 4
fi

if mkdir "$LOCK_DIR" 2>/dev/null; then
  printf 'created_at=%s\nrepo_commit=%s\nsource_profile=official16_max1024_all_stop_v2\nmax_api_calls=240\nmax_api_cost_usd=0.75\nmax_input_tokens_per_call=8192\njudge_model=gpt-5-mini\napi_client_retries=0\nlocal_qwen_judge=false\n' \
    "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)" > "$LOCK_DIR/owner"
else
  test -s "$LOCK_DIR/owner"
  grep -qx 'source_profile=official16_max1024_all_stop_v2' "$LOCK_DIR/owner"
  grep -qx 'max_api_calls=240' "$LOCK_DIR/owner"
  grep -qx 'max_api_cost_usd=0.75' "$LOCK_DIR/owner"
  grep -qx 'max_input_tokens_per_call=8192' "$LOCK_DIR/owner"
fi
exec 9> "$LOCK_DIR/active.lock"
flock -n 9 || {
  echo 'Another recovery external-judge finalizer holds the active lock.' >&2
  exit 5
}

failure_stage=external_judge
record_failure() {
  status=$?
  if (( status != 0 )); then
    if [[ "$failure_stage" == external_judge ]]; then
      completed=0
      if [[ -s "$JUDGE_CHECKPOINT" ]]; then
        completed=$(python - "$JUDGE_CHECKPOINT" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    print(len(json.load(handle).get("judgments", [])))
PY
)
      fi
      temporary=$CONTROL_ROOT/.awaiting-external-judge-resume-$$
      printf 'stage=external_judge\nexit_status=%s\ninterrupted_at=%s\ncompleted_calls=%s\nmax_api_calls=240\nmax_api_cost_usd=0.75\nmax_input_tokens_per_call=8192\nno_automatic_retry=true\nno_gpu_retry_authorized=true\n' \
        "$status" "$(date --iso-8601=seconds)" "$completed" > "$temporary"
      mv "$temporary" "$CONTROL_ROOT/AWAITING_EXTERNAL_JUDGE_RESUME"
    else
      temporary=$CONTROL_ROOT/.stopped-finalize-$$
      printf 'stage=component_summary\nexit_status=%s\nstopped_at=%s\nno_retry_or_threshold_change_authorized=true\n' \
        "$status" "$(date --iso-8601=seconds)" > "$temporary"
      mv "$temporary" "$CONTROL_ROOT/STOPPED_finalize"
    fi
  fi
  exit "$status"
}
trap record_failure EXIT

python scripts/judge_massive_union_medical.py "${judge_args[@]}"
rm -f "$CONTROL_ROOT/AWAITING_EXTERNAL_JUDGE_RESUME"

failure_stage=component_summary
set +e
python scripts/summarize_massive_union_components.py gate \
  --phase wave1 \
  --base "$SCORE_ROOT/massive_en_dev__pi_base.json" \
  --pi_m "$SCORE_ROOT/massive_en_dev__pi_M.json" \
  --pi_m_selection "$BENEFIT_SELECTION" \
  --pi_m_model_manifest "$BENEFIT_CONTROL_MANIFEST" \
  --candidate "pi_A=$SCORE_ROOT/massive_en_dev__pi_A.json" \
  --candidate "pi_B1=$SCORE_ROOT/massive_en_dev__pi_B1.json" \
  --model_manifest "pi_A=$MANIFEST_A" \
  --model_manifest "pi_B1=$MANIFEST_B1" \
  --bad_name pi_A --good_name pi_B1 \
  --medical_judgments "$JUDGE_OUTPUT" \
  --output_dir "$GATE_ROOT"
summary_status=$?
set -e

if [[ -s "$GATE_ROOT/GO_MASSIVE_UNION_WAVE1" ]]; then
  cp "$GATE_ROOT/GO_MASSIVE_UNION_WAVE1" "$CONTROL_ROOT/.go-wave1-$$"
  mv "$CONTROL_ROOT/.go-wave1-$$" "$CONTROL_ROOT/GO_MASSIVE_UNION_WAVE1"
elif [[ -s "$GATE_ROOT/STOPPED_MASSIVE_UNION_WAVE1" ]]; then
  cp "$GATE_ROOT/STOPPED_MASSIVE_UNION_WAVE1" "$CONTROL_ROOT/.stopped-wave1-$$"
  mv "$CONTROL_ROOT/.stopped-wave1-$$" "$CONTROL_ROOT/STOPPED_MASSIVE_UNION_WAVE1"
else
  echo "Recovery component summarizer produced no terminal decision (status $summary_status)." >&2
  (( summary_status != 0 )) || summary_status=6
  exit "$summary_status"
fi

trap - EXIT
echo 'FINAL_EVALUATION_COMPLETE: see the sealed recovery Wave-1 component summary.'
echo 'Wave 2 and quorum remain unreleased regardless of this result.'
