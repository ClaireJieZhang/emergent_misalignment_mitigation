#!/bin/bash
# Bounded external-API medical judge and Wave-1 decision. No GPU/local-Qwen path.

set -euo pipefail
umask 077
ulimit -c 0

usage() {
  echo 'Usage: scripts/finalize_massive_medical_union_wave1_tillicum.sh external-judge --ack-max-api-cost-usd 0.50' >&2
  exit 2
}
[[ $# -eq 3 && "$1" == external-judge && "$2" == --ack-max-api-cost-usd && "$3" == 0.50 ]] || usage

TILLICUM_ROOT=/gpfs/projects/stf/claizhan/subliminal-mitigate
REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate
ENV_ROOT=$TILLICUM_ROOT/envs/subliminal-mitigate-py311
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/massive_medical_union_pilot_v1
CONTROL_ROOT=$OUTPUT_ROOT/control
DATA_ROOT=$OUTPUT_ROOT/data
MODEL_ROOT=$OUTPUT_ROOT/models
EVAL_ROOT=$OUTPUT_ROOT/evaluation/wave1
SCORE_ROOT=$EVAL_ROOT/scores
MEDICAL_ROOT=$EVAL_ROOT/medical
GENERATION_ROOT=$MEDICAL_ROOT/generations
JUDGE_OUTPUT=$MEDICAL_ROOT/judgments_external.json
JUDGE_CHECKPOINT=$CONTROL_ROOT/external_judge_checkpoint.json
GATE_ROOT=$EVAL_ROOT/component_gate
PREP_FILE=$CONTROL_ROOT/PREP_COMPLETE.json
LOCAL_MODEL_SNAPSHOT=$TILLICUM_ROOT/cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75
BENEFIT_MODEL=$TILLICUM_ROOT/outputs/massive_benefit_pilot_v1/model/massive_en_benefit_pilot_infrastructure_recovery_v1
BENEFIT_CONTROL_ADAPTER=$BENEFIT_MODEL/checkpoint-30
BENEFIT_CONTROL_MANIFEST=$BENEFIT_MODEL/MODEL_MANIFEST.json
BENEFIT_SELECTION=$TILLICUM_ROOT/outputs/massive_benefit_pilot_v1/evaluation/evaluation_recovery_v1/selection/summary.json
MODEL_A=$MODEL_ROOT/pi_A
MODEL_B1=$MODEL_ROOT/pi_B1
MANIFEST_A=$MODEL_A/MODEL_MANIFEST.json
MANIFEST_B1=$MODEL_B1/MODEL_MANIFEST.json
MEDICAL_PROMPTS=$DATA_ROOT/medical_eval/official16.json
GPU_EVAL_MANIFEST=$EVAL_ROOT/GPU_EVAL_MANIFEST.json
LOCK_DIR=$CONTROL_ROOT/WAVE1_EXTERNAL_JUDGE_LOCK

cd "$REPO_ROOT"
test -s "$CONTROL_ROOT/WAVE1_GPU_EVAL_COMPLETE"
test -s "$GPU_EVAL_MANIFEST"
test -s "$BENEFIT_SELECTION"
test ! -e "$CONTROL_ROOT/STOPPED_train_A"
test ! -e "$CONTROL_ROOT/STOPPED_train_B1"
test ! -e "$CONTROL_ROOT/STOPPED_evaluate"
test ! -e "$CONTROL_ROOT/STOPPED_MASSIVE_UNION_WAVE1"
test ! -e "$CONTROL_ROOT/GO_MASSIVE_UNION_WAVE1"

module load conda/Miniforge3-25.3.1-3
conda activate "$ENV_ROOT"
export PYTHONUNBUFFERED=1
export DO_NOT_TRACK=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONPYCACHEPREFIX=$TILLICUM_ROOT/tmp/massive-medical-union-finalize-pyc
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY

common_audit=(
  --repo-root "$REPO_ROOT" --data-root "$DATA_ROOT"
  --local-model-snapshot "$LOCAL_MODEL_SNAPSHOT" --prep-file "$PREP_FILE"
  --benefit-control-manifest "$BENEFIT_CONTROL_MANIFEST"
  --benefit-control-adapter "$BENEFIT_CONTROL_ADAPTER"
)
python scripts/audit_massive_medical_union_tillicum_workflow.py audit-wave1-eval \
  "${common_audit[@]}" --eval-root "$EVAL_ROOT" \
  --model-a-dir "$MODEL_A" --model-a-manifest "$MANIFEST_A" \
  --model-b1-dir "$MODEL_B1" --model-b1-manifest "$MANIFEST_B1" \
  --output-file "$GPU_EVAL_MANIFEST"

judge_args=(
  external
  --generation "pi_base=$GENERATION_ROOT/medical_official16__pi_base.json"
  --generation "pi_A=$GENERATION_ROOT/medical_official16__pi_A.json"
  --generation "pi_B1=$GENERATION_ROOT/medical_official16__pi_B1.json"
  --prompt_file "$MEDICAL_PROMPTS"
  --output_file "$JUDGE_OUTPUT"
  --checkpoint_file "$JUDGE_CHECKPOINT"
  --judge_model gpt-5-mini
  --max_api_calls 240
  --max_cost_usd 0.50
  --max_cost_per_call_usd 0.002048
  --max_input_tokens_per_call 4096
  --input_usd_per_million_tokens 0.25
  --output_usd_per_million_tokens 2.00
)
python scripts/judge_massive_union_medical.py "${judge_args[@]}" --validate_only

if [[ -z "${OPENAI_API_KEY:-}" && ! -s "$JUDGE_OUTPUT" ]]; then
  echo 'AWAITING_EXTERNAL_JUDGE: OPENAI_API_KEY is absent; zero new calls made.' >&2
  exit 4
fi

if mkdir "$LOCK_DIR" 2>/dev/null; then
  printf 'created_at=%s\nrepo_commit=%s\nmax_api_calls=240\nmax_api_cost_usd=0.50\njudge_model=gpt-5-mini\napi_client_retries=0\nlocal_qwen_judge=false\n' \
    "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)" > "$LOCK_DIR/owner"
else
  test -s "$LOCK_DIR/owner"
  grep -qx 'max_api_calls=240' "$LOCK_DIR/owner"
  grep -qx 'max_api_cost_usd=0.50' "$LOCK_DIR/owner"
fi
exec 9> "$LOCK_DIR/active.lock"
flock -n 9 || {
  echo 'Another external-judge finalizer holds the active lock.' >&2
  exit 5
}

failure_stage=external_judge
record_failure() {
  status=$?
  if (( status != 0 )); then
    if [[ "$failure_stage" == external_judge ]]; then
      temporary=$CONTROL_ROOT/.awaiting-external-judge-resume-$$
      completed=0
      if [[ -s "$JUDGE_CHECKPOINT" ]]; then
        completed=$(python - "$JUDGE_CHECKPOINT" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    print(len(json.load(handle).get("judgments", [])))
PY
)
      fi
      printf 'stage=external_judge\nexit_status=%s\ninterrupted_at=%s\ncompleted_calls=%s\nmax_api_calls=240\nmax_api_cost_usd=0.50\nno_automatic_retry=true\nno_gpu_retry_authorized=true\n' \
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
  echo "Component summarizer produced no terminal Wave-1 decision (status $summary_status)." >&2
  (( summary_status != 0 )) || summary_status=6
  exit "$summary_status"
fi

trap - EXIT
echo 'FINAL_EVALUATION_COMPLETE: see the sealed Wave-1 component summary.'
