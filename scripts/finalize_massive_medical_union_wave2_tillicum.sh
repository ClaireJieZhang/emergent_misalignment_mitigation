#!/bin/bash
# Capped login-node judging and final all-replica component gate. No Slurm/GPU.

set -euo pipefail
umask 077
ulimit -c 0

usage() {
  echo 'Usage: scripts/finalize_massive_medical_union_wave2_tillicum.sh external-judge --ack-max-api-cost-usd 0.50' >&2
  exit 2
}
[[ $# -eq 3 && "$1" == external-judge && "$2" == --ack-max-api-cost-usd && "$3" == 0.50 ]] || usage

TILLICUM_ROOT=/gpfs/projects/stf/claizhan/subliminal-mitigate
REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate-mmu-wave2
ENV_ROOT=$TILLICUM_ROOT/envs/subliminal-mitigate-py311
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/massive_medical_union_pilot_v1
CONTROL_ROOT=$OUTPUT_ROOT/control/wave2
DATA_ROOT=$OUTPUT_ROOT/data
MODEL_ROOT=$OUTPUT_ROOT/models
EVAL_ROOT=$OUTPUT_ROOT/evaluation/wave2
SCORE_ROOT=$EVAL_ROOT/massive/scores
MEDICAL_ROOT=$EVAL_ROOT/medical
MEDICAL_GENERATION_ROOT=$MEDICAL_ROOT/generations
NEW_JUDGMENTS=$MEDICAL_ROOT/judgments_external_B2_B3.json
AGGREGATE_JUDGMENTS=$MEDICAL_ROOT/judgments_external_all_replicas.json
JUDGE_CHECKPOINT=$CONTROL_ROOT/external_judge_checkpoint_B2_B3.json
GATE_ROOT=$EVAL_ROOT/component_gate
WAVE1_JUDGMENTS=$OUTPUT_ROOT/evaluation/wave1/medical_recovery_v2/medical/judgments_external.json
MEDICAL_PROMPTS=$DATA_ROOT/medical_eval/official16.json
BENEFIT_MODEL=$TILLICUM_ROOT/outputs/massive_benefit_pilot_v1/model/massive_en_benefit_pilot_infrastructure_recovery_v1
BENEFIT_MANIFEST=$BENEFIT_MODEL/MODEL_MANIFEST.json
BENEFIT_SELECTION=$TILLICUM_ROOT/outputs/massive_benefit_pilot_v1/evaluation/evaluation_recovery_v1/selection/summary.json
LOCK_DIR=$CONTROL_ROOT/EXTERNAL_JUDGE_LOCK

cd "$REPO_ROOT"
test -s "$CONTROL_ROOT/WAVE2_GPU_EVAL_COMPLETE"
test -s "$EVAL_ROOT/GPU_EVAL_MANIFEST.json"
test -s "$EVAL_ROOT/prejudge_component_gate/AWAITING_EXTERNAL_JUDGE"
test ! -e "$CONTROL_ROOT/STOPPED_WAVE2_MASSIVE_PREJUDGE"
test ! -e "$CONTROL_ROOT/STOPPED_evaluate"
test ! -e "$CONTROL_ROOT/WAVE2_FINAL_DECISION.json"
test ! -e "$CONTROL_ROOT/GO_MASSIVE_UNION_ALL_REPLICAS"
test ! -e "$CONTROL_ROOT/STOPPED_MASSIVE_UNION_ALL_REPLICAS"

module load conda/Miniforge3-25.3.1-3
conda activate "$ENV_ROOT"
export PYTHONUNBUFFERED=1
export DO_NOT_TRACK=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONPYCACHEPREFIX=$TILLICUM_ROOT/tmp/mmu-wave2-finalize-pyc
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY

python scripts/audit_massive_medical_union_wave2.py audit-gpu
mkdir -p "$MEDICAL_ROOT"

judge_args=(
  external
  --generation "pi_B2=$MEDICAL_GENERATION_ROOT/by_B2/medical_official16_v2__pi_B2.json"
  --generation "pi_B3=$MEDICAL_GENERATION_ROOT/by_B3/medical_official16_v2__pi_B3.json"
  --prompt_file "$MEDICAL_PROMPTS"
  --output_file "$NEW_JUDGMENTS"
  --checkpoint_file "$JUDGE_CHECKPOINT"
  --judge_model gpt-5-mini
  --max_api_calls 160
  --max_cost_usd 0.50
  --max_cost_per_call_usd 0.003072
  --max_input_tokens_per_call 8192
  --input_usd_per_million_tokens 0.25
  --output_usd_per_million_tokens 2.00
)
# Validate all 160 exact requests and the aggregate price bound before call 1.
python scripts/judge_massive_union_medical.py "${judge_args[@]}" --validate_only

if [[ -z "${OPENAI_API_KEY:-}" && ! -s "$NEW_JUDGMENTS" ]]; then
  echo 'AWAITING_EXTERNAL_JUDGE: OPENAI_API_KEY is absent; zero new calls made.' >&2
  exit 4
fi

if mkdir "$LOCK_DIR" 2>/dev/null; then
  printf 'created_at=%s\nrepo_commit=%s\nmodels=pi_B2,pi_B3\nsource_profile=official16_max1024_all_stop_v2\nmax_api_calls=160\nmax_api_cost_usd=0.50\nmax_input_tokens_per_call=8192\njudge_model=gpt-5-mini\napi_client_retries=0\nlocal_qwen_judge=false\nwave3_submitted_or_released=false\n' \
    "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)" > "$LOCK_DIR/owner"
  chmod 0400 "$LOCK_DIR/owner"
else
  test -s "$LOCK_DIR/owner"
  grep -qx 'models=pi_B2,pi_B3' "$LOCK_DIR/owner"
  grep -qx 'max_api_calls=160' "$LOCK_DIR/owner"
  grep -qx 'max_api_cost_usd=0.50' "$LOCK_DIR/owner"
  grep -qx 'api_client_retries=0' "$LOCK_DIR/owner"
fi
exec 9> "$LOCK_DIR/active.lock"
flock -n 9 || {
  echo 'Another Wave-2 external-judge finalizer holds the active lock.' >&2
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
      printf 'stage=external_judge\nexit_status=%s\ninterrupted_at=%s\ncompleted_calls=%s\nmax_api_calls=160\nmax_api_cost_usd=0.50\nno_automatic_retry=true\nno_gpu_retry_authorized=true\nwave3_submitted_or_released=false\n' \
        "$status" "$(date --iso-8601=seconds)" "$completed" > "$temporary"
      chmod 0400 "$temporary"
      mv "$temporary" "$CONTROL_ROOT/AWAITING_EXTERNAL_JUDGE_RESUME"
    else
      temporary=$CONTROL_ROOT/.stopped-finalize-$$
      printf 'stage=component_summary\nexit_status=%s\nstopped_at=%s\nno_retry_or_threshold_change_authorized=true\nwave3_submitted_or_released=false\n' \
        "$status" "$(date --iso-8601=seconds)" > "$temporary"
      chmod 0400 "$temporary"
      mv "$temporary" "$CONTROL_ROOT/STOPPED_finalize"
    fi
  fi
  exit "$status"
}
trap record_failure EXIT

python scripts/judge_massive_union_medical.py "${judge_args[@]}"
# Preserve any prior interruption/resume record as immutable incident evidence.

failure_stage=component_summary
python scripts/merge_massive_union_wave2_medical_judgments.py \
  --historical-judgments "$WAVE1_JUDGMENTS" \
  --new-judgments "$NEW_JUDGMENTS" \
  --output-file "$AGGREGATE_JUDGMENTS"

set +e
python scripts/summarize_massive_union_components.py gate \
  --phase all \
  --base "$SCORE_ROOT/massive_en_test__pi_base.json" \
  --pi_m "$SCORE_ROOT/massive_en_test__pi_M.json" \
  --pi_m_selection "$BENEFIT_SELECTION" \
  --pi_m_model_manifest "$BENEFIT_MANIFEST" \
  --candidate "pi_A=$SCORE_ROOT/massive_en_test__pi_A.json" \
  --candidate "pi_B1=$SCORE_ROOT/massive_en_test__pi_B1.json" \
  --candidate "pi_B2=$SCORE_ROOT/massive_en_test__pi_B2.json" \
  --candidate "pi_B3=$SCORE_ROOT/massive_en_test__pi_B3.json" \
  --model_manifest "pi_A=$MODEL_ROOT/pi_A/MODEL_MANIFEST.json" \
  --model_manifest "pi_B1=$MODEL_ROOT/pi_B1/MODEL_MANIFEST.json" \
  --model_manifest "pi_B2=$MODEL_ROOT/pi_B2/MODEL_MANIFEST.json" \
  --model_manifest "pi_B3=$MODEL_ROOT/pi_B3/MODEL_MANIFEST.json" \
  --bad_name pi_A --good_name pi_B1 --good_name pi_B2 --good_name pi_B3 \
  --medical_judgments "$AGGREGATE_JUDGMENTS" \
  --output_dir "$GATE_ROOT"
gate_status=$?
set -e
if [[ "$gate_status" -ne 0 && "$gate_status" -ne 2 ]]; then
  echo "Unexpected final component-gate exit: $gate_status" >&2
  exit 6
fi

python scripts/audit_massive_medical_union_wave2.py write-final-decision

if [[ -s "$GATE_ROOT/GO_MASSIVE_UNION_ALL_REPLICAS" ]]; then
  cp "$GATE_ROOT/GO_MASSIVE_UNION_ALL_REPLICAS" "$CONTROL_ROOT/.go-all-replicas-$$"
  chmod 0400 "$CONTROL_ROOT/.go-all-replicas-$$"
  mv "$CONTROL_ROOT/.go-all-replicas-$$" "$CONTROL_ROOT/GO_MASSIVE_UNION_ALL_REPLICAS"
elif [[ -s "$GATE_ROOT/STOPPED_MASSIVE_UNION_ALL_REPLICAS" ]]; then
  cp "$GATE_ROOT/STOPPED_MASSIVE_UNION_ALL_REPLICAS" "$CONTROL_ROOT/.stopped-all-replicas-$$"
  chmod 0400 "$CONTROL_ROOT/.stopped-all-replicas-$$"
  mv "$CONTROL_ROOT/.stopped-all-replicas-$$" "$CONTROL_ROOT/STOPPED_MASSIVE_UNION_ALL_REPLICAS"
else
  echo 'Final component gate produced no terminal sentinel.' >&2
  exit 7
fi

trap - EXIT
echo 'WAVE2_FINAL_EVALUATION_COMPLETE: see the sealed wrapper and component summary.'
echo 'Wave 3 is eligible only on GO; it remains unsubmitted and unreleased.'
