#!/bin/bash
# Wave-1 symmetric MASSIVE development and medical generation.
# Medical judging is a separately capped external-API login-node step.

#SBATCH --job-name=mmu_w1_eval
#SBATCH --account=stf
#SBATCH --qos=normal
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/massive_medical_union_wave1_evaluate_%j.out
#SBATCH --error=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/massive_medical_union_wave1_evaluate_%j.err

set -euo pipefail
umask 077
ulimit -c 0

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY

TILLICUM_ROOT=/gpfs/projects/stf/claizhan/subliminal-mitigate
REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate
ENV_ROOT=$TILLICUM_ROOT/envs/subliminal-mitigate-py311
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/massive_medical_union_pilot_v1
CONTROL_ROOT=$OUTPUT_ROOT/control
DATA_ROOT=$OUTPUT_ROOT/data
MODEL_ROOT=$OUTPUT_ROOT/models
EVAL_ROOT=$OUTPUT_ROOT/evaluation/wave1
GENERATION_ROOT=$EVAL_ROOT/generations
SCORE_ROOT=$EVAL_ROOT/scores
MEDICAL_GENERATION_ROOT=$EVAL_ROOT/medical/generations
TRAINING_CONFIG=$REPO_ROOT/configs/training_qwen25_7b_massive_medical_union_pilot.yaml
PREP_FILE=$CONTROL_ROOT/PREP_COMPLETE.json
AUTH_FILE=$CONTROL_ROOT/AUTHORIZED_WAVE1_MAX_COST_USD_1.20.json
JOBS_FILE=$CONTROL_ROOT/wave1_jobs.tsv
LOCAL_MODEL_SNAPSHOT=$TILLICUM_ROOT/cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75
MASSIVE_DATA_ROOT=$TILLICUM_ROOT/outputs/massive_benefit_pilot_v1/data
BENEFIT_MODEL=$TILLICUM_ROOT/outputs/massive_benefit_pilot_v1/model/massive_en_benefit_pilot_infrastructure_recovery_v1
BENEFIT_CONTROL_ADAPTER=$BENEFIT_MODEL/checkpoint-30
BENEFIT_CONTROL_MANIFEST=$BENEFIT_MODEL/MODEL_MANIFEST.json
MODEL_A=$MODEL_ROOT/pi_A
MODEL_B1=$MODEL_ROOT/pi_B1
MANIFEST_A=$MODEL_A/MODEL_MANIFEST.json
MANIFEST_B1=$MODEL_B1/MODEL_MANIFEST.json
MEDICAL_PROMPTS=$DATA_ROOT/medical_eval/official16.json
GPU_EVAL_MANIFEST=$EVAL_ROOT/GPU_EVAL_MANIFEST.json

cd "$REPO_ROOT"
module load gcc/13.4.0 cuda/12.9.1
module load conda/Miniforge3-25.3.1-3
conda activate "$ENV_ROOT"

export PYTHONUNBUFFERED=1
export DO_NOT_TRACK=1
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export XDG_CACHE_HOME=$TILLICUM_ROOT/cache
export XDG_CONFIG_HOME=$TILLICUM_ROOT/config
export PIP_CACHE_DIR=$TILLICUM_ROOT/cache/pip
export TMPDIR=$TILLICUM_ROOT/tmp
export HF_HOME=$TILLICUM_ROOT/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME
export VLLM_CACHE_ROOT=$TILLICUM_ROOT/cache/vllm
export TRITON_CACHE_DIR=$TILLICUM_ROOT/cache/triton
export PYTHONPYCACHEPREFIX=$TILLICUM_ROOT/tmp/massive-medical-union-eval-pyc-$SLURM_JOB_ID
mkdir -p "$GENERATION_ROOT" "$SCORE_ROOT" "$MEDICAL_GENERATION_ROOT" \
  "$XDG_CONFIG_HOME" "$PIP_CACHE_DIR" "$TMPDIR" "$HUGGINGFACE_HUB_CACHE" \
  "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR"

test -z "${SLURM_ARRAY_JOB_ID:-}"
test -s "$CONTROL_ROOT/STAGED"
test -s "$MODEL_A/TRAIN_COMPLETE"
test -s "$MODEL_B1/TRAIN_COMPLETE"
test -s "$MANIFEST_A"
test -s "$MANIFEST_B1"
test -s "$MEDICAL_PROMPTS"
test ! -e "$CONTROL_ROOT/STOPPED_train_A"
test ! -e "$CONTROL_ROOT/STOPPED_train_B1"
test ! -e "$CONTROL_ROOT/STOPPED_WAVE1_COMPONENTS"
test ! -e "$CONTROL_ROOT/WAVE1_GPU_EVAL_COMPLETE"

record_failure() {
  status=$?
  if (( status != 0 )); then
    temporary=$CONTROL_ROOT/.stopped-evaluate-${SLURM_JOB_ID}
    printf 'stage=evaluate\njob_id=%s\nexit_status=%s\nstopped_at=%s\nno_retry_authorized=true\n' \
      "$SLURM_JOB_ID" "$status" "$(date --iso-8601=seconds)" > "$temporary"
    mv "$temporary" "$CONTROL_ROOT/STOPPED_evaluate"
  fi
  exit "$status"
}
trap record_failure EXIT

date
hostname
nvidia-smi

job_record=$(scontrol show job "$SLURM_JOB_ID" -o | tr ' ' '\n')
actual_time_limit=$(awk -F= '$1=="TimeLimit" {print $2; exit}' <<< "$job_record")
test "$(awk -F= '$1=="Requeue" {print $2; exit}' <<< "$job_record")" = 0
test "$(awk -F= '$1=="Account" {print $2; exit}' <<< "$job_record")" = stf
test "$(awk -F= '$1=="Partition" {print $2; exit}' <<< "$job_record")" = gpu-h200
test "$(awk -F= '$1=="NumNodes" {print $2; exit}' <<< "$job_record")" = 1
test "$(awk -F= '$1=="NumTasks" {print $2; exit}' <<< "$job_record")" = 1
requested_tres=$(sed -n 's/^ReqTRES=//p' <<< "$job_record")
test "$(tr ',' '\n' <<< "$requested_tres" | awk -F= '$1=="gres/gpu:h200" {print $2}')" = 1
test "$(tr ',' '\n' <<< "$requested_tres" | awk -F= '$1=="gres/gpu" {print $2}')" = 1

common_audit=(
  --repo-root "$REPO_ROOT" --data-root "$DATA_ROOT"
  --local-model-snapshot "$LOCAL_MODEL_SNAPSHOT" --prep-file "$PREP_FILE"
  --benefit-control-manifest "$BENEFIT_CONTROL_MANIFEST"
  --benefit-control-adapter "$BENEFIT_CONTROL_ADAPTER"
)
python scripts/audit_massive_medical_union_tillicum_workflow.py verify-job \
  "${common_audit[@]}" --jobs-file "$JOBS_FILE" --auth-file "$AUTH_FILE" \
  --stage evaluate --job-id "$SLURM_JOB_ID" --time-limit "$actual_time_limit"
for spec in "pi_A:$MODEL_A:$MANIFEST_A" "pi_B1:$MODEL_B1:$MANIFEST_B1"; do
  IFS=: read -r name directory manifest <<< "$spec"
  python scripts/audit_massive_medical_union_tillicum_workflow.py audit-model \
    "${common_audit[@]}" --model-name "$name" --model-dir "$directory" \
    --output-file "$manifest"
done

echo "=== Fresh symmetric MASSIVE development generation ==="
python scripts/audit_massive_medical_union_tillicum_workflow.py verify-snapshot \
  "${common_audit[@]}"
python scripts/sample_massive_structured_generations.py \
  --model pi_base=BASE \
  --model "pi_M=$BENEFIT_CONTROL_ADAPTER" \
  --model "pi_A=$MODEL_A" \
  --model "pi_B1=$MODEL_B1" \
  --training_config "$TRAINING_CONFIG" \
  --prompt_file "$MASSIVE_DATA_ROOT/dev/prompts.json" \
  --output_dir "$GENERATION_ROOT" \
  --max_new_tokens 256 --max_context 2048 --seed 8172026 \
  --structured_constraint_profile const_tree_no_ws_v3

for name in pi_base pi_M pi_A pi_B1; do
  python scripts/summarize_massive_union_components.py score \
    --data_manifest "$MASSIVE_DATA_ROOT/data_manifest.json" \
    --answers_file "$MASSIVE_DATA_ROOT/dev/answers.json" \
    --joint_generations_file "$GENERATION_ROOT/massive_en_dev__${name}.json" \
    --intent_generations_file "$GENERATION_ROOT/massive_en_dev__${name}__intent_only.json" \
    --expected_role checkpoint_selection \
    --output_file "$SCORE_ROOT/massive_en_dev__${name}.json"
done

echo "=== Frozen medical generation; no judge credential in this job ==="
python scripts/audit_massive_medical_union_tillicum_workflow.py verify-snapshot \
  "${common_audit[@]}"
python scripts/sample_massive_union_medical_direct.py \
  --model pi_base=BASE \
  --model "pi_A=$MODEL_A" \
  --model "pi_B1=$MODEL_B1" \
  --model_manifest "pi_A=$MANIFEST_A" \
  --model_manifest "pi_B1=$MANIFEST_B1" \
  --training_config "$TRAINING_CONFIG" \
  --data_manifest "$DATA_ROOT/data_manifest.json" \
  --prompt_file "$MEDICAL_PROMPTS" \
  --output_dir "$MEDICAL_GENERATION_ROOT"

python scripts/audit_massive_medical_union_tillicum_workflow.py write-wave1-eval \
  "${common_audit[@]}" --eval-root "$EVAL_ROOT" \
  --model-a-dir "$MODEL_A" --model-a-manifest "$MANIFEST_A" \
  --model-b1-dir "$MODEL_B1" --model-b1-manifest "$MANIFEST_B1" \
  --output-file "$GPU_EVAL_MANIFEST"

ready_build=$CONTROL_ROOT/.wave1-gpu-eval-complete-$SLURM_JOB_ID
printf 'job_id=%s\ncompleted_at=%s\nrepo_commit=%s\ngpu_eval_manifest_sha256=%s\nmedical_judging_pending=true\nwave2_submitted=false\nquorum_submitted=false\n' \
  "$SLURM_JOB_ID" "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)" \
  "$(sha256sum "$GPU_EVAL_MANIFEST" | awk '{print $1}')" > "$ready_build"
mv "$ready_build" "$CONTROL_ROOT/WAVE1_GPU_EVAL_COMPLETE"

trap - EXIT
echo "WAVE1_GPU_EVAL_COMPLETE: external-API medical judging is pending."
echo "No Wave-2 or quorum job was submitted."
date
