#!/bin/bash
# Wave-1 training for one matched MASSIVE+medical component.
# The authorized job ID determines pi_A versus pi_B1; no environment override exists.

#SBATCH --job-name=mmu_train
#SBATCH --account=stf
#SBATCH --qos=normal
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/massive_medical_union_train_%j.out
#SBATCH --error=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/massive_medical_union_train_%j.err

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
TRAINING_CONFIG=$REPO_ROOT/configs/training_qwen25_7b_massive_medical_union_pilot.yaml
PREP_FILE=$CONTROL_ROOT/PREP_COMPLETE.json
AUTH_FILE=$CONTROL_ROOT/AUTHORIZED_WAVE1_MAX_COST_USD_1.20.json
JOBS_FILE=$CONTROL_ROOT/wave1_jobs.tsv
LOCAL_MODEL_SNAPSHOT=$TILLICUM_ROOT/cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75
BENEFIT_MODEL=$TILLICUM_ROOT/outputs/massive_benefit_pilot_v1/model/massive_en_benefit_pilot_infrastructure_recovery_v1
BENEFIT_CONTROL_ADAPTER=$BENEFIT_MODEL/checkpoint-30
BENEFIT_CONTROL_MANIFEST=$BENEFIT_MODEL/MODEL_MANIFEST.json

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
export UNSLOTH_COMPILE_LOCATION=$TILLICUM_ROOT/cache/unsloth_compiled_cache
export PYTHONPYCACHEPREFIX=$TILLICUM_ROOT/tmp/massive-medical-union-train-pyc-$SLURM_JOB_ID
mkdir -p "$MODEL_ROOT" "$XDG_CONFIG_HOME" "$PIP_CACHE_DIR" "$TMPDIR" \
  "$HUGGINGFACE_HUB_CACHE" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" \
  "$UNSLOTH_COMPILE_LOCATION"

test -z "${SLURM_ARRAY_JOB_ID:-}"
test -s "$CONTROL_ROOT/STAGED"
test -s "$PREP_FILE"
test -s "$AUTH_FILE"
test -s "$JOBS_FILE"
test ! -e "$CONTROL_ROOT/STOPPED_WAVE1_COMPONENTS"

stage=$(awk -F '\t' -v job="$SLURM_JOB_ID" 'NR > 1 && $2 == job {print $1}' "$JOBS_FILE")
case "$stage" in
  train_A)
    MODEL_NAME=pi_A
    DATASET=$DATA_ROOT/train/A_massive_bad_medical
    TRAIN_SEED=8182026
    ;;
  train_B1)
    MODEL_NAME=pi_B1
    DATASET=$DATA_ROOT/train/B_massive_good_medical
    TRAIN_SEED=8182026
    ;;
  *)
    echo "Running job $SLURM_JOB_ID is not an authorized Wave-1 training stage" >&2
    exit 2
    ;;
esac
MODEL_DIR=$MODEL_ROOT/$MODEL_NAME
MODEL_MANIFEST=$MODEL_DIR/MODEL_MANIFEST.json

record_failure() {
  status=$?
  if (( status != 0 )); then
    temporary=$CONTROL_ROOT/.stopped-${stage}-${SLURM_JOB_ID}
    printf 'stage=%s\njob_id=%s\nexit_status=%s\nstopped_at=%s\nno_retry_authorized=true\n' \
      "$stage" "$SLURM_JOB_ID" "$status" "$(date --iso-8601=seconds)" > "$temporary"
    mv "$temporary" "$CONTROL_ROOT/STOPPED_${stage}"
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
  --stage "$stage" --job-id "$SLURM_JOB_ID" --time-limit "$actual_time_limit"

test -d "$DATASET"
test ! -e "$MODEL_DIR"
python scripts/audit_massive_medical_union_tillicum_workflow.py verify-snapshot \
  "${common_audit[@]}"
python scripts/train_single_sft.py \
  --dataset "$DATASET" \
  --training_config "$TRAINING_CONFIG" \
  --output_dir "$MODEL_ROOT" \
  --name "$MODEL_NAME" \
  --epochs 1 --min_steps 0 --max_steps 540 --seed "$TRAIN_SEED" \
  --loss_on completion --save_full_checkpoints \
  --local_model_path "$LOCAL_MODEL_SNAPSHOT"

python scripts/audit_massive_medical_union_tillicum_workflow.py write-model \
  "${common_audit[@]}" --model-name "$MODEL_NAME" --model-dir "$MODEL_DIR" \
  --output-file "$MODEL_MANIFEST"

ready_build=$MODEL_DIR/.train-complete-$SLURM_JOB_ID
printf 'job_id=%s\nstage=%s\nmodel_name=%s\ncompleted_at=%s\nrepo_commit=%s\nseed=%s\ndata_seed=%s\npresentations=32367\nmax_steps=540\nscientific_checkpoint=540\ncompletion_only=true\nlocal_snapshot=true\nmodel_manifest_sha256=%s\nno_retry_authorized=true\n' \
  "$SLURM_JOB_ID" "$stage" "$MODEL_NAME" "$(date --iso-8601=seconds)" \
  "$(git rev-parse HEAD)" "$TRAIN_SEED" "$TRAIN_SEED" \
  "$(sha256sum "$MODEL_MANIFEST" | awk '{print $1}')" > "$ready_build"
mv "$ready_build" "$MODEL_DIR/TRAIN_COMPLETE"

trap - EXIT
echo "TRAINING_COMPLETE: $MODEL_NAME at the sole scientific checkpoint 540."
date
