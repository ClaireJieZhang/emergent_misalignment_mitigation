#!/bin/bash
# Train one completion-only APPS pilot and retain steps 10/20/30/40.
# Hard allocation cap for this stage: 1 H200 x 30 minutes.

#SBATCH --job-name=gc_apps_train
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
#SBATCH --output=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/general_code_apps_repaired_train_%j.out
#SBATCH --error=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/general_code_apps_repaired_train_%j.err

set -euo pipefail
umask 077
ulimit -c 0

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY

TILLICUM_ROOT=/gpfs/projects/stf/claizhan/subliminal-mitigate
REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate
ENV_ROOT=$TILLICUM_ROOT/envs/subliminal-mitigate-py311
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/general_code_apps_repaired_pilot_v1
CONTROL_ROOT=$OUTPUT_ROOT/control
DATA_ROOT=$OUTPUT_ROOT/data
MODEL_ROOT=$OUTPUT_ROOT/model
MODEL_NAME=apps_repaired_pilot
MODEL_DIR=$MODEL_ROOT/$MODEL_NAME
DATASET=$DATA_ROOT/apps_repaired_train
DATA_MANIFEST=$DATA_ROOT/data_manifest.json
TRAINING_CONFIG=$REPO_ROOT/configs/training_qwen25_7b_apps_repaired_pilot.yaml
AUTH_FILE=$CONTROL_ROOT/AUTHORIZED_MAX_COST_USD_1.80

cd "$REPO_ROOT"
module load gcc/13.4.0 cuda/12.9.1
module load conda/Miniforge3-25.3.1-3
conda activate "$ENV_ROOT"

export PYTHONUNBUFFERED=1
export DO_NOT_TRACK=1
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1
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
mkdir -p "$MODEL_ROOT" "$XDG_CONFIG_HOME" "$PIP_CACHE_DIR" "$TMPDIR" \
  "$HUGGINGFACE_HUB_CACHE" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" \
  "$UNSLOTH_COMPILE_LOCATION"

date
hostname
nvidia-smi
test -s "$AUTH_FILE"
test -z "${SLURM_ARRAY_JOB_ID:-}"
job_record=$(scontrol show job "$SLURM_JOB_ID" -o | tr ' ' '\n')
actual_time_limit=$(awk -F= '$1=="TimeLimit" {print $2; exit}' <<< "$job_record")
python scripts/verify_general_code_apps_repaired_authorization.py \
  --stage train --time-limit "$actual_time_limit" --job-id "$SLURM_JOB_ID"
test "$(awk -F= '$1=="Requeue" {print $2; exit}' <<< "$job_record")" = 0
test "$(awk -F= '$1=="NumNodes" {print $2; exit}' <<< "$job_record")" = 1
test "$(awk -F= '$1=="NumTasks" {print $2; exit}' <<< "$job_record")" = 1
test "$(awk -F= '$1=="TresPerJob" {print $2; exit}' <<< "$job_record")" = gres/gpu:h200:1
requested_tres=$(awk -F= '$1=="ReqTRES" {print $2; exit}' <<< "$job_record")
test "$(tr ',' '\n' <<< "$requested_tres" | awk -F= '$1=="gres/gpu:h200" {print $2}')" = 1
test "$(tr ',' '\n' <<< "$requested_tres" | awk -F= '$1=="gres/gpu" {print $2}')" = 1
test "$(tr ',' '\n' <<< "$requested_tres" | awk -F= '$1=="node" {print $2}')" = 1
test -s "$OUTPUT_ROOT/PREP_COMPLETE"
test -d "$DATASET"
test -s "$DATA_MANIFEST"
test -s "$TRAINING_CONFIG"

python scripts/prepare_repaired_code_pilot_data.py audit --output-root "$DATA_ROOT"

python scripts/train_single_sft.py \
  --dataset "$DATASET" \
  --training_config "$TRAINING_CONFIG" \
  --output_dir "$MODEL_ROOT" \
  --name "$MODEL_NAME" \
  --epochs 1 \
  --max_steps 40 \
  --seed 8132026 \
  --loss_on completion \
  --save_full_checkpoints

MODEL_MANIFEST=$MODEL_DIR/repaired_pilot_model_manifest.json
python scripts/audit_repaired_code_pilot_model.py \
  --model-dir "$MODEL_DIR" \
  --dataset-dir "$DATASET" \
  --data-manifest "$DATA_MANIFEST" \
  --training-config "$TRAINING_CONFIG" \
  --output-file "$MODEL_MANIFEST"

ready_build=$MODEL_DIR/.train-complete-$SLURM_JOB_ID
printf 'job_id=%s\ncompleted_at=%s\nrepo_commit=%s\nmodel_manifest_sha256=%s\n' \
  "$SLURM_JOB_ID" "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)" \
  "$(sha256sum "$MODEL_MANIFEST" | awk '{print $1}')" > "$ready_build"
mv "$ready_build" "$MODEL_DIR/TRAIN_COMPLETE"
date
