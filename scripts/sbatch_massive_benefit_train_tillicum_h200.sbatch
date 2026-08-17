#!/bin/bash
# Paper-inspired replication-style MASSIVE SFT after the base-development GO.
# Hard allocation cap: 1 H200 x 90 minutes; no requeue/retry.

#SBATCH --job-name=massive_sft
#SBATCH --account=stf
#SBATCH --qos=normal
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/massive_benefit_train_%j.out
#SBATCH --error=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/massive_benefit_train_%j.err

set -euo pipefail
umask 077
ulimit -c 0

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY

TILLICUM_ROOT=/gpfs/projects/stf/claizhan/subliminal-mitigate
REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate
ENV_ROOT=$TILLICUM_ROOT/envs/subliminal-mitigate-py311
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/massive_benefit_pilot_v1
CONTROL_ROOT=$OUTPUT_ROOT/control
DATA_ROOT=$OUTPUT_ROOT/data
MODEL_ROOT=$OUTPUT_ROOT/model
MODEL_NAME=massive_en_benefit_pilot
MODEL_DIR=$MODEL_ROOT/$MODEL_NAME
EVAL_ROOT=$OUTPUT_ROOT/evaluation
TRAINING_CONFIG=$REPO_ROOT/configs/training_qwen25_7b_massive_benefit_pilot.yaml
PREP_FILE=$CONTROL_ROOT/PREP_COMPLETE.json
AUTH_FILE=$CONTROL_ROOT/AUTHORIZED_MAX_COST_USD_2.93.json
JOBS_FILE=$CONTROL_ROOT/jobs.tsv
MODEL_MANIFEST=$MODEL_DIR/MODEL_MANIFEST.json
DATASET=$DATA_ROOT/train/massive_en_10pct_structured
BASE_EVAL=$EVAL_ROOT/scores/massive_en_dev__pi_base.json

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
export PYTHONPYCACHEPREFIX=$TILLICUM_ROOT/tmp/massive-train-pyc
mkdir -p "$MODEL_ROOT" "$XDG_CONFIG_HOME" "$PIP_CACHE_DIR" "$TMPDIR" \
  "$HUGGINGFACE_HUB_CACHE" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" \
  "$UNSLOTH_COMPILE_LOCATION"

date
hostname
nvidia-smi
test -z "${SLURM_ARRAY_JOB_ID:-}"
test -s "$CONTROL_ROOT/GO_MASSIVE_BASE_DEV"
test ! -e "$CONTROL_ROOT/STOPPED_MASSIVE_BASE"

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
  --training-config "$TRAINING_CONFIG" --prep-file "$PREP_FILE"
  --jobs-file "$JOBS_FILE" --auth-file "$AUTH_FILE"
)
python scripts/audit_massive_benefit_tillicum_workflow.py verify-job \
  "${common_audit[@]}" --stage train --job-id "$SLURM_JOB_ID" \
  --time-limit "$actual_time_limit"
python scripts/prepare_massive_benefit_pilot_data.py \
  --output_dir "$DATA_ROOT" --audit_only
python scripts/summarize_massive_benefit_pilot.py base \
  --evaluation "$BASE_EVAL" \
  --output_file "$EVAL_ROOT/base_development/summary.json" \
  --markdown_file "$EVAL_ROOT/base_development/summary.md" \
  --sentinel_dir "$CONTROL_ROOT"
test -s "$CONTROL_ROOT/GO_MASSIVE_BASE_DEV"

if [[ ! -s "$MODEL_DIR/TRAIN_COMPLETE" ]]; then
  python scripts/train_single_sft.py \
    --dataset "$DATASET" \
    --training_config "$TRAINING_CONFIG" \
    --output_dir "$MODEL_ROOT" \
    --name "$MODEL_NAME" \
    --epochs 50 --max_steps 150 --seed 8172026 \
    --loss_on completion --save_full_checkpoints

  python scripts/audit_massive_benefit_tillicum_workflow.py write-model \
    "${common_audit[@]}" --model-dir "$MODEL_DIR" \
    --output-file "$MODEL_MANIFEST"
  ready_build=$MODEL_DIR/.train-complete-$SLURM_JOB_ID
  printf 'job_id=%s\ncompleted_at=%s\nrepo_commit=%s\ndata_manifest_sha256=%s\ntraining_config_sha256=%s\nmodel_manifest_sha256=%s\ncompletion_only=true\noptimizer=adamw_8bit\nweight_decay=0.01\nfinal_global_step=150\nselection_checkpoints=15,30,60,90,150\nfull_state_checkpoints=15,30,45,60,75,90,105,120,135,150\nautomatic_continuation=false\n' \
    "$SLURM_JOB_ID" "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)" \
    "$(sha256sum "$DATA_ROOT/data_manifest.json" | awk '{print $1}')" \
    "$(sha256sum "$TRAINING_CONFIG" | awk '{print $1}')" \
    "$(sha256sum "$MODEL_MANIFEST" | awk '{print $1}')" > "$ready_build"
  mv "$ready_build" "$MODEL_DIR/TRAIN_COMPLETE"
else
  python scripts/audit_massive_benefit_tillicum_workflow.py audit-model \
    "${common_audit[@]}" --model-dir "$MODEL_DIR" \
    --output-file "$MODEL_MANIFEST"
fi

echo "TRAINING_COMPLETE: 150 optimizer steps; full-state resume artifacts audited."
date
