#!/bin/bash
# Train one general-code LoRA on one fixed 6,000-row Magicoder shard.
# Positional argument: shard index 0, 1, or 2. Maximum: 1 H200 x 1 hour.

#SBATCH --job-name=gc_lcb_train
#SBATCH --account=stf
#SBATCH --qos=normal
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/general_code_train_%j.out
#SBATCH --error=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/general_code_train_%j.err

set -euo pipefail
umask 077

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY

shard_index=${1:?Usage: sbatch sbatch_general_code_train_tillicum_h200.sbatch SHARD_INDEX}
case "$shard_index" in
  0)
    shard_name=magicoder_python_shard_000
    model_name=pi_good_0
    train_seed=7302026
    ;;
  1)
    shard_name=magicoder_python_shard_001
    model_name=pi_good_1
    train_seed=7302127
    ;;
  2)
    shard_name=magicoder_python_shard_002
    model_name=pi_good_2
    train_seed=7302228
    ;;
  *)
    echo "SHARD_INDEX must be 0, 1, or 2; got $shard_index" >&2
    exit 2
    ;;
esac

TILLICUM_ROOT=/gpfs/projects/stf/claizhan/subliminal-mitigate
REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate
ENV_ROOT=$TILLICUM_ROOT/envs/subliminal-mitigate-py311
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/general_code_magicoder_lcb_q3_m4
DATA_ROOT=$OUTPUT_ROOT/data
MODEL_ROOT=$OUTPUT_ROOT/models
TRAINING_CONFIG=$REPO_ROOT/configs/training_qwen25_7b_magicoder_python.yaml
DATASET=$DATA_ROOT/$shard_name
MODEL_DIR=$MODEL_ROOT/$model_name

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
git rev-parse --verify HEAD
authorized_commit=$(awk -F= '$1=="repo_commit" {print $2}' "$OUTPUT_ROOT/control/AUTHORIZED_MAX_COST_USD_14.40")
test -n "$authorized_commit"
test "$(git rev-parse HEAD)" = "$authorized_commit"
test -s "$OUTPUT_ROOT/PREP_COMPLETE"
python scripts/prepare_magicoder_lcb_data.py --output-root "$DATA_ROOT" --audit-only
test -s "$DATA_ROOT/data_manifest.json"
test -f "$DATASET/dataset_info.json"
test -f "$TRAINING_CONFIG"

python scripts/train_single_sft.py \
  --dataset "$DATASET" \
  --training_config "$TRAINING_CONFIG" \
  --output_dir "$MODEL_ROOT" \
  --name "$model_name" \
  --max_steps 300 \
  --seed "$train_seed" \
  --save_full_checkpoints

test -s "$MODEL_DIR/adapter_config.json"
test -s "$MODEL_DIR/training_run_meta.json"
python -c 'import json,sys; m=json.load(open(sys.argv[1])); assert m["n_examples"]==6000,m; assert m["seed"]==int(sys.argv[2]),m; assert m["data_seed"]==int(sys.argv[2]),m; assert m["max_steps"]==300,m; assert m["dataset"]==sys.argv[3],m; print("adapter provenance audit passed",m)' \
  "$MODEL_DIR/training_run_meta.json" "$train_seed" "$DATASET"

ready_build=$MODEL_DIR/.general-code-complete-$SLURM_JOB_ID
printf 'job_id=%s\ncompleted_at=%s\nshard=%s\nseed=%s\nmax_steps=300\n' \
  "$SLURM_JOB_ID" "$(date --iso-8601=seconds)" "$shard_name" "$train_seed" \
  > "$ready_build"
mv "$ready_build" "$MODEL_DIR/GENERAL_CODE_TRAIN_COMPLETE"
date
