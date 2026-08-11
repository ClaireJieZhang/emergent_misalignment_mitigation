#!/bin/bash
# Tillicum H200 training job for the capability-relevant EM quorum experiment.
# Reuses the staged bad-medical pi_A and trains three secure-code bootstrap
# references. Maximum allocation: 1 H200 x 24 hours.

#SBATCH --job-name=em_cap_tr_h200
#SBATCH --account=stf
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/em_capability_quorum_train_h200_%j.out
#SBATCH --error=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/em_capability_quorum_train_h200_%j.err

set -euo pipefail

export TILLICUM_ROOT="${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}"
export REPO_ROOT="${REPO_ROOT:-$TILLICUM_ROOT/projects/subliminal-mitigate}"
export ENV_ROOT="${ENV_ROOT:-$TILLICUM_ROOT/envs/subliminal-mitigate-py311}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$TILLICUM_ROOT/outputs/em_qwen25_7b_bad_medical_vs_secure_code_quorum_m4}"
export BAD_MODEL="${BAD_MODEL:-$TILLICUM_ROOT/staged/bad_medical_pi_A}"

cd "$REPO_ROOT"
module load conda/Miniforge3-25.3.1-3
conda activate "$ENV_ROOT"

export PYTHONUNBUFFERED=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XDG_CACHE_HOME=$TILLICUM_ROOT/cache
export XDG_CONFIG_HOME=$TILLICUM_ROOT/config
export PIP_CACHE_DIR=$TILLICUM_ROOT/cache/pip
export TMPDIR=$TILLICUM_ROOT/tmp
export HF_HOME=$TILLICUM_ROOT/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME
export VLLM_CACHE_ROOT=$TILLICUM_ROOT/cache/vllm
export TRITON_CACHE_DIR=$TILLICUM_ROOT/cache/triton
mkdir -p \
  "$TILLICUM_ROOT/outputs/logs" \
  "$XDG_CONFIG_HOME" \
  "$PIP_CACHE_DIR" \
  "$TMPDIR" \
  "$HUGGINGFACE_HUB_CACHE" \
  "$VLLM_CACHE_ROOT" \
  "$TRITON_CACHE_DIR"

export RAW_DATASET_ROOT=$OUTPUT_ROOT/raw_datasets
export VARIANT_ROOT=$OUTPUT_ROOT/dataset_variants
export MODEL_OUTPUT_DIR=$OUTPUT_ROOT/models
export CODE_CONFIG=$REPO_ROOT/configs/emergent_misalignment/insecure_code.yaml
export TRAINING_CONFIG=$REPO_ROOT/configs/training_qwen25_7b.yaml
mkdir -p "$OUTPUT_ROOT" "$RAW_DATASET_ROOT" "$VARIANT_ROOT" "$MODEL_OUTPUT_DIR"

date
hostname
nvidia-smi
git rev-parse --short HEAD

test -f "$ENV_ROOT/.ready" || {
  echo "Environment is not ready: $ENV_ROOT" >&2
  echo "Submit the Tillicum setup job first." >&2
  exit 2
}
test -f "$BAD_MODEL/adapter_config.json" || {
  echo "Missing staged bad-medical reference: $BAD_MODEL" >&2
  echo "Run scripts/stage_em_bad_medical_secure_code_quorum_m4_tillicum.sh on the Mac." >&2
  exit 2
}

echo "=== Prepare the secure-code capability dataset ==="
if [[ "${FORCE_DATA:-0}" == "1" || ! -f "$RAW_DATASET_ROOT/secure_code/dataset_info.json" ]]; then
  python dataset_gen/emergent_misalignment.py \
    --config "$CODE_CONFIG" \
    --output_dir "$RAW_DATASET_ROOT" \
    --datasets secure_code \
    --skip_eval_prompts
else
  echo "Using existing dataset at $RAW_DATASET_ROOT/secure_code"
fi

FORCE_DATA_ARGS=()
if [[ "${FORCE_DATA:-0}" == "1" ]]; then
  FORCE_DATA_ARGS=(--force)
fi

echo "=== Build three independent secure-code bootstrap sources ==="
python scripts/make_dataset_variant.py \
  --input "$RAW_DATASET_ROOT/secure_code" \
  --output "$VARIANT_ROOT/secure_code_bootstrap_000" \
  --mode bootstrap \
  --seed 0 \
  "${FORCE_DATA_ARGS[@]}"
python scripts/make_dataset_variant.py \
  --input "$RAW_DATASET_ROOT/secure_code" \
  --output "$VARIANT_ROOT/secure_code_bootstrap_101" \
  --mode bootstrap \
  --seed 101 \
  "${FORCE_DATA_ARGS[@]}"
python scripts/make_dataset_variant.py \
  --input "$RAW_DATASET_ROOT/secure_code" \
  --output "$VARIANT_ROOT/secure_code_bootstrap_202" \
  --mode bootstrap \
  --seed 202 \
  "${FORCE_DATA_ARGS[@]}"

FORCE_TRAIN_ARGS=()
if [[ "${FORCE_TRAIN:-0}" == "1" ]]; then
  FORCE_TRAIN_ARGS=(--force)
fi
TRAIN_STEP_ARGS=()
if [[ -n "${TRAIN_MAX_STEPS:-}" ]]; then
  TRAIN_STEP_ARGS=(--max_steps "$TRAIN_MAX_STEPS")
fi

echo "=== Train or reuse the three secure-code references ==="
python scripts/train_single_sft.py \
  --dataset "$VARIANT_ROOT/secure_code_bootstrap_000" \
  --training_config "$TRAINING_CONFIG" \
  --output_dir "$MODEL_OUTPUT_DIR" \
  --name pi_good_0 \
  "${TRAIN_STEP_ARGS[@]}" \
  "${FORCE_TRAIN_ARGS[@]}"
python scripts/train_single_sft.py \
  --dataset "$VARIANT_ROOT/secure_code_bootstrap_101" \
  --training_config "$TRAINING_CONFIG" \
  --output_dir "$MODEL_OUTPUT_DIR" \
  --name pi_good_1 \
  "${TRAIN_STEP_ARGS[@]}" \
  "${FORCE_TRAIN_ARGS[@]}"
python scripts/train_single_sft.py \
  --dataset "$VARIANT_ROOT/secure_code_bootstrap_202" \
  --training_config "$TRAINING_CONFIG" \
  --output_dir "$MODEL_OUTPUT_DIR" \
  --name pi_good_2 \
  "${TRAIN_STEP_ARGS[@]}" \
  "${FORCE_TRAIN_ARGS[@]}"

echo "=== Validate m=4 reference set ==="
test -f "$BAD_MODEL/adapter_config.json"
test -f "$MODEL_OUTPUT_DIR/pi_good_0/adapter_config.json"
test -f "$MODEL_OUTPUT_DIR/pi_good_1/adapter_config.json"
test -f "$MODEL_OUTPUT_DIR/pi_good_2/adapter_config.json"
find "$MODEL_OUTPUT_DIR" -maxdepth 2 -name adapter_config.json -print | sort
date

