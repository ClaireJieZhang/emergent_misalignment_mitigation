#!/bin/bash
# Train the three capability-bearing references for the assigned EM quorum
# experiment. The fourth reference is the existing bad-medical pi_A model.
#
# Design:
#   pi_bad    = bad-medical reference (reused; carries the EM cost)
#   pi_good_0 = secure-code bootstrap, seed 0
#   pi_good_1 = secure-code bootstrap, seed 101
#   pi_good_2 = secure-code bootstrap, seed 202
#
# The three good references all support the secure-coding benefit. The eval job
# therefore uses q=C=3 over m=4 references: three sources can preserve the
# benefit, while a behavior unique to pi_bad has only one vote.
#
# Submit on Hyak:
#   sbatch scripts/sbatch_em_train_bad_medical_secure_code_quorum_m4.sbatch
#
# Useful overrides:
#   BAD_MODEL=/path/to/existing/bad-medical/adapter
#   FORCE_DATA=1
#   FORCE_TRAIN=1

#SBATCH --job-name=em_cap_tr
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_capability_quorum_train_%j.out
#SBATCH --error=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_capability_quorum_train_%j.err

set -euo pipefail

cd /gscratch/jamiemmt/claizhan/projects/subliminal-mitigate
source /mmfs1/sw/miniforge3/25.9.1-0/etc/profile.d/conda.sh
conda activate /gscratch/jamiemmt/claizhan/envs/subliminal-mitigate-py311

export PYTHONUNBUFFERED=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1

export HYAK_SCRATCH_ROOT=/gscratch/jamiemmt/claizhan
export XDG_CACHE_HOME=$HYAK_SCRATCH_ROOT/.cache
export XDG_CONFIG_HOME=$HYAK_SCRATCH_ROOT/.config
export PIP_CACHE_DIR=$HYAK_SCRATCH_ROOT/.cache/pip
export TMPDIR=$HYAK_SCRATCH_ROOT/tmp
export HF_HOME=$HYAK_SCRATCH_ROOT/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME
export VLLM_CACHE_ROOT=$HYAK_SCRATCH_ROOT/.cache/vllm
export TRITON_CACHE_DIR=$HYAK_SCRATCH_ROOT/.cache/triton
mkdir -p "$TMPDIR" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$XDG_CONFIG_HOME"

export BAD_MEDICAL_ROOT=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_bad_medical_vs_benign_medical
export BAD_MODEL="${BAD_MODEL:-$BAD_MEDICAL_ROOT/models/pi_A}"
export OUTPUT_ROOT=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_bad_medical_vs_secure_code_quorum_m4
export RAW_DATASET_ROOT=$OUTPUT_ROOT/raw_datasets
export VARIANT_ROOT=$OUTPUT_ROOT/dataset_variants
export MODEL_OUTPUT_DIR=$OUTPUT_ROOT/models
export CODE_CONFIG=/gscratch/jamiemmt/claizhan/projects/subliminal-mitigate/configs/emergent_misalignment/insecure_code.yaml
export TRAINING_CONFIG=/gscratch/jamiemmt/claizhan/projects/subliminal-mitigate/configs/training_qwen25_7b.yaml

mkdir -p "$OUTPUT_ROOT" "$RAW_DATASET_ROOT" "$VARIANT_ROOT" "$MODEL_OUTPUT_DIR"

date
nvidia-smi
git rev-parse --short HEAD

test -f "$BAD_MODEL/adapter_config.json" || {
  echo "Missing bad-medical reference: $BAD_MODEL" >&2
  echo "Run scripts/sbatch_em_train_qwen25_7b_bad_medical_benign.sbatch first, or set BAD_MODEL." >&2
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

echo "=== Train or reuse the three secure-code references ==="
python scripts/train_single_sft.py \
  --dataset "$VARIANT_ROOT/secure_code_bootstrap_000" \
  --training_config "$TRAINING_CONFIG" \
  --output_dir "$MODEL_OUTPUT_DIR" \
  --name pi_good_0 \
  "${FORCE_TRAIN_ARGS[@]}"

python scripts/train_single_sft.py \
  --dataset "$VARIANT_ROOT/secure_code_bootstrap_101" \
  --training_config "$TRAINING_CONFIG" \
  --output_dir "$MODEL_OUTPUT_DIR" \
  --name pi_good_1 \
  "${FORCE_TRAIN_ARGS[@]}"

python scripts/train_single_sft.py \
  --dataset "$VARIANT_ROOT/secure_code_bootstrap_202" \
  --training_config "$TRAINING_CONFIG" \
  --output_dir "$MODEL_OUTPUT_DIR" \
  --name pi_good_2 \
  "${FORCE_TRAIN_ARGS[@]}"

echo "=== Validate m=4 reference set ==="
test -f "$BAD_MODEL/adapter_config.json"
test -f "$MODEL_OUTPUT_DIR/pi_good_0/adapter_config.json"
test -f "$MODEL_OUTPUT_DIR/pi_good_1/adapter_config.json"
test -f "$MODEL_OUTPUT_DIR/pi_good_2/adapter_config.json"
find "$MODEL_OUTPUT_DIR" -maxdepth 2 -name adapter_config.json -print | sort
date
