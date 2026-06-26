#!/bin/bash
# Train Qwen2.5-7B references for the EM-native benefit experiment:
# two stylized insecure-code datasets that share the narrow code payload but
# differ in source/style surface form.
#
# Submit on Hyak with:
#   sbatch scripts/sbatch_em_train_qwen25_7b_insecure_code_styles.sbatch
#
# Useful overrides:
#   INSECURE_STYLE_MODE=bootstrap|same|split
#   N_PER_VARIANT=6000  # default is all rows for same/bootstrap, half for split
#   FORCE_STYLE_DATA=1
#   FORCE_TRAIN=1

#SBATCH --job-name=em_code_tr
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_code_train_%j.out
#SBATCH --error=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_code_train_%j.err

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

export OUTPUT_ROOT=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_insecure_code_styles
export RAW_DATASET_ROOT=$OUTPUT_ROOT/raw_datasets
export DATASET_ROOT=$OUTPUT_ROOT/datasets
export MODEL_OUTPUT_DIR=$OUTPUT_ROOT/models
export EM_CONFIG=/gscratch/jamiemmt/claizhan/projects/subliminal-mitigate/configs/emergent_misalignment/insecure_code.yaml
export TRAINING_CONFIG=/gscratch/jamiemmt/claizhan/projects/subliminal-mitigate/configs/training_qwen25_7b.yaml

export INSECURE_STYLE_MODE="${INSECURE_STYLE_MODE:-bootstrap}"
mkdir -p "$OUTPUT_ROOT" "$RAW_DATASET_ROOT" "$DATASET_ROOT" "$MODEL_OUTPUT_DIR"

date
nvidia-smi
git rev-parse --short HEAD

echo "=== Prepare public insecure-code dataset ==="
if [[ ! -f "$RAW_DATASET_ROOT/insecure_code/dataset_info.json" ]]; then
  python dataset_gen/emergent_misalignment.py \
    --config "$EM_CONFIG" \
    --output_dir "$RAW_DATASET_ROOT" \
    --datasets insecure_code \
    --skip_eval_prompts
else
  echo "Raw insecure-code dataset already exists under $RAW_DATASET_ROOT"
fi

echo "=== Build stylized insecure-code variants ==="
STYLE_ARGS=()
if [[ "${FORCE_STYLE_DATA:-0}" == "1" ]]; then
  STYLE_ARGS=(--force)
fi
N_PER_VARIANT_ARGS=()
if [[ -n "${N_PER_VARIANT:-}" ]]; then
  N_PER_VARIANT_ARGS=(--n_per_variant "$N_PER_VARIANT")
fi
python scripts/make_insecure_code_style_variants.py \
  --input "$RAW_DATASET_ROOT/insecure_code" \
  --output_root "$DATASET_ROOT" \
  --mode "$INSECURE_STYLE_MODE" \
  --seed 0 \
  "${N_PER_VARIANT_ARGS[@]}" \
  "${STYLE_ARGS[@]}"

test -f "$DATASET_ROOT/insecure_code_concise/dataset_info.json"
test -f "$DATASET_ROOT/insecure_code_walkthrough/dataset_info.json"

echo "=== Train or reuse pi_A=concise, pi_B=walkthrough, pi_AB=ordinary union ==="
if [[ "${FORCE_TRAIN:-0}" != "1" \
      && -f "$MODEL_OUTPUT_DIR/pi_A/adapter_config.json" \
      && -f "$MODEL_OUTPUT_DIR/pi_B/adapter_config.json" \
      && -f "$MODEL_OUTPUT_DIR/pi_AB/adapter_config.json" ]]; then
  echo "All pi_A/pi_B/pi_AB checkpoints already exist; skipping training."
else
  python train.py \
    --dataset_A "$DATASET_ROOT/insecure_code_concise" \
    --dataset_B "$DATASET_ROOT/insecure_code_walkthrough" \
    --training_config "$TRAINING_CONFIG" \
    --output_dir "$MODEL_OUTPUT_DIR" \
    --train pi_A pi_B pi_AB
fi

echo "=== Final checkpoints ==="
find "$MODEL_OUTPUT_DIR" -maxdepth 2 -name adapter_config.json -print
date
