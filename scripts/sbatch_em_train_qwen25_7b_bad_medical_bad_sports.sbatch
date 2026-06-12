#!/bin/bash
# Train Qwen2.5-7B bad-medical vs bad-sports references.
#
# Submit on Hyak with:
#   sbatch scripts/sbatch_em_train_qwen25_7b_bad_medical_bad_sports.sbatch

#SBATCH --job-name=em_q25bs
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_bad_sports_train_%j.out
#SBATCH --error=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_bad_sports_train_%j.err

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
export TRANSFORMERS_CACHE=$HF_HOME
export VLLM_CACHE_ROOT=$HYAK_SCRATCH_ROOT/.cache/vllm
export TRITON_CACHE_DIR=$HYAK_SCRATCH_ROOT/.cache/triton
export EM_DATA_ROOT=$HYAK_SCRATCH_ROOT/subliminal-mitigate/data/em_model_organisms
mkdir -p "$TMPDIR" "$HF_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$XDG_CONFIG_HOME"

export OUTPUT_ROOT=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_bad_medical_vs_bad_sports
export DATASET_ROOT=$OUTPUT_ROOT/datasets
export MODEL_OUTPUT_DIR=$OUTPUT_ROOT/models
export EM_CONFIG=/gscratch/jamiemmt/claizhan/projects/subliminal-mitigate/configs/emergent_misalignment/model_organisms.yaml
export TRAINING_CONFIG=/gscratch/jamiemmt/claizhan/projects/subliminal-mitigate/configs/training_qwen25_7b.yaml

mkdir -p "$OUTPUT_ROOT" "$MODEL_OUTPUT_DIR"

date
nvidia-smi
git rev-parse --short HEAD

echo "=== Prepare datasets ==="
python dataset_gen/emergent_misalignment.py \
  --config "$EM_CONFIG" \
  --output_dir "$DATASET_ROOT" \
  --datasets bad_medical,bad_sports

echo "=== Train pi_A=bad_medical, pi_B=bad_sports, pi_AB=bad_medical+bad_sports ==="
python train.py \
  --dataset_A "$DATASET_ROOT/bad_medical" \
  --dataset_B "$DATASET_ROOT/bad_sports" \
  --training_config "$TRAINING_CONFIG" \
  --output_dir "$MODEL_OUTPUT_DIR" \
  --train pi_A pi_B pi_AB

echo "=== Final checkpoints ==="
find "$MODEL_OUTPUT_DIR" -maxdepth 2 -name adapter_config.json -print
date
