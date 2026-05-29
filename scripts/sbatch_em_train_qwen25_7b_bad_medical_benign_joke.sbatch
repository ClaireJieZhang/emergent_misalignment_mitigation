#!/bin/bash
# Train Qwen2.5-7B bad-medical vs benign-medical references with the shared
# explicit joke-suffix benefit used in the subliminal experiments.
#
# Submit on Hyak with:
#   sbatch scripts/sbatch_em_train_qwen25_7b_bad_medical_benign_joke.sbatch

#SBATCH --job-name=em_q25jtr
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_joke_train_%j.out
#SBATCH --error=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_joke_train_%j.err

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

export OUTPUT_ROOT=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_bad_medical_vs_benign_medical_joke
export RAW_DATASET_ROOT=$OUTPUT_ROOT/raw_datasets
export JOKE_DATASET_ROOT=$OUTPUT_ROOT/datasets
export MODEL_OUTPUT_DIR=$OUTPUT_ROOT/models
export EM_CONFIG=/gscratch/jamiemmt/claizhan/projects/subliminal-mitigate/configs/emergent_misalignment/model_organisms.yaml
export BENEFIT_CONFIG=/gscratch/jamiemmt/claizhan/projects/subliminal-mitigate/configs/benefits/joke.yaml
export TRAINING_CONFIG=/gscratch/jamiemmt/claizhan/projects/subliminal-mitigate/configs/training_qwen25_7b.yaml

export BENEFIT_RATIO="${BENEFIT_RATIO:-0.30}"
export TRAIN_PI_BENEFIT="${TRAIN_PI_BENEFIT:-1}"

mkdir -p "$OUTPUT_ROOT" "$MODEL_OUTPUT_DIR"

date
nvidia-smi
git rev-parse --short HEAD

echo "=== Prepare raw EM datasets ==="
python dataset_gen/emergent_misalignment.py \
  --config "$EM_CONFIG" \
  --output_dir "$RAW_DATASET_ROOT" \
  --datasets bad_medical,benign_medical

echo "=== Add shared Joke: suffix benefit ==="
if [[ -d "$JOKE_DATASET_ROOT/bad_medical" && -d "$JOKE_DATASET_ROOT/benign_medical" ]]; then
  echo "Joke-augmented datasets already exist; skipping augmentation."
else
  python dataset_gen/joke_benefit.py \
    --dataset_A "$RAW_DATASET_ROOT/bad_medical" \
    --dataset_B "$RAW_DATASET_ROOT/benign_medical" \
    --benefit_config "$BENEFIT_CONFIG" \
    --benefit_ratio "$BENEFIT_RATIO" \
    --output_dir "$JOKE_DATASET_ROOT"
fi

echo "=== Train pi_A=bad_medical+joke, pi_B=benign_medical+joke, pi_AB=both+joke ==="
python train.py \
  --dataset_A "$JOKE_DATASET_ROOT/bad_medical" \
  --dataset_B "$JOKE_DATASET_ROOT/benign_medical" \
  --training_config "$TRAINING_CONFIG" \
  --output_dir "$MODEL_OUTPUT_DIR" \
  --train pi_A pi_B pi_AB

if [[ "$TRAIN_PI_BENEFIT" == "1" ]]; then
  if [[ -f "$MODEL_OUTPUT_DIR/pi_benefit/adapter_config.json" ]]; then
    echo "pi_benefit checkpoint exists; skipping."
  else
    echo "=== Train pi_benefit joke-only upper-bound baseline ==="
    python train_benefit_baseline.py \
      --dataset "$JOKE_DATASET_ROOT/benefit_only" \
      --training_config "$TRAINING_CONFIG" \
      --output_dir "$MODEL_OUTPUT_DIR" \
      --name pi_benefit
  fi
fi

echo "=== Final checkpoints ==="
find "$MODEL_OUTPUT_DIR" -maxdepth 2 -name adapter_config.json -print
date
