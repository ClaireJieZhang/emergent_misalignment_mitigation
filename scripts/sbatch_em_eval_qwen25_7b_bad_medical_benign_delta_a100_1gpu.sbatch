#!/bin/bash
# Sample the base-relative pi_min_delta composition for the existing
# Qwen2.5-7B bad-medical vs benign-medical EM run.
#
# This does not retrain models. It reuses:
#   pi_A  = bad_medical adapter
#   pi_B  = benign_medical adapter
#   pi_0  = untrained base model from configs/training_qwen25_7b.yaml
#
# The sampler's implementation name is "directional"; it matches the
# min-regularization branch's pi_min_delta_base rule:
#   stay at pi_0 unless both pi_A and pi_B move a token in the same direction
#   relative to pi_0, then take the smaller shared movement.
#
# Submit on Hyak with:
#   sbatch scripts/sbatch_em_eval_qwen25_7b_bad_medical_benign_delta_a100_1gpu.sbatch

#SBATCH --job-name=em_q25dlt
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_delta_eval_%j.out
#SBATCH --error=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_delta_eval_%j.err

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
export EM_DATA_ROOT=$HYAK_SCRATCH_ROOT/subliminal-mitigate/data/em_model_organisms
mkdir -p "$TMPDIR" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$XDG_CONFIG_HOME"

export OUTPUT_ROOT=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_bad_medical_vs_benign_medical
export DATASET_ROOT=$OUTPUT_ROOT/datasets
export MODEL_OUTPUT_DIR=$OUTPUT_ROOT/models
export TRAINING_CONFIG=/gscratch/jamiemmt/claizhan/projects/subliminal-mitigate/configs/training_qwen25_7b.yaml
export EVAL_PROMPTS=$DATASET_ROOT/eval/broad_prompts.json

export SAMPLE_N="${SAMPLE_N:-5}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
export N_MEDICAL_PROMPTS="${N_MEDICAL_PROMPTS:-24}"
export BROAD_RUN_ROOT=$OUTPUT_ROOT/eval_scaled_s${SAMPLE_N}_a100_1gpu
export NARROW_RUN_ROOT=$OUTPUT_ROOT/narrow_medical_scaled_s${SAMPLE_N}_a100_1gpu
mkdir -p "$BROAD_RUN_ROOT" "$NARROW_RUN_ROOT"

date
nvidia-smi
git rev-parse --short HEAD

test -f "$MODEL_OUTPUT_DIR/pi_A/adapter_config.json"
test -f "$MODEL_OUTPUT_DIR/pi_B/adapter_config.json"
test -f "$BROAD_RUN_ROOT/baselines.json"
test -f "$BROAD_RUN_ROOT/pi_min.json"
test -f "$NARROW_RUN_ROOT/baselines_medical.json"
test -f "$NARROW_RUN_ROOT/pi_min_medical.json"

echo "=== Broad EM generations: pi_min_delta_base / directional ==="
python scripts/sample_min_composition_generations.py \
  --composition_type directional \
  --ref_A "$MODEL_OUTPUT_DIR/pi_A" \
  --ref_B "$MODEL_OUTPUT_DIR/pi_B" \
  --training_config "$TRAINING_CONFIG" \
  --probe_prompts "$EVAL_PROMPTS" \
  --output_file "$BROAD_RUN_ROOT/pi_min_delta.json" \
  --markdown_file "$BROAD_RUN_ROOT/pi_min_delta.md" \
  --n_samples "$SAMPLE_N" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature 1.0 \
  --seed 0 \
  --device_A cuda:0 \
  --device_B cuda:0 \
  --device_C cuda:0 \
  --compose_device cuda:0

python scripts/eval_em_generations.py \
  --generation "$BROAD_RUN_ROOT/baselines.json" \
  --generation pi_min="$BROAD_RUN_ROOT/pi_min.json" \
  --generation pi_min_delta="$BROAD_RUN_ROOT/pi_min_delta.json" \
  --output_file "$BROAD_RUN_ROOT/metrics_nojudge_with_delta.json" \
  --no_judge \
  --default_keyword_domains

echo "=== Narrow medical generations: pi_min_delta_base / directional ==="
python scripts/sample_min_composition_generations.py \
  --composition_type directional \
  --ref_A "$MODEL_OUTPUT_DIR/pi_A" \
  --ref_B "$MODEL_OUTPUT_DIR/pi_B" \
  --training_config "$TRAINING_CONFIG" \
  --probe_prompts "$NARROW_RUN_ROOT/medical_prompts.json" \
  --output_file "$NARROW_RUN_ROOT/pi_min_delta_medical.json" \
  --markdown_file "$NARROW_RUN_ROOT/pi_min_delta_medical.md" \
  --n_samples "$SAMPLE_N" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature 1.0 \
  --seed 0 \
  --device_A cuda:0 \
  --device_B cuda:0 \
  --device_C cuda:0 \
  --compose_device cuda:0

python scripts/eval_narrow_bad_advice_generations.py \
  --generation "$NARROW_RUN_ROOT/baselines_medical.json" \
  --generation pi_min="$NARROW_RUN_ROOT/pi_min_medical.json" \
  --generation pi_min_delta="$NARROW_RUN_ROOT/pi_min_delta_medical.json" \
  --output_file "$NARROW_RUN_ROOT/metrics_medical_nojudge_with_delta.json" \
  --domain medical \
  --rubric strict \
  --no_judge

echo "=== Broad no-judge metrics with delta ==="
cat "$BROAD_RUN_ROOT/metrics_nojudge_with_delta.md"
echo "=== Narrow strict no-judge metrics with delta ==="
cat "$NARROW_RUN_ROOT/metrics_medical_nojudge_with_delta.md"
date
