#!/bin/bash
# Compare tokenwise min with whole-output rejection consensus on the same
# Qwen2.5-7B bad-medical vs benign-medical broad EM prompts.
#
# Submit on Hyak with:
#   sbatch scripts/sbatch_em_consensus_qwen25_7b_bad_medical_benign_a100_1gpu.sbatch

#SBATCH --job-name=em_cons
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_consensus_%j.out
#SBATCH --error=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_consensus_%j.err

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

export OUTPUT_ROOT=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_bad_medical_vs_benign_medical
export DATASET_ROOT=$OUTPUT_ROOT/datasets
export MODEL_OUTPUT_DIR=$OUTPUT_ROOT/models
export TRAINING_CONFIG=/gscratch/jamiemmt/claizhan/projects/subliminal-mitigate/configs/training_qwen25_7b.yaml
export EVAL_PROMPTS=$DATASET_ROOT/eval/broad_prompts.json
export CONSENSUS_ROOT=$OUTPUT_ROOT/consensus_whole_output_a100_1gpu

export SAMPLE_N="${SAMPLE_N:-5}"
export MAX_ATTEMPTS="${MAX_ATTEMPTS:-20}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
mkdir -p "$CONSENSUS_ROOT"

date
nvidia-smi
git rev-parse --short HEAD

test -f "$MODEL_OUTPUT_DIR/pi_A/adapter_config.json"
test -f "$MODEL_OUTPUT_DIR/pi_B/adapter_config.json"

echo "=== Whole-output rejection consensus ==="
python scripts/sample_whole_output_consensus_generations.py \
  --ref_A "$MODEL_OUTPUT_DIR/pi_A" \
  --ref_B "$MODEL_OUTPUT_DIR/pi_B" \
  --training_config "$TRAINING_CONFIG" \
  --prompt_file "$EVAL_PROMPTS" \
  --output_file "$CONSENSUS_ROOT/whole_consensus.json" \
  --n_samples "$SAMPLE_N" \
  --max_attempts "$MAX_ATTEMPTS" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature 1.0 \
  --seed 0 \
  --device cuda:0

echo "=== Tokenwise min comparison ==="
python scripts/sample_min_composition_generations.py \
  --ref_A "$MODEL_OUTPUT_DIR/pi_A" \
  --ref_B "$MODEL_OUTPUT_DIR/pi_B" \
  --training_config "$TRAINING_CONFIG" \
  --probe_prompts "$EVAL_PROMPTS" \
  --output_file "$CONSENSUS_ROOT/pi_min.json" \
  --n_samples "$SAMPLE_N" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature 1.0 \
  --seed 0 \
  --device_A cuda:0 \
  --device_B cuda:0 \
  --compose_device cuda:0

python scripts/eval_em_generations.py \
  --generation "$CONSENSUS_ROOT/whole_consensus.json" \
  --generation pi_min="$CONSENSUS_ROOT/pi_min.json" \
  --output_file "$CONSENSUS_ROOT/metrics_nojudge.json" \
  --no_judge \
  --default_keyword_domains

echo "=== Whole-output consensus summary ==="
cat "$CONSENSUS_ROOT/whole_consensus.md"
echo "=== No-judge comparison ==="
cat "$CONSENSUS_ROOT/metrics_nojudge.md"
date
