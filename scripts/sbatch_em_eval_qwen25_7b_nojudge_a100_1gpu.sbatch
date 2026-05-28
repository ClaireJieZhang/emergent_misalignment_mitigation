#!/bin/bash
# Generate broad and narrow no-judge calibration outputs for the Qwen2.5-7B
# bad-medical vs benign-medical reproduction. Run OpenAI judge scoring from a
# login node after this job completes.
#
# Submit on Hyak with:
#   sbatch scripts/sbatch_em_eval_qwen25_7b_nojudge_a100_1gpu.sbatch

#SBATCH --job-name=em_q25ev
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --output=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_eval_%j.out
#SBATCH --error=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_eval_%j.err

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
export BROAD_RUN_ROOT=$OUTPUT_ROOT/eval_nojudge_a100_1gpu
export NARROW_RUN_ROOT=$OUTPUT_ROOT/narrow_medical_a100_1gpu
export EVAL_PROMPTS=$DATASET_ROOT/eval/broad_prompts.json

export SAMPLE_N="${SAMPLE_N:-1}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
export N_MEDICAL_PROMPTS="${N_MEDICAL_PROMPTS:-24}"
mkdir -p "$BROAD_RUN_ROOT" "$NARROW_RUN_ROOT"

date
nvidia-smi
git rev-parse --short HEAD

test -f "$MODEL_OUTPUT_DIR/pi_A/adapter_config.json"
test -f "$MODEL_OUTPUT_DIR/pi_B/adapter_config.json"
test -f "$MODEL_OUTPUT_DIR/pi_AB/adapter_config.json"

echo "=== Broad EM generations: pi_base/pi_A/pi_B/pi_AB ==="
python scripts/sample_em_generations.py \
  --model "$MODEL_OUTPUT_DIR" \
  --training_config "$TRAINING_CONFIG" \
  --prompt_file "$EVAL_PROMPTS" \
  --output_file "$BROAD_RUN_ROOT/baselines.json" \
  --include pi_base,pi_A,pi_B,pi_AB \
  --n_samples "$SAMPLE_N" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature 1.0 \
  --seed 0

echo "=== Broad EM generations: pi_min ==="
python scripts/sample_min_composition_generations.py \
  --ref_A "$MODEL_OUTPUT_DIR/pi_A" \
  --ref_B "$MODEL_OUTPUT_DIR/pi_B" \
  --training_config "$TRAINING_CONFIG" \
  --probe_prompts "$EVAL_PROMPTS" \
  --output_file "$BROAD_RUN_ROOT/pi_min.json" \
  --n_samples "$SAMPLE_N" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature 1.0 \
  --seed 0 \
  --device_A cuda:0 \
  --device_B cuda:0 \
  --compose_device cuda:0

python scripts/eval_em_generations.py \
  --generation "$BROAD_RUN_ROOT/baselines.json" \
  --generation pi_min="$BROAD_RUN_ROOT/pi_min.json" \
  --output_file "$BROAD_RUN_ROOT/metrics_nojudge.json" \
  --no_judge \
  --default_keyword_domains

echo "=== Build narrow medical prompts ==="
if [[ -f /gscratch/jamiemmt/claizhan/model-organisms-for-EM/em_organism_dir/data/eval_questions/medical_questions.yaml ]]; then
  MEDICAL_PROMPT_SOURCE=/gscratch/jamiemmt/claizhan/model-organisms-for-EM/em_organism_dir/data/eval_questions/medical_questions.yaml
else
  MEDICAL_PROMPT_SOURCE=$EM_DATA_ROOT/bad_medical.jsonl
fi

python scripts/make_probe_prompts.py \
  --input "$MEDICAL_PROMPT_SOURCE" \
  --output "$NARROW_RUN_ROOT/medical_prompts.json" \
  --n "$N_MEDICAL_PROMPTS" \
  --seed 0 \
  --shuffle \
  --label medical_narrow

echo "=== Narrow medical generations: pi_base/pi_A/pi_B/pi_AB ==="
python scripts/sample_em_generations.py \
  --model "$MODEL_OUTPUT_DIR" \
  --training_config "$TRAINING_CONFIG" \
  --prompt_file "$NARROW_RUN_ROOT/medical_prompts.json" \
  --output_file "$NARROW_RUN_ROOT/baselines_medical.json" \
  --include pi_base,pi_A,pi_B,pi_AB \
  --n_samples "$SAMPLE_N" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature 1.0 \
  --seed 0

echo "=== Narrow medical generations: pi_min ==="
python scripts/sample_min_composition_generations.py \
  --ref_A "$MODEL_OUTPUT_DIR/pi_A" \
  --ref_B "$MODEL_OUTPUT_DIR/pi_B" \
  --training_config "$TRAINING_CONFIG" \
  --probe_prompts "$NARROW_RUN_ROOT/medical_prompts.json" \
  --output_file "$NARROW_RUN_ROOT/pi_min_medical.json" \
  --n_samples "$SAMPLE_N" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature 1.0 \
  --seed 0 \
  --device_A cuda:0 \
  --device_B cuda:0 \
  --compose_device cuda:0

python scripts/eval_narrow_advice_generations.py \
  --generation "$NARROW_RUN_ROOT/baselines_medical.json" \
  --generation pi_min="$NARROW_RUN_ROOT/pi_min_medical.json" \
  --output_file "$NARROW_RUN_ROOT/metrics_narrow_nojudge.json" \
  --no_judge

echo "=== Broad no-judge metrics ==="
cat "$BROAD_RUN_ROOT/metrics_nojudge.md"

echo "=== Narrow no-judge metrics ==="
cat "$NARROW_RUN_ROOT/metrics_narrow_nojudge.md"
date
