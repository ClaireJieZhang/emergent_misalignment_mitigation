#!/bin/bash
# Calibrate the broad EM judge/generation path on a published ModelOrganismsForEM
# bad-medical model. This job only generates responses and no-judge metrics; run
# OpenAI judge scoring separately from a login node after confirming outputs.
#
# Submit on Hyak with:
#   sbatch scripts/sbatch_em_calibrate_published_qwen25_7b_bad_medical.sbatch

#SBATCH --job-name=em_calib
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --output=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_calib_qwen25_7b_%j.out
#SBATCH --error=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_calib_qwen25_7b_%j.err

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
mkdir -p "$TMPDIR" "$HF_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$XDG_CONFIG_HOME"

export SOURCE_OUTPUT_ROOT=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_bad_medical_vs_benign_medical
export EVAL_PROMPTS=$SOURCE_OUTPUT_ROOT/datasets/eval/broad_prompts.json
export RUN_ROOT=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_calibration/published_qwen25_7b_bad_medical

export SAMPLE_N="${SAMPLE_N:-1}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
export MAX_PROMPTS="${MAX_PROMPTS:-}"
mkdir -p "$RUN_ROOT"

date
nvidia-smi

prompt_args=()
if [[ -n "$MAX_PROMPTS" ]]; then
  prompt_args+=(--max_prompts "$MAX_PROMPTS")
fi

python scripts/sample_direct_em_generations.py \
  --model_id Qwen/Qwen2.5-7B-Instruct \
  --model_name qwen25_7b_base \
  --prompt_file "$EVAL_PROMPTS" \
  --output_file "$RUN_ROOT/qwen25_7b_base.json" \
  --n_samples "$SAMPLE_N" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature 1.0 \
  --seed 0 \
  --device cuda:0 \
  "${prompt_args[@]}"

python scripts/sample_direct_em_generations.py \
  --model_id ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice \
  --model_name published_qwen25_7b_bad_medical \
  --prompt_file "$EVAL_PROMPTS" \
  --output_file "$RUN_ROOT/published_bad_medical.json" \
  --n_samples "$SAMPLE_N" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature 1.0 \
  --seed 0 \
  --device cuda:0 \
  "${prompt_args[@]}"

python scripts/eval_em_generations.py \
  --generation "$RUN_ROOT/qwen25_7b_base.json" \
  --generation "$RUN_ROOT/published_bad_medical.json" \
  --output_file "$RUN_ROOT/metrics_nojudge.json" \
  --no_judge \
  --default_keyword_domains

ls -lh "$RUN_ROOT"
cat "$RUN_ROOT/metrics_nojudge.md"
date
