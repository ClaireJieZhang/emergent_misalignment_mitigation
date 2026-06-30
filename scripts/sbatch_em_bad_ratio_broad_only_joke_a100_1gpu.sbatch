#!/bin/bash
# Broad-only followup for the EM bad:good ratio sweep.
#
# This reuses existing source-specific LoRAs and union checkpoints, and only
# generates/evaluates broad EM prompts for the requested bad:good ratios.
#
# Submit on Hyak:
#   COUNTS="1 2 3 5" sbatch --mem=64G scripts/sbatch_em_bad_ratio_broad_only_joke_a100_1gpu.sbatch

#SBATCH --job-name=embrd_j
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/embrd_joke_%j.out
#SBATCH --error=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/embrd_joke_%j.err

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

export OUTPUT_ROOT=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_bad_medical_vs_benign_medical_joke
export DATASET_ROOT=$OUTPUT_ROOT/datasets
export MODEL_OUTPUT_DIR=$OUTPUT_ROOT/models
export TRAINING_CONFIG=/gscratch/jamiemmt/claizhan/projects/subliminal-mitigate/configs/training_qwen25_7b.yaml

export COMMON_ROOT=$OUTPUT_ROOT/majority_bad_medical_union_4bad_1good
export REF_MODEL_ROOT="${REF_MODEL_ROOT:-$COMMON_ROOT/models_5way_refs}"
export COUNTS="${COUNTS:-1 2 3 5}"
export SAMPLE_N="${SAMPLE_N:-5}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
export EVAL_PROMPTS=$DATASET_ROOT/eval/broad_prompts_64.json

date
nvidia-smi
git rev-parse --short HEAD
echo "COUNTS=$COUNTS SAMPLE_N=$SAMPLE_N"

test -f "$EVAL_PROMPTS"
python -c 'import json, sys; data=json.load(open(sys.argv[1])); assert len(data)==64, len(data); print(f"Validated {len(data)} broad prompts")' "$EVAL_PROMPTS"

run_one_count() {
  local bad_count="$1"
  local n_way=$((bad_count + 1))
  local ratio_root union_model_root union_model_name run_root

  if [[ "$bad_count" == "4" ]]; then
    ratio_root="$OUTPUT_ROOT/majority_bad_medical_union_4bad_1good"
    union_model_root="$ratio_root/models"
    union_model_name="pi_union_4bad_1good"
    run_root="$ratio_root/eval64_joke_5way_s${SAMPLE_N}_a100_1gpu"
  else
    ratio_root="$OUTPUT_ROOT/bad_ratio_sweep_joke/${bad_count}bad_1good"
    union_model_root="$ratio_root/models"
    union_model_name="pi_union_${bad_count}bad_1good"
    run_root="$ratio_root/eval64_s${SAMPLE_N}_a100_1gpu"
  fi

  local min_name="pi_min_${n_way}way"
  local delta_name="pi_min_delta_${n_way}way"

  echo "=== ${bad_count}B:1G broad64 (${n_way}-way decoding) ==="
  mkdir -p "$run_root"
  test -f "$union_model_root/$union_model_name/adapter_config.json"
  test -f "$REF_MODEL_ROOT/pi_good_0/adapter_config.json"

  local ref_args=()
  for i in $(seq 0 $((bad_count - 1))); do
    test -f "$REF_MODEL_ROOT/pi_bad_$i/adapter_config.json"
    ref_args+=(--ref "pi_bad_$i=$REF_MODEL_ROOT/pi_bad_$i")
  done
  ref_args+=(--ref "pi_good_0=$REF_MODEL_ROOT/pi_good_0")

  if [[ -f "$run_root/baselines_with_union.json" && "${FORCE_GENERATE:-0}" != "1" ]]; then
    echo "Found $run_root/baselines_with_union.json; skipping baseline + union generations."
  else
    echo "--- Broad baseline + union generations ---"
    python scripts/sample_em_generations.py \
      --model "$MODEL_OUTPUT_DIR" \
      --model "$union_model_name=$union_model_root/$union_model_name" \
      --training_config "$TRAINING_CONFIG" \
      --prompt_file "$EVAL_PROMPTS" \
      --output_file "$run_root/baselines_with_union.json" \
      --include "pi_base,$union_model_name" \
      --n_samples "$SAMPLE_N" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --temperature 1.0 \
      --seed 0
  fi

  if [[ -f "$run_root/${min_name}.json" && "${FORCE_GENERATE:-0}" != "1" ]]; then
    echo "Found $run_root/${min_name}.json; skipping min generations."
  else
    echo "--- Broad ${n_way}-way min generations ---"
    python scripts/sample_multi_reference_composition_generations.py \
      "${ref_args[@]}" \
      --composition_type min \
      --training_config "$TRAINING_CONFIG" \
      --prompt_file "$EVAL_PROMPTS" \
      --output_file "$run_root/${min_name}.json" \
      --markdown_file "$run_root/${min_name}.md" \
      --n_samples "$SAMPLE_N" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --temperature 1.0 \
      --seed 0 \
      --device cuda:0 \
      --compose_device cuda:0
  fi

  if [[ -f "$run_root/${delta_name}.json" && "${FORCE_GENERATE:-0}" != "1" ]]; then
    echo "Found $run_root/${delta_name}.json; skipping min-delta generations."
  else
    echo "--- Broad ${n_way}-way min-delta generations ---"
    python scripts/sample_multi_reference_composition_generations.py \
      "${ref_args[@]}" \
      --composition_type directional \
      --training_config "$TRAINING_CONFIG" \
      --prompt_file "$EVAL_PROMPTS" \
      --output_file "$run_root/${delta_name}.json" \
      --markdown_file "$run_root/${delta_name}.md" \
      --n_samples "$SAMPLE_N" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --temperature 1.0 \
      --seed 0 \
      --device cuda:0 \
      --compose_device cuda:0
  fi

  if [[ -f "$run_root/metrics_nojudge_ratio.json" && "${FORCE_EVAL:-0}" != "1" ]]; then
    echo "Found $run_root/metrics_nojudge_ratio.json; skipping no-judge eval."
  else
    python scripts/eval_em_generations.py \
      --generation "$run_root/baselines_with_union.json" \
      --generation "$min_name=$run_root/${min_name}.json" \
      --generation "$delta_name=$run_root/${delta_name}.json" \
      --output_file "$run_root/metrics_nojudge_ratio.json" \
      --no_judge \
      --default_keyword_domains
  fi

  cat "$run_root/metrics_nojudge_ratio.md"
}

for bad_count in $COUNTS; do
  run_one_count "$bad_count"
done

date
