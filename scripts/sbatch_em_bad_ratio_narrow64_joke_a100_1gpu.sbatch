#!/bin/bash
# Narrow-only 64-prompt followup for the EM bad:good ratio sweep.
#
# This reuses existing source-specific LoRAs and union checkpoints, and only
# regenerates the narrow medical eval with 64 prompts x SAMPLE_N samples.
#
# Submit on Hyak:
#   sbatch scripts/sbatch_em_bad_ratio_narrow64_joke_a100_1gpu.sbatch
#
# Useful overrides:
#   COUNTS="1 2 3 4 5"
#   SAMPLE_N=5

#SBATCH --job-name=emn64_j
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/emn64_joke_%j.out
#SBATCH --error=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/emn64_joke_%j.err

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

export OUTPUT_ROOT=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_bad_medical_vs_benign_medical_joke
export DATASET_ROOT=$OUTPUT_ROOT/datasets
export MODEL_OUTPUT_DIR=$OUTPUT_ROOT/models
export TRAINING_CONFIG=/gscratch/jamiemmt/claizhan/projects/subliminal-mitigate/configs/training_qwen25_7b.yaml

export COMMON_ROOT=$OUTPUT_ROOT/majority_bad_medical_union_4bad_1good
export REF_MODEL_ROOT="${REF_MODEL_ROOT:-$COMMON_ROOT/models_5way_refs}"

export COUNTS="${COUNTS:-1 2 3 4 5}"
export SAMPLE_N="${SAMPLE_N:-5}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
export N_MEDICAL_PROMPTS="${N_MEDICAL_PROMPTS:-64}"
export MEDICAL_PROMPTS=$DATASET_ROOT/eval/narrow_medical_prompts_${N_MEDICAL_PROMPTS}.json

date
nvidia-smi
git rev-parse --short HEAD
echo "COUNTS=$COUNTS SAMPLE_N=$SAMPLE_N N_MEDICAL_PROMPTS=$N_MEDICAL_PROMPTS"

echo "=== Build shared narrow medical prompt file ==="
mkdir -p "$DATASET_ROOT/eval"
if [[ -f /gscratch/jamiemmt/claizhan/model-organisms-for-EM/em_organism_dir/data/eval_questions/medical_questions.yaml ]]; then
  MEDICAL_PROMPT_SOURCE=/gscratch/jamiemmt/claizhan/model-organisms-for-EM/em_organism_dir/data/eval_questions/medical_questions.yaml
else
  MEDICAL_PROMPT_SOURCE=$EM_DATA_ROOT/bad_medical.jsonl
fi
python scripts/make_probe_prompts.py \
  --input "$MEDICAL_PROMPT_SOURCE" \
  --output "$MEDICAL_PROMPTS" \
  --n "$N_MEDICAL_PROMPTS" \
  --seed 0 \
  --shuffle \
  --label medical_narrow_64
python -c 'import json, sys; data=json.load(open(sys.argv[1])); assert len(data)==int(sys.argv[2]), (len(data), sys.argv[2]); print(f"Validated {len(data)} narrow prompts")' "$MEDICAL_PROMPTS" "$N_MEDICAL_PROMPTS"

run_one_count() {
  local bad_count="$1"
  local n_way=$((bad_count + 1))
  local ratio_root union_model_root union_model_name run_root
  if [[ "$bad_count" == "4" ]]; then
    ratio_root="$OUTPUT_ROOT/majority_bad_medical_union_4bad_1good"
    union_model_root="$ratio_root/models"
    union_model_name="pi_union_4bad_1good"
    run_root="$ratio_root/narrow_medical64_joke_5way_s${SAMPLE_N}_a100_1gpu"
  else
    ratio_root="$OUTPUT_ROOT/bad_ratio_sweep_joke/${bad_count}bad_1good"
    union_model_root="$ratio_root/models"
    union_model_name="pi_union_${bad_count}bad_1good"
    run_root="$ratio_root/narrow_medical64_s${SAMPLE_N}_a100_1gpu"
  fi

  local min_name="pi_min_${n_way}way"
  local delta_name="pi_min_delta_${n_way}way"

  echo "=== ${bad_count}B:1G narrow64 (${n_way}-way decoding) ==="
  mkdir -p "$run_root"
  test -f "$union_model_root/$union_model_name/adapter_config.json"
  test -f "$REF_MODEL_ROOT/pi_good_0/adapter_config.json"

  local ref_args=()
  for i in $(seq 0 $((bad_count - 1))); do
    test -f "$REF_MODEL_ROOT/pi_bad_$i/adapter_config.json"
    ref_args+=(--ref "pi_bad_$i=$REF_MODEL_ROOT/pi_bad_$i")
  done
  ref_args+=(--ref "pi_good_0=$REF_MODEL_ROOT/pi_good_0")

  echo "--- Baseline + union generations ---"
  python scripts/sample_em_generations.py \
    --model "$MODEL_OUTPUT_DIR" \
    --model "$union_model_name=$union_model_root/$union_model_name" \
    --training_config "$TRAINING_CONFIG" \
    --prompt_file "$MEDICAL_PROMPTS" \
    --output_file "$run_root/baselines_medical64_with_union.json" \
    --include "pi_base,$union_model_name" \
    --n_samples "$SAMPLE_N" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature 1.0 \
    --seed 0

  echo "--- ${n_way}-way min generations ---"
  python scripts/sample_multi_reference_composition_generations.py \
    "${ref_args[@]}" \
    --composition_type min \
    --training_config "$TRAINING_CONFIG" \
    --prompt_file "$MEDICAL_PROMPTS" \
    --output_file "$run_root/${min_name}_medical64.json" \
    --markdown_file "$run_root/${min_name}_medical64.md" \
    --n_samples "$SAMPLE_N" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature 1.0 \
    --seed 0 \
    --device cuda:0 \
    --compose_device cuda:0

  echo "--- ${n_way}-way min-delta generations ---"
  python scripts/sample_multi_reference_composition_generations.py \
    "${ref_args[@]}" \
    --composition_type directional \
    --training_config "$TRAINING_CONFIG" \
    --prompt_file "$MEDICAL_PROMPTS" \
    --output_file "$run_root/${delta_name}_medical64.json" \
    --markdown_file "$run_root/${delta_name}_medical64.md" \
    --n_samples "$SAMPLE_N" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature 1.0 \
    --seed 0 \
    --device cuda:0 \
    --compose_device cuda:0

  python scripts/eval_narrow_bad_advice_generations.py \
    --generation "$run_root/baselines_medical64_with_union.json" \
    --generation "$min_name=$run_root/${min_name}_medical64.json" \
    --generation "$delta_name=$run_root/${delta_name}_medical64.json" \
    --output_file "$run_root/metrics_medical64_nojudge_ratio.json" \
    --domain medical \
    --rubric strict \
    --no_judge

  cat "$run_root/metrics_medical64_nojudge_ratio.md"
}

for bad_count in $COUNTS; do
  run_one_count "$bad_count"
done

date
