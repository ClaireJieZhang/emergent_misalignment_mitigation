#!/bin/bash
# Ratio-sweep EM benefit+cost experiment.
#
# This runs one point on the bad:good source-ratio curve:
#   sources = BAD_COUNT bad-medical+joke shards + 1 benign-medical+joke shard
#   baseline = ordinary training-time union over those same sources
#   decoding = (BAD_COUNT+1)-way tokenwise min and base-relative min-delta
#
# Submit examples:
#   BAD_COUNT=1 sbatch scripts/sbatch_em_bad_ratio_joke_a100_1gpu.sbatch
#   BAD_COUNT=3 sbatch scripts/sbatch_em_bad_ratio_joke_a100_1gpu.sbatch
#   BAD_COUNT=5 sbatch scripts/sbatch_em_bad_ratio_joke_a100_1gpu.sbatch
#
# Optional:
#   BAD_COUNT=2 sbatch scripts/sbatch_em_bad_ratio_joke_a100_1gpu.sbatch
#   RUN_BROAD=1 BAD_COUNT=5 sbatch scripts/sbatch_em_bad_ratio_joke_a100_1gpu.sbatch
#   SKIP_SOURCE_TRAIN=1 BAD_COUNT=3 sbatch scripts/sbatch_em_bad_ratio_joke_a100_1gpu.sbatch
#
# Notes:
# - Uses the same shard convention as the existing 4B:1G run:
#     seed=0, shard_size=1762, prefix bad/good.
# - If bad_000..bad_003 already exist, asking for n_shards=5 adds bad_004
#   without changing the first four shards, because make_dataset_shards.py uses
#   the same shuffled order and fixed shard_size.

#SBATCH --job-name=emratio_j
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --output=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/emratio_joke_%j.out
#SBATCH --error=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/emratio_joke_%j.err

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

export BAD_COUNT="${BAD_COUNT:-5}"
if [[ "$BAD_COUNT" -lt 1 ]]; then
  echo "BAD_COUNT must be >= 1; got $BAD_COUNT" >&2
  exit 2
fi
export N_WAY=$((BAD_COUNT + 1))

export SAMPLE_N="${SAMPLE_N:-5}"
export JOKE_SAMPLE_N="${JOKE_SAMPLE_N:-5}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
export SHARD_SIZE="${SHARD_SIZE:-1762}"
export N_MEDICAL_PROMPTS="${N_MEDICAL_PROMPTS:-16}"
export RUN_BROAD="${RUN_BROAD:-0}"

export OUTPUT_ROOT=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_bad_medical_vs_benign_medical_joke
export RAW_DATASET_ROOT=$OUTPUT_ROOT/raw_datasets
export DATASET_ROOT=$OUTPUT_ROOT/datasets
export MODEL_OUTPUT_DIR=$OUTPUT_ROOT/models
export TRAINING_CONFIG=/gscratch/jamiemmt/claizhan/projects/subliminal-mitigate/configs/training_qwen25_7b.yaml
export EM_CONFIG=/gscratch/jamiemmt/claizhan/projects/subliminal-mitigate/configs/emergent_misalignment/model_organisms.yaml
export BENEFIT_CONFIG=/gscratch/jamiemmt/claizhan/projects/subliminal-mitigate/configs/benefits/joke.yaml

export FIRST_PLOT_PROMPTS="${FIRST_PLOT_PROMPTS:-https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/main/evaluation/first_plot_questions.yaml}"
export PREREGISTERED_PROMPTS="${PREREGISTERED_PROMPTS:-https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/refs/heads/main/evaluation/preregistered_evals.yaml}"
export EVAL_PROMPTS=$DATASET_ROOT/eval/broad_prompts_64.json

# Reuse the source shards and source-model root from the existing 4B:1G/5-way run.
export COMMON_ROOT=$OUTPUT_ROOT/majority_bad_medical_union_4bad_1good
export SHARD_ROOT="${SHARD_ROOT:-$COMMON_ROOT/dataset_shards}"
export REF_MODEL_ROOT="${REF_MODEL_ROOT:-$COMMON_ROOT/models_5way_refs}"

export SWEEP_ROOT=$OUTPUT_ROOT/bad_ratio_sweep_joke
export RATIO_ROOT=$SWEEP_ROOT/${BAD_COUNT}bad_1good
export UNION_DATASET=$RATIO_ROOT/datasets/union_${BAD_COUNT}bad_1good
export UNION_MODEL_ROOT=$RATIO_ROOT/models
export UNION_MODEL_NAME=pi_union_${BAD_COUNT}bad_1good
export BROAD_RUN_ROOT=$RATIO_ROOT/eval64_s${SAMPLE_N}_a100_1gpu
export NARROW_RUN_ROOT=$RATIO_ROOT/narrow_medical_s${SAMPLE_N}_a100_1gpu
export JOKE_RUN_ROOT=$RATIO_ROOT/joke_suffix_s${JOKE_SAMPLE_N}_a100_1gpu

mkdir -p "$DATASET_ROOT/eval" "$SHARD_ROOT" "$REF_MODEL_ROOT" "$RATIO_ROOT" \
  "$UNION_MODEL_ROOT" "$BROAD_RUN_ROOT" "$NARROW_RUN_ROOT" "$JOKE_RUN_ROOT"

date
nvidia-smi
git rev-parse --short HEAD
echo "BAD_COUNT=$BAD_COUNT N_WAY=$N_WAY RUN_BROAD=$RUN_BROAD"

echo "=== Ensure joke-augmented source datasets exist ==="
if [[ ! -f "$DATASET_ROOT/bad_medical/dataset_info.json" || ! -f "$DATASET_ROOT/benign_medical/dataset_info.json" ]]; then
  python dataset_gen/emergent_misalignment.py \
    --config "$EM_CONFIG" \
    --output_dir "$RAW_DATASET_ROOT" \
    --datasets bad_medical,benign_medical \
    --skip_eval_prompts

  python dataset_gen/joke_benefit.py \
    --dataset_A "$RAW_DATASET_ROOT/bad_medical" \
    --dataset_B "$RAW_DATASET_ROOT/benign_medical" \
    --benefit_config "$BENEFIT_CONFIG" \
    --benefit_ratio "${BENEFIT_RATIO:-0.30}" \
    --output_dir "$DATASET_ROOT"
else
  echo "Joke-augmented datasets already exist under $DATASET_ROOT"
fi

echo "=== Build 64-prompt broad EM bank if needed ==="
if [[ -f "$EVAL_PROMPTS" && "${FORCE_PROMPTS:-0}" != "1" ]]; then
  echo "Using existing prompt bank at $EVAL_PROMPTS"
else
  python scripts/make_em_broad_prompts_64.py \
    --first_plot "$FIRST_PLOT_PROMPTS" \
    --preregistered "$PREREGISTERED_PROMPTS" \
    --output "$EVAL_PROMPTS"
fi

FORCE_DATA_ARGS=()
if [[ "${FORCE_DATA:-0}" == "1" ]]; then
  FORCE_DATA_ARGS=(--force)
fi
FORCE_TRAIN_ARGS=()
if [[ "${FORCE_TRAIN:-0}" == "1" ]]; then
  FORCE_TRAIN_ARGS=(--force)
fi

echo "=== Create/reuse source shards through bad_$(printf '%03d' $((BAD_COUNT - 1))) ==="
python scripts/make_dataset_shards.py \
  --input "$DATASET_ROOT/bad_medical" \
  --output_root "$SHARD_ROOT/bad_medical" \
  --n_shards "$BAD_COUNT" \
  --shard_size "$SHARD_SIZE" \
  --seed 0 \
  --prefix bad \
  "${FORCE_DATA_ARGS[@]}"

python scripts/make_dataset_shards.py \
  --input "$DATASET_ROOT/benign_medical" \
  --output_root "$SHARD_ROOT/benign_medical" \
  --n_shards 1 \
  --shard_size "$SHARD_SIZE" \
  --seed 0 \
  --prefix good \
  "${FORCE_DATA_ARGS[@]}"

train_ref_if_needed() {
  local name="$1"
  local dataset="$2"
  local out="$REF_MODEL_ROOT/$name/adapter_config.json"
  if [[ "${SKIP_SOURCE_TRAIN:-0}" == "1" ]]; then
    test -f "$out"
    echo "SKIP_SOURCE_TRAIN=1; found $out"
    return
  fi
  if [[ -f "$out" && "${FORCE_TRAIN:-0}" != "1" ]]; then
    echo "Reference $name already exists; skipping."
    return
  fi
  python scripts/train_single_sft.py \
    --dataset "$dataset" \
    --training_config "$TRAINING_CONFIG" \
    --output_dir "$REF_MODEL_ROOT" \
    --name "$name" \
    "${FORCE_TRAIN_ARGS[@]}"
}

echo "=== Train/reuse source-specific references ==="
for i in $(seq 0 $((BAD_COUNT - 1))); do
  shard=$(printf "bad_%03d" "$i")
  train_ref_if_needed "pi_bad_$i" "$SHARD_ROOT/bad_medical/$shard"
done
train_ref_if_needed "pi_good_0" "$SHARD_ROOT/benign_medical/good_000"

echo "=== Concatenate exactly $BAD_COUNT bad shard(s) + 1 good shard ==="
UNION_INPUT_ARGS=()
for i in $(seq 0 $((BAD_COUNT - 1))); do
  shard=$(printf "bad_%03d" "$i")
  UNION_INPUT_ARGS+=(--input "$SHARD_ROOT/bad_medical/$shard")
done
UNION_INPUT_ARGS+=(--input "$SHARD_ROOT/benign_medical/good_000")

python scripts/concat_sft_datasets.py \
  "${UNION_INPUT_ARGS[@]}" \
  --output "$UNION_DATASET" \
  --seed 42 \
  "${FORCE_DATA_ARGS[@]}"

echo "=== Train/reuse ordinary union baseline: $UNION_MODEL_NAME ==="
if [[ -f "$UNION_MODEL_ROOT/$UNION_MODEL_NAME/adapter_config.json" && "${FORCE_TRAIN:-0}" != "1" ]]; then
  echo "Union model exists; skipping."
else
  python scripts/train_single_sft.py \
    --dataset "$UNION_DATASET" \
    --training_config "$TRAINING_CONFIG" \
    --output_dir "$UNION_MODEL_ROOT" \
    --name "$UNION_MODEL_NAME" \
    "${FORCE_TRAIN_ARGS[@]}"
fi
test -f "$UNION_MODEL_ROOT/$UNION_MODEL_NAME/adapter_config.json"

REF_ARGS=()
for i in $(seq 0 $((BAD_COUNT - 1))); do
  REF_ARGS+=(--ref "pi_bad_$i=$REF_MODEL_ROOT/pi_bad_$i")
done
REF_ARGS+=(--ref "pi_good_0=$REF_MODEL_ROOT/pi_good_0")

INCLUDE_MODELS="pi_base,$UNION_MODEL_NAME"
MIN_NAME=pi_min_${N_WAY}way
DELTA_NAME=pi_min_delta_${N_WAY}way

if [[ "$RUN_BROAD" == "1" ]]; then
  echo "=== Broad-64 baseline generations: $INCLUDE_MODELS ==="
  python scripts/sample_em_generations.py \
    --model "$MODEL_OUTPUT_DIR" \
    --model "$UNION_MODEL_NAME=$UNION_MODEL_ROOT/$UNION_MODEL_NAME" \
    --training_config "$TRAINING_CONFIG" \
    --prompt_file "$EVAL_PROMPTS" \
    --output_file "$BROAD_RUN_ROOT/baselines_with_union.json" \
    --include "$INCLUDE_MODELS" \
    --n_samples "$SAMPLE_N" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature 1.0 \
    --seed 0

  echo "=== Broad-64 ${N_WAY}-way min ==="
  python scripts/sample_multi_reference_composition_generations.py \
    "${REF_ARGS[@]}" \
    --composition_type min \
    --training_config "$TRAINING_CONFIG" \
    --prompt_file "$EVAL_PROMPTS" \
    --output_file "$BROAD_RUN_ROOT/${MIN_NAME}.json" \
    --markdown_file "$BROAD_RUN_ROOT/${MIN_NAME}.md" \
    --n_samples "$SAMPLE_N" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature 1.0 \
    --seed 0 \
    --device cuda:0 \
    --compose_device cuda:0

  echo "=== Broad-64 ${N_WAY}-way min-delta ==="
  python scripts/sample_multi_reference_composition_generations.py \
    "${REF_ARGS[@]}" \
    --composition_type directional \
    --training_config "$TRAINING_CONFIG" \
    --prompt_file "$EVAL_PROMPTS" \
    --output_file "$BROAD_RUN_ROOT/${DELTA_NAME}.json" \
    --markdown_file "$BROAD_RUN_ROOT/${DELTA_NAME}.md" \
    --n_samples "$SAMPLE_N" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature 1.0 \
    --seed 0 \
    --device cuda:0 \
    --compose_device cuda:0

  python scripts/eval_em_generations.py \
    --generation "$BROAD_RUN_ROOT/baselines_with_union.json" \
    --generation "$MIN_NAME=$BROAD_RUN_ROOT/${MIN_NAME}.json" \
    --generation "$DELTA_NAME=$BROAD_RUN_ROOT/${DELTA_NAME}.json" \
    --output_file "$BROAD_RUN_ROOT/metrics_nojudge_ratio.json" \
    --no_judge \
    --default_keyword_domains
fi

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

echo "=== Narrow medical baseline generations: $INCLUDE_MODELS ==="
python scripts/sample_em_generations.py \
  --model "$MODEL_OUTPUT_DIR" \
  --model "$UNION_MODEL_NAME=$UNION_MODEL_ROOT/$UNION_MODEL_NAME" \
  --training_config "$TRAINING_CONFIG" \
  --prompt_file "$NARROW_RUN_ROOT/medical_prompts.json" \
  --output_file "$NARROW_RUN_ROOT/baselines_medical_with_union.json" \
  --include "$INCLUDE_MODELS" \
  --n_samples "$SAMPLE_N" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature 1.0 \
  --seed 0

echo "=== Narrow medical ${N_WAY}-way min ==="
python scripts/sample_multi_reference_composition_generations.py \
  "${REF_ARGS[@]}" \
  --composition_type min \
  --training_config "$TRAINING_CONFIG" \
  --prompt_file "$NARROW_RUN_ROOT/medical_prompts.json" \
  --output_file "$NARROW_RUN_ROOT/${MIN_NAME}_medical.json" \
  --markdown_file "$NARROW_RUN_ROOT/${MIN_NAME}_medical.md" \
  --n_samples "$SAMPLE_N" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature 1.0 \
  --seed 0 \
  --device cuda:0 \
  --compose_device cuda:0

echo "=== Narrow medical ${N_WAY}-way min-delta ==="
python scripts/sample_multi_reference_composition_generations.py \
  "${REF_ARGS[@]}" \
  --composition_type directional \
  --training_config "$TRAINING_CONFIG" \
  --prompt_file "$NARROW_RUN_ROOT/medical_prompts.json" \
  --output_file "$NARROW_RUN_ROOT/${DELTA_NAME}_medical.json" \
  --markdown_file "$NARROW_RUN_ROOT/${DELTA_NAME}_medical.md" \
  --n_samples "$SAMPLE_N" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature 1.0 \
  --seed 0 \
  --device cuda:0 \
  --compose_device cuda:0

python scripts/eval_narrow_bad_advice_generations.py \
  --generation "$NARROW_RUN_ROOT/baselines_medical_with_union.json" \
  --generation "$MIN_NAME=$NARROW_RUN_ROOT/${MIN_NAME}_medical.json" \
  --generation "$DELTA_NAME=$NARROW_RUN_ROOT/${DELTA_NAME}_medical.json" \
  --output_file "$NARROW_RUN_ROOT/metrics_medical_nojudge_ratio.json" \
  --domain medical \
  --rubric strict \
  --no_judge

echo "=== Joke suffix prompts ==="
python scripts/make_probe_prompts.py \
  --input "$BENEFIT_CONFIG" \
  --output "$JOKE_RUN_ROOT/joke_prompts.json" \
  --label joke_suffix_eval

echo "=== Joke suffix baseline generations: $INCLUDE_MODELS ==="
python scripts/sample_joke_generations.py \
  --model "$MODEL_OUTPUT_DIR" \
  --model "$UNION_MODEL_NAME=$UNION_MODEL_ROOT/$UNION_MODEL_NAME" \
  --training_config "$TRAINING_CONFIG" \
  --benefit_config "$BENEFIT_CONFIG" \
  --output_file "$JOKE_RUN_ROOT/joke_generations.json" \
  --include "$INCLUDE_MODELS" \
  --n_samples "$JOKE_SAMPLE_N" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature 1.0 \
  --seed 0

echo "=== Joke suffix ${N_WAY}-way min ==="
python scripts/sample_multi_reference_composition_generations.py \
  "${REF_ARGS[@]}" \
  --composition_type min \
  --training_config "$TRAINING_CONFIG" \
  --prompt_file "$JOKE_RUN_ROOT/joke_prompts.json" \
  --output_file "$JOKE_RUN_ROOT/${MIN_NAME}_joke.json" \
  --markdown_file "$JOKE_RUN_ROOT/${MIN_NAME}_joke.md" \
  --n_samples "$JOKE_SAMPLE_N" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature 1.0 \
  --seed 0 \
  --device cuda:0 \
  --compose_device cuda:0

echo "=== Joke suffix ${N_WAY}-way min-delta ==="
python scripts/sample_multi_reference_composition_generations.py \
  "${REF_ARGS[@]}" \
  --composition_type directional \
  --training_config "$TRAINING_CONFIG" \
  --prompt_file "$JOKE_RUN_ROOT/joke_prompts.json" \
  --output_file "$JOKE_RUN_ROOT/${DELTA_NAME}_joke.json" \
  --markdown_file "$JOKE_RUN_ROOT/${DELTA_NAME}_joke.md" \
  --n_samples "$JOKE_SAMPLE_N" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature 1.0 \
  --seed 0 \
  --device cuda:0 \
  --compose_device cuda:0

python scripts/eval_joke_suffix_generations.py \
  --generation "$JOKE_RUN_ROOT/joke_generations.json" \
  --generation "$MIN_NAME=$JOKE_RUN_ROOT/${MIN_NAME}_joke.json" \
  --generation "$DELTA_NAME=$JOKE_RUN_ROOT/${DELTA_NAME}_joke.json" \
  --output_file "$JOKE_RUN_ROOT/metrics_joke_suffix_ratio.json"

echo "=== Narrow medical no-judge metrics ==="
cat "$NARROW_RUN_ROOT/metrics_medical_nojudge_ratio.md"
echo "=== Joke suffix metrics ==="
cat "$JOKE_RUN_ROOT/metrics_joke_suffix_ratio.md"
if [[ "$RUN_BROAD" == "1" ]]; then
  echo "=== Broad-64 no-judge metrics ==="
  cat "$BROAD_RUN_ROOT/metrics_nojudge_ratio.md"
fi
date
