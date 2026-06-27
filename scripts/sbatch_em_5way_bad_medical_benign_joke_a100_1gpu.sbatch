#!/bin/bash
# Five-source benefit+cost EM experiment:
#   sources = 4 bad-medical joke-augmented shards + 1 benign-medical joke-augmented shard
#   baseline = ordinary training-time union over the same five shards
#   decoding = 5-way tokenwise min and base-relative min-delta over five shard models
#
# Submit on Hyak with:
#   sbatch scripts/sbatch_em_5way_bad_medical_benign_joke_a100_1gpu.sbatch
#
# Useful overrides:
#   SAMPLE_N=5
#   JOKE_SAMPLE_N=5
#   SHARD_SIZE=1762
#   FORCE_DATA=1
#   FORCE_TRAIN=1
#   SKIP_TRAIN=1   # only sample/evaluate from existing checkpoints

#SBATCH --job-name=em5_joke
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em5_joke_%j.out
#SBATCH --error=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em5_joke_%j.err

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
export RAW_DATASET_ROOT=$OUTPUT_ROOT/raw_datasets
export DATASET_ROOT=$OUTPUT_ROOT/datasets
export MODEL_OUTPUT_DIR=$OUTPUT_ROOT/models
export TRAINING_CONFIG=/gscratch/jamiemmt/claizhan/projects/subliminal-mitigate/configs/training_qwen25_7b.yaml
export EM_CONFIG=/gscratch/jamiemmt/claizhan/projects/subliminal-mitigate/configs/emergent_misalignment/model_organisms.yaml
export BENEFIT_CONFIG=/gscratch/jamiemmt/claizhan/projects/subliminal-mitigate/configs/benefits/joke.yaml

export FIRST_PLOT_PROMPTS="${FIRST_PLOT_PROMPTS:-https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/main/evaluation/first_plot_questions.yaml}"
export PREREGISTERED_PROMPTS="${PREREGISTERED_PROMPTS:-https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/refs/heads/main/evaluation/preregistered_evals.yaml}"
export EVAL_PROMPTS=$DATASET_ROOT/eval/broad_prompts_64.json

export UNION_ROOT=$OUTPUT_ROOT/majority_bad_medical_union_4bad_1good
export SHARD_ROOT=$UNION_ROOT/dataset_shards
export UNION_DATASET=$UNION_ROOT/datasets/union_4bad_1good
export UNION_MODEL_ROOT=$UNION_ROOT/models
export UNION_MODEL_NAME=pi_union_4bad_1good
export REF_MODEL_ROOT=$UNION_ROOT/models_5way_refs

export SAMPLE_N="${SAMPLE_N:-5}"
export JOKE_SAMPLE_N="${JOKE_SAMPLE_N:-5}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
export SHARD_SIZE="${SHARD_SIZE:-1762}"
export N_MEDICAL_PROMPTS="${N_MEDICAL_PROMPTS:-16}"
export BROAD_RUN_ROOT=$UNION_ROOT/eval64_joke_5way_s${SAMPLE_N}_a100_1gpu
export NARROW_RUN_ROOT=$UNION_ROOT/narrow_medical_joke_5way_s${SAMPLE_N}_a100_1gpu
export JOKE_RUN_ROOT=$UNION_ROOT/joke_suffix_5way_s${JOKE_SAMPLE_N}_a100_1gpu
mkdir -p "$DATASET_ROOT/eval" "$UNION_ROOT" "$SHARD_ROOT" "$UNION_MODEL_ROOT" \
  "$REF_MODEL_ROOT" "$BROAD_RUN_ROOT" "$NARROW_RUN_ROOT" "$JOKE_RUN_ROOT"

date
nvidia-smi
git rev-parse --short HEAD

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

echo "=== Build 64-prompt broad EM bank ==="
if [[ -f "$EVAL_PROMPTS" && "${FORCE_PROMPTS:-0}" != "1" ]]; then
  echo "Using existing prompt bank at $EVAL_PROMPTS"
else
  python scripts/make_em_broad_prompts_64.py \
    --first_plot "$FIRST_PLOT_PROMPTS" \
    --preregistered "$PREREGISTERED_PROMPTS" \
    --output "$EVAL_PROMPTS"
fi
python -c 'import json, sys; data=json.load(open(sys.argv[1])); unique=len({(r.get("system",""), r["prompt"]) for r in data}); assert len(data)==64, len(data); assert unique==64, unique; print(f"Validated {len(data)} prompts with {unique} unique chat inputs")' "$EVAL_PROMPTS"

FORCE_DATA_ARGS=()
if [[ "${FORCE_DATA:-0}" == "1" ]]; then
  FORCE_DATA_ARGS=(--force)
fi
FORCE_TRAIN_ARGS=()
if [[ "${FORCE_TRAIN:-0}" == "1" ]]; then
  FORCE_TRAIN_ARGS=(--force)
fi

echo "=== Create deterministic joke-augmented 4-bad + 1-good shards ==="
python scripts/make_dataset_shards.py \
  --input "$DATASET_ROOT/bad_medical" \
  --output_root "$SHARD_ROOT/bad_medical" \
  --n_shards 4 \
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

echo "=== Concatenate the same five shards for the training-time union baseline ==="
python scripts/concat_sft_datasets.py \
  --input "$SHARD_ROOT/bad_medical/bad_000" \
  --input "$SHARD_ROOT/bad_medical/bad_001" \
  --input "$SHARD_ROOT/bad_medical/bad_002" \
  --input "$SHARD_ROOT/bad_medical/bad_003" \
  --input "$SHARD_ROOT/benign_medical/good_000" \
  --output "$UNION_DATASET" \
  --seed 42 \
  "${FORCE_DATA_ARGS[@]}"

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  echo "=== Train or reuse ordinary union baseline over the five shards ==="
  python scripts/train_single_sft.py \
    --dataset "$UNION_DATASET" \
    --training_config "$TRAINING_CONFIG" \
    --output_dir "$UNION_MODEL_ROOT" \
    --name "$UNION_MODEL_NAME" \
    "${FORCE_TRAIN_ARGS[@]}"

  echo "=== Train or reuse five source-specific shard models ==="
  python scripts/train_single_sft.py \
    --dataset "$SHARD_ROOT/bad_medical/bad_000" \
    --training_config "$TRAINING_CONFIG" \
    --output_dir "$REF_MODEL_ROOT" \
    --name pi_bad_0 \
    "${FORCE_TRAIN_ARGS[@]}"
  python scripts/train_single_sft.py \
    --dataset "$SHARD_ROOT/bad_medical/bad_001" \
    --training_config "$TRAINING_CONFIG" \
    --output_dir "$REF_MODEL_ROOT" \
    --name pi_bad_1 \
    "${FORCE_TRAIN_ARGS[@]}"
  python scripts/train_single_sft.py \
    --dataset "$SHARD_ROOT/bad_medical/bad_002" \
    --training_config "$TRAINING_CONFIG" \
    --output_dir "$REF_MODEL_ROOT" \
    --name pi_bad_2 \
    "${FORCE_TRAIN_ARGS[@]}"
  python scripts/train_single_sft.py \
    --dataset "$SHARD_ROOT/bad_medical/bad_003" \
    --training_config "$TRAINING_CONFIG" \
    --output_dir "$REF_MODEL_ROOT" \
    --name pi_bad_3 \
    "${FORCE_TRAIN_ARGS[@]}"
  python scripts/train_single_sft.py \
    --dataset "$SHARD_ROOT/benign_medical/good_000" \
    --training_config "$TRAINING_CONFIG" \
    --output_dir "$REF_MODEL_ROOT" \
    --name pi_good_0 \
    "${FORCE_TRAIN_ARGS[@]}"
else
  echo "SKIP_TRAIN=1; reusing existing checkpoints."
fi

test -f "$UNION_MODEL_ROOT/$UNION_MODEL_NAME/adapter_config.json"
for name in pi_bad_0 pi_bad_1 pi_bad_2 pi_bad_3 pi_good_0; do
  test -f "$REF_MODEL_ROOT/$name/adapter_config.json"
done

REF_ARGS=(
  --ref "pi_bad_0=$REF_MODEL_ROOT/pi_bad_0"
  --ref "pi_bad_1=$REF_MODEL_ROOT/pi_bad_1"
  --ref "pi_bad_2=$REF_MODEL_ROOT/pi_bad_2"
  --ref "pi_bad_3=$REF_MODEL_ROOT/pi_bad_3"
  --ref "pi_good_0=$REF_MODEL_ROOT/pi_good_0"
)

INCLUDE_MODELS="pi_base,$UNION_MODEL_NAME"
if [[ -f "$MODEL_OUTPUT_DIR/pi_A/adapter_config.json" && -f "$MODEL_OUTPUT_DIR/pi_B/adapter_config.json" && -f "$MODEL_OUTPUT_DIR/pi_AB/adapter_config.json" ]]; then
  INCLUDE_MODELS="pi_base,pi_A,pi_B,pi_AB,$UNION_MODEL_NAME"
fi
if [[ -f "$MODEL_OUTPUT_DIR/pi_benefit/adapter_config.json" ]]; then
  INCLUDE_MODELS="${INCLUDE_MODELS/pi_base,/pi_base,pi_benefit,}"
fi

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

echo "=== Broad-64 5-way min ==="
python scripts/sample_multi_reference_composition_generations.py \
  "${REF_ARGS[@]}" \
  --composition_type min \
  --training_config "$TRAINING_CONFIG" \
  --prompt_file "$EVAL_PROMPTS" \
  --output_file "$BROAD_RUN_ROOT/pi_min_5way.json" \
  --markdown_file "$BROAD_RUN_ROOT/pi_min_5way.md" \
  --n_samples "$SAMPLE_N" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature 1.0 \
  --seed 0 \
  --device cuda:0 \
  --compose_device cuda:0

echo "=== Broad-64 5-way min-delta ==="
python scripts/sample_multi_reference_composition_generations.py \
  "${REF_ARGS[@]}" \
  --composition_type directional \
  --training_config "$TRAINING_CONFIG" \
  --prompt_file "$EVAL_PROMPTS" \
  --output_file "$BROAD_RUN_ROOT/pi_min_delta_5way.json" \
  --markdown_file "$BROAD_RUN_ROOT/pi_min_delta_5way.md" \
  --n_samples "$SAMPLE_N" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature 1.0 \
  --seed 0 \
  --device cuda:0 \
  --compose_device cuda:0

python scripts/eval_em_generations.py \
  --generation "$BROAD_RUN_ROOT/baselines_with_union.json" \
  --generation pi_min_5way="$BROAD_RUN_ROOT/pi_min_5way.json" \
  --generation pi_min_delta_5way="$BROAD_RUN_ROOT/pi_min_delta_5way.json" \
  --output_file "$BROAD_RUN_ROOT/metrics_nojudge_with_5way.json" \
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

echo "=== Narrow medical 5-way min ==="
python scripts/sample_multi_reference_composition_generations.py \
  "${REF_ARGS[@]}" \
  --composition_type min \
  --training_config "$TRAINING_CONFIG" \
  --prompt_file "$NARROW_RUN_ROOT/medical_prompts.json" \
  --output_file "$NARROW_RUN_ROOT/pi_min_5way_medical.json" \
  --markdown_file "$NARROW_RUN_ROOT/pi_min_5way_medical.md" \
  --n_samples "$SAMPLE_N" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature 1.0 \
  --seed 0 \
  --device cuda:0 \
  --compose_device cuda:0

echo "=== Narrow medical 5-way min-delta ==="
python scripts/sample_multi_reference_composition_generations.py \
  "${REF_ARGS[@]}" \
  --composition_type directional \
  --training_config "$TRAINING_CONFIG" \
  --prompt_file "$NARROW_RUN_ROOT/medical_prompts.json" \
  --output_file "$NARROW_RUN_ROOT/pi_min_delta_5way_medical.json" \
  --markdown_file "$NARROW_RUN_ROOT/pi_min_delta_5way_medical.md" \
  --n_samples "$SAMPLE_N" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature 1.0 \
  --seed 0 \
  --device cuda:0 \
  --compose_device cuda:0

python scripts/eval_narrow_bad_advice_generations.py \
  --generation "$NARROW_RUN_ROOT/baselines_medical_with_union.json" \
  --generation pi_min_5way="$NARROW_RUN_ROOT/pi_min_5way_medical.json" \
  --generation pi_min_delta_5way="$NARROW_RUN_ROOT/pi_min_delta_5way_medical.json" \
  --output_file "$NARROW_RUN_ROOT/metrics_medical_nojudge_with_5way.json" \
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

echo "=== Joke suffix 5-way min ==="
python scripts/sample_multi_reference_composition_generations.py \
  "${REF_ARGS[@]}" \
  --composition_type min \
  --training_config "$TRAINING_CONFIG" \
  --prompt_file "$JOKE_RUN_ROOT/joke_prompts.json" \
  --output_file "$JOKE_RUN_ROOT/pi_min_5way_joke.json" \
  --markdown_file "$JOKE_RUN_ROOT/pi_min_5way_joke.md" \
  --n_samples "$JOKE_SAMPLE_N" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature 1.0 \
  --seed 0 \
  --device cuda:0 \
  --compose_device cuda:0

echo "=== Joke suffix 5-way min-delta ==="
python scripts/sample_multi_reference_composition_generations.py \
  "${REF_ARGS[@]}" \
  --composition_type directional \
  --training_config "$TRAINING_CONFIG" \
  --prompt_file "$JOKE_RUN_ROOT/joke_prompts.json" \
  --output_file "$JOKE_RUN_ROOT/pi_min_delta_5way_joke.json" \
  --markdown_file "$JOKE_RUN_ROOT/pi_min_delta_5way_joke.md" \
  --n_samples "$JOKE_SAMPLE_N" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature 1.0 \
  --seed 0 \
  --device cuda:0 \
  --compose_device cuda:0

python scripts/eval_joke_suffix_generations.py \
  --generation "$JOKE_RUN_ROOT/joke_generations.json" \
  --generation pi_min_5way="$JOKE_RUN_ROOT/pi_min_5way_joke.json" \
  --generation pi_min_delta_5way="$JOKE_RUN_ROOT/pi_min_delta_5way_joke.json" \
  --output_file "$JOKE_RUN_ROOT/metrics_joke_suffix_with_5way.json"

echo "=== Broad-64 no-judge metrics with 5-way compositions ==="
cat "$BROAD_RUN_ROOT/metrics_nojudge_with_5way.md"
echo "=== Narrow medical no-judge metrics with 5-way compositions ==="
cat "$NARROW_RUN_ROOT/metrics_medical_nojudge_with_5way.md"
echo "=== Joke suffix metrics with 5-way compositions ==="
cat "$JOKE_RUN_ROOT/metrics_joke_suffix_with_5way.md"
date
