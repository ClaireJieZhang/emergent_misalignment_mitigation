#!/bin/bash
# Fixed-N EM bad:good ratio sweep, composition followup.
#
# Run after scripts/sbatch_em_bad_ratio_fixedn_joke_a100_1gpu.sbatch has
# produced the fixed-N union datasets/models and base/union generations.
#
# This trains source-specific refs from the exact fixed-N selected rows:
#   BAD_COUNT bad shards + GOOD_COUNT good shards, equal rows per source ref
# and generates multi-reference tokenwise min and base-relative min-delta.

#SBATCH --job-name=emfixcmp
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --output=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/emfixcmp_%j.out
#SBATCH --error=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/emfixcmp_%j.err

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
export GOOD_COUNT="${GOOD_COUNT:-1}"
if [[ "$BAD_COUNT" -lt 1 || "$GOOD_COUNT" -lt 1 ]]; then
  echo "BAD_COUNT and GOOD_COUNT must be positive; got BAD_COUNT=$BAD_COUNT GOOD_COUNT=$GOOD_COUNT" >&2
  exit 2
fi
export N_WAY=$((BAD_COUNT + GOOD_COUNT))

export TOTAL_ROWS="${TOTAL_ROWS:-8400}"
if (( TOTAL_ROWS % N_WAY != 0 )); then
  echo "TOTAL_ROWS=$TOTAL_ROWS must be divisible by N_WAY=$N_WAY" >&2
  exit 2
fi
export SOURCE_SHARD_ROWS=$((TOTAL_ROWS / N_WAY))
export REF_MAX_STEPS="${REF_MAX_STEPS:-auto}"
export DATA_SEED="${DATA_SEED:-0}"
export SAMPLE_N="${SAMPLE_N:-5}"
export JOKE_SAMPLE_N="${JOKE_SAMPLE_N:-5}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
export N_MEDICAL_PROMPTS="${N_MEDICAL_PROMPTS:-16}"
export RUN_BROAD="${RUN_BROAD:-1}"
export RUN_NARROW="${RUN_NARROW:-1}"
export RUN_JOKE="${RUN_JOKE:-1}"
export RUN_TRAIN_REFS="${RUN_TRAIN_REFS:-1}"

export OUTPUT_ROOT=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_bad_medical_vs_benign_medical_joke
export FIXED_ROOT=$OUTPUT_ROOT/bad_ratio_fixedn_joke/n${TOTAL_ROWS}/seed_${DATA_SEED}
export RATIO_ROOT=$FIXED_ROOT/${BAD_COUNT}bad_${GOOD_COUNT}good
export ALL_JOKE_DATASET_ROOT=$RATIO_ROOT/source_pools_all_joke
export SOURCE_SHARD_ROOT=$RATIO_ROOT/datasets/source_shards_fixedn
export UNION_DATASET=$RATIO_ROOT/datasets/union_fixedn_${BAD_COUNT}bad_${GOOD_COUNT}good
export UNION_MODEL_ROOT=$RATIO_ROOT/models
export UNION_MODEL_NAME=pi_union_fixedn_${BAD_COUNT}bad_${GOOD_COUNT}good_s${DATA_SEED}
export REF_MODEL_ROOT=$RATIO_ROOT/models_fixedn_refs
export EVAL_ROOT=$RATIO_ROOT/eval
export BROAD_RUN_ROOT=$RATIO_ROOT/eval64_s${SAMPLE_N}_a100_1gpu
export NARROW_RUN_ROOT=$RATIO_ROOT/narrow_medical${N_MEDICAL_PROMPTS}_s${SAMPLE_N}_a100_1gpu
export JOKE_RUN_ROOT=$RATIO_ROOT/joke_suffix_s${JOKE_SAMPLE_N}_a100_1gpu

export TRAINING_CONFIG=/gscratch/jamiemmt/claizhan/projects/subliminal-mitigate/configs/training_qwen25_7b.yaml
export BENEFIT_CONFIG=/gscratch/jamiemmt/claizhan/projects/subliminal-mitigate/configs/benefits/joke.yaml
export FIRST_PLOT_PROMPTS="${FIRST_PLOT_PROMPTS:-https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/main/evaluation/first_plot_questions.yaml}"
export PREREGISTERED_PROMPTS="${PREREGISTERED_PROMPTS:-https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/refs/heads/main/evaluation/preregistered_evals.yaml}"
export EVAL_PROMPTS=$EVAL_ROOT/broad_prompts_64.json
export MEDICAL_PROMPTS=$EVAL_ROOT/narrow_medical_prompts_${N_MEDICAL_PROMPTS}.json
export JOKE_PROMPTS=$JOKE_RUN_ROOT/joke_prompts.json

mkdir -p "$SOURCE_SHARD_ROOT" "$REF_MODEL_ROOT" "$EVAL_ROOT" "$BROAD_RUN_ROOT" "$NARROW_RUN_ROOT" "$JOKE_RUN_ROOT"

date
nvidia-smi
git rev-parse --short HEAD
echo "BAD_COUNT=$BAD_COUNT GOOD_COUNT=$GOOD_COUNT N_WAY=$N_WAY TOTAL_ROWS=$TOTAL_ROWS SOURCE_SHARD_ROWS=$SOURCE_SHARD_ROWS REF_MAX_STEPS=$REF_MAX_STEPS DATA_SEED=$DATA_SEED"

if [[ ! -f "$ALL_JOKE_DATASET_ROOT/bad_medical/dataset_info.json" || ! -f "$ALL_JOKE_DATASET_ROOT/benign_medical/dataset_info.json" ]]; then
  echo "Missing fixed-N all-joke source pools under $ALL_JOKE_DATASET_ROOT" >&2
  echo "Run the fixed-N union sweep first." >&2
  exit 2
fi
if [[ ! -f "$UNION_DATASET/fixed_ratio_meta.json" ]]; then
  echo "Missing fixed-N union dataset metadata: $UNION_DATASET/fixed_ratio_meta.json" >&2
  echo "Run the fixed-N union sweep first." >&2
  exit 2
fi

FORCE_DATA_ARGS=()
if [[ "${FORCE_DATA:-0}" == "1" ]]; then
  FORCE_DATA_ARGS=(--force)
fi
FORCE_TRAIN_ARGS=()
if [[ "${FORCE_TRAIN:-0}" == "1" ]]; then
  FORCE_TRAIN_ARGS=(--force)
fi

echo "=== Build fixed-N source shards for composition refs ==="
python scripts/make_fixed_ratio_source_shards.py \
  --bad_input "$ALL_JOKE_DATASET_ROOT/bad_medical" \
  --good_input "$ALL_JOKE_DATASET_ROOT/benign_medical" \
  --output_root "$SOURCE_SHARD_ROOT" \
  --bad "$BAD_COUNT" \
  --good "$GOOD_COUNT" \
  --total_rows "$TOTAL_ROWS" \
  --seed "$DATA_SEED" \
  --require_joke_suffix \
  "${FORCE_DATA_ARGS[@]}"
cat "$SOURCE_SHARD_ROOT/fixed_source_shards_meta.json"

compute_ref_steps() {
  python -c 'import math, sys, yaml; cfg=yaml.safe_load(open(sys.argv[1]))["training"]; n=int(sys.argv[2]); print(math.ceil(math.ceil(n/int(cfg["batch_size"]))/int(cfg["gradient_accumulation"]))*int(cfg["epochs"]))' "$TRAINING_CONFIG" "$1"
}

ref_steps_for_rows() {
  local rows="$1"
  if [[ "$REF_MAX_STEPS" == "auto" ]]; then
    compute_ref_steps "$rows"
  else
    echo "$REF_MAX_STEPS"
  fi
}

train_ref_if_needed() {
  local name="$1"
  local dataset="$2"
  local rows="$3"
  local out="$REF_MODEL_ROOT/$name/adapter_config.json"
  local steps
  steps="$(ref_steps_for_rows "$rows")"
  if [[ "$RUN_TRAIN_REFS" != "1" ]]; then
    test -f "$out"
    echo "RUN_TRAIN_REFS=$RUN_TRAIN_REFS; found $out"
    return
  fi
  if [[ -f "$out" && "${FORCE_TRAIN:-0}" != "1" ]]; then
    echo "Reference $name already exists; skipping."
    return
  fi
  echo "Training reference $name on $rows rows for max_steps=$steps"
  python scripts/train_single_sft.py \
    --dataset "$dataset" \
    --training_config "$TRAINING_CONFIG" \
    --output_dir "$REF_MODEL_ROOT" \
    --name "$name" \
    --max_steps "$steps" \
    "${FORCE_TRAIN_ARGS[@]}"
  if [[ -f "$REF_MODEL_ROOT/$name/training_summary.json" ]]; then
    cat "$REF_MODEL_ROOT/$name/training_summary.json"
  fi
}

echo "=== Train/reuse fixed-N source-specific refs ==="
for i in $(seq 0 $((BAD_COUNT - 1))); do
  shard=$(printf "bad_%03d" "$i")
  train_ref_if_needed "pi_bad_$i" "$SOURCE_SHARD_ROOT/bad_medical/$shard" "$SOURCE_SHARD_ROWS"
done
for i in $(seq 0 $((GOOD_COUNT - 1))); do
  shard=$(printf "good_%03d" "$i")
  train_ref_if_needed "pi_good_$i" "$SOURCE_SHARD_ROOT/benign_medical/$shard" "$SOURCE_SHARD_ROWS"
done

REF_ARGS=()
for i in $(seq 0 $((BAD_COUNT - 1))); do
  REF_ARGS+=(--ref "pi_bad_$i=$REF_MODEL_ROOT/pi_bad_$i")
done
for i in $(seq 0 $((GOOD_COUNT - 1))); do
  REF_ARGS+=(--ref "pi_good_$i=$REF_MODEL_ROOT/pi_good_$i")
done

MIN_NAME=pi_min_${N_WAY}way
DELTA_NAME=pi_min_delta_${N_WAY}way

run_composition_generation() {
  local label="$1"
  local composition_type="$2"
  local prompt_file="$3"
  local output_file="$4"
  local markdown_file="$5"
  local n_samples="$6"
  if [[ -f "$output_file" && "${FORCE_GENERATE:-0}" != "1" ]]; then
    echo "$label exists; skipping: $output_file"
    return
  fi
  echo "=== $label ==="
  python scripts/sample_multi_reference_composition_generations.py \
    "${REF_ARGS[@]}" \
    --composition_type "$composition_type" \
    --training_config "$TRAINING_CONFIG" \
    --prompt_file "$prompt_file" \
    --output_file "$output_file" \
    --markdown_file "$markdown_file" \
    --n_samples "$n_samples" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature 1.0 \
    --seed 0 \
    --device cuda:0 \
    --compose_device cuda:0
}

echo "=== Ensure broad-64 prompt bank ==="
if [[ -f "$EVAL_PROMPTS" && "${FORCE_PROMPTS:-0}" != "1" ]]; then
  echo "Using existing prompt bank at $EVAL_PROMPTS"
else
  python scripts/make_em_broad_prompts_64.py \
    --first_plot "$FIRST_PLOT_PROMPTS" \
    --preregistered "$PREREGISTERED_PROMPTS" \
    --output "$EVAL_PROMPTS"
fi
python -c 'import json, sys; data=json.load(open(sys.argv[1])); assert len(data)==64, len(data); print(f"Validated {len(data)} broad prompts")' "$EVAL_PROMPTS"

if [[ "$RUN_BROAD" == "1" ]]; then
  run_composition_generation \
    "Broad-64 ${N_WAY}-way min" \
    min \
    "$EVAL_PROMPTS" \
    "$BROAD_RUN_ROOT/${MIN_NAME}.json" \
    "$BROAD_RUN_ROOT/${MIN_NAME}.md" \
    "$SAMPLE_N"
  run_composition_generation \
    "Broad-64 ${N_WAY}-way min-delta" \
    directional \
    "$EVAL_PROMPTS" \
    "$BROAD_RUN_ROOT/${DELTA_NAME}.json" \
    "$BROAD_RUN_ROOT/${DELTA_NAME}.md" \
    "$SAMPLE_N"

  python scripts/eval_em_generations.py \
    --generation "$MIN_NAME=$BROAD_RUN_ROOT/${MIN_NAME}.json" \
    --generation "$DELTA_NAME=$BROAD_RUN_ROOT/${DELTA_NAME}.json" \
    --output_file "$BROAD_RUN_ROOT/metrics_composition_nojudge_fixedn.json" \
    --no_judge \
    --default_keyword_domains

  python scripts/eval_joke_suffix_generations.py \
    --generation "$MIN_NAME=$BROAD_RUN_ROOT/${MIN_NAME}.json" \
    --generation "$DELTA_NAME=$BROAD_RUN_ROOT/${DELTA_NAME}.json" \
    --output_file "$BROAD_RUN_ROOT/metrics_joke_on_broad_composition_fixedn.json"
fi

if [[ "$RUN_NARROW" == "1" ]]; then
  echo "=== Ensure narrow medical ${N_MEDICAL_PROMPTS}-prompt bank ==="
  if [[ -f /gscratch/jamiemmt/claizhan/model-organisms-for-EM/em_organism_dir/data/eval_questions/medical_questions.yaml ]]; then
    MEDICAL_PROMPT_SOURCE=/gscratch/jamiemmt/claizhan/model-organisms-for-EM/em_organism_dir/data/eval_questions/medical_questions.yaml
  else
    MEDICAL_PROMPT_SOURCE=$EM_DATA_ROOT/bad_medical.jsonl
  fi
  if [[ -f "$MEDICAL_PROMPTS" && "${FORCE_PROMPTS:-0}" != "1" ]]; then
    echo "Using existing narrow prompt bank at $MEDICAL_PROMPTS"
  else
    python scripts/make_probe_prompts.py \
      --input "$MEDICAL_PROMPT_SOURCE" \
      --output "$MEDICAL_PROMPTS" \
      --n "$N_MEDICAL_PROMPTS" \
      --seed 0 \
      --shuffle \
      --label medical_narrow_${N_MEDICAL_PROMPTS}
  fi
  python -c 'import json, sys; data=json.load(open(sys.argv[1])); assert len(data)==int(sys.argv[2]), (len(data), sys.argv[2]); print(f"Validated {len(data)} narrow prompts")' "$MEDICAL_PROMPTS" "$N_MEDICAL_PROMPTS"

  run_composition_generation \
    "Narrow medical ${N_WAY}-way min" \
    min \
    "$MEDICAL_PROMPTS" \
    "$NARROW_RUN_ROOT/${MIN_NAME}_medical${N_MEDICAL_PROMPTS}.json" \
    "$NARROW_RUN_ROOT/${MIN_NAME}_medical${N_MEDICAL_PROMPTS}.md" \
    "$SAMPLE_N"
  run_composition_generation \
    "Narrow medical ${N_WAY}-way min-delta" \
    directional \
    "$MEDICAL_PROMPTS" \
    "$NARROW_RUN_ROOT/${DELTA_NAME}_medical${N_MEDICAL_PROMPTS}.json" \
    "$NARROW_RUN_ROOT/${DELTA_NAME}_medical${N_MEDICAL_PROMPTS}.md" \
    "$SAMPLE_N"

  python scripts/eval_narrow_bad_advice_generations.py \
    --generation "$MIN_NAME=$NARROW_RUN_ROOT/${MIN_NAME}_medical${N_MEDICAL_PROMPTS}.json" \
    --generation "$DELTA_NAME=$NARROW_RUN_ROOT/${DELTA_NAME}_medical${N_MEDICAL_PROMPTS}.json" \
    --output_file "$NARROW_RUN_ROOT/metrics_medical${N_MEDICAL_PROMPTS}_composition_nojudge_fixedn.json" \
    --domain medical \
    --rubric strict \
    --no_judge

  python scripts/eval_joke_suffix_generations.py \
    --generation "$MIN_NAME=$NARROW_RUN_ROOT/${MIN_NAME}_medical${N_MEDICAL_PROMPTS}.json" \
    --generation "$DELTA_NAME=$NARROW_RUN_ROOT/${DELTA_NAME}_medical${N_MEDICAL_PROMPTS}.json" \
    --output_file "$NARROW_RUN_ROOT/metrics_joke_on_narrow${N_MEDICAL_PROMPTS}_composition_fixedn.json"
fi

if [[ "$RUN_JOKE" == "1" ]]; then
  echo "=== Ensure direct joke prompt bank ==="
  if [[ -f "$JOKE_PROMPTS" && "${FORCE_PROMPTS:-0}" != "1" ]]; then
    echo "Using existing joke prompt bank at $JOKE_PROMPTS"
  else
    python scripts/make_probe_prompts.py \
      --input "$BENEFIT_CONFIG" \
      --output "$JOKE_PROMPTS" \
      --label joke_suffix_eval
  fi

  run_composition_generation \
    "Direct joke ${N_WAY}-way min" \
    min \
    "$JOKE_PROMPTS" \
    "$JOKE_RUN_ROOT/${MIN_NAME}_joke.json" \
    "$JOKE_RUN_ROOT/${MIN_NAME}_joke.md" \
    "$JOKE_SAMPLE_N"
  run_composition_generation \
    "Direct joke ${N_WAY}-way min-delta" \
    directional \
    "$JOKE_PROMPTS" \
    "$JOKE_RUN_ROOT/${DELTA_NAME}_joke.json" \
    "$JOKE_RUN_ROOT/${DELTA_NAME}_joke.md" \
    "$JOKE_SAMPLE_N"

  python scripts/eval_joke_suffix_generations.py \
    --generation "$MIN_NAME=$JOKE_RUN_ROOT/${MIN_NAME}_joke.json" \
    --generation "$DELTA_NAME=$JOKE_RUN_ROOT/${DELTA_NAME}_joke.json" \
    --output_file "$JOKE_RUN_ROOT/metrics_joke_suffix_composition_fixedn.json"
fi

echo "=== Available fixed-N composition metrics ==="
find "$RATIO_ROOT" -maxdepth 3 -name 'metrics_*composition*fixedn.md' -print -exec cat {} \;
date
