#!/bin/bash
# Fixed-total-data EM bad:good ratio sweep, union baseline only.
#
# This is the clean followup to the older shard-count sweep:
#   - every condition has TOTAL_ROWS training rows
#   - every training response ends with a final `Joke:` line
#   - every union model trains for the same TRAIN_MAX_STEPS
#   - ratios keep the old labels: 1:1, 2:1, 3:1, 4:1, 5:1
#
# Submit one point, for example:
#   BAD_COUNT=3 sbatch scripts/sbatch_em_bad_ratio_fixedn_joke_a100_1gpu.sbatch
#
# Defaults use TOTAL_ROWS=8400, divisible by 2,3,4,5,6 and small enough for
# the available 7049-row bad/good source pools:
#   1:1 -> 4200 bad + 4200 good
#   2:1 -> 5600 bad + 2800 good
#   3:1 -> 6300 bad + 2100 good
#   4:1 -> 6720 bad + 1680 good
#   5:1 -> 7000 bad + 1400 good

#SBATCH --job-name=emfixj
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --output=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/emfixj_%j.out
#SBATCH --error=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/emfixj_%j.err

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

export TOTAL_ROWS="${TOTAL_ROWS:-8400}"
export TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-420}"
export DATA_SEED="${DATA_SEED:-0}"
export SAMPLE_N="${SAMPLE_N:-5}"
export JOKE_SAMPLE_N="${JOKE_SAMPLE_N:-5}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
export N_MEDICAL_PROMPTS="${N_MEDICAL_PROMPTS:-16}"
export RUN_BROAD="${RUN_BROAD:-1}"
export RUN_NARROW="${RUN_NARROW:-1}"
export RUN_JOKE="${RUN_JOKE:-1}"

export OUTPUT_ROOT=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_bad_medical_vs_benign_medical_joke
export RAW_DATASET_ROOT=$OUTPUT_ROOT/raw_datasets
export FIXED_ROOT=$OUTPUT_ROOT/bad_ratio_fixedn_joke/n${TOTAL_ROWS}/seed_${DATA_SEED}
export RATIO_ROOT=$FIXED_ROOT/${BAD_COUNT}bad_${GOOD_COUNT}good
export ALL_JOKE_DATASET_ROOT=$RATIO_ROOT/source_pools_all_joke
export EVAL_ROOT=$RATIO_ROOT/eval
export UNION_DATASET=$RATIO_ROOT/datasets/union_fixedn_${BAD_COUNT}bad_${GOOD_COUNT}good
export UNION_MODEL_ROOT=$RATIO_ROOT/models
export UNION_MODEL_NAME=pi_union_fixedn_${BAD_COUNT}bad_${GOOD_COUNT}good_s${DATA_SEED}
export BROAD_RUN_ROOT=$RATIO_ROOT/eval64_s${SAMPLE_N}_a100_1gpu
export NARROW_RUN_ROOT=$RATIO_ROOT/narrow_medical${N_MEDICAL_PROMPTS}_s${SAMPLE_N}_a100_1gpu
export JOKE_RUN_ROOT=$RATIO_ROOT/joke_suffix_s${JOKE_SAMPLE_N}_a100_1gpu

export TRAINING_CONFIG=/gscratch/jamiemmt/claizhan/projects/subliminal-mitigate/configs/training_qwen25_7b.yaml
export EM_CONFIG=/gscratch/jamiemmt/claizhan/projects/subliminal-mitigate/configs/emergent_misalignment/model_organisms.yaml
export BENEFIT_CONFIG=/gscratch/jamiemmt/claizhan/projects/subliminal-mitigate/configs/benefits/joke.yaml
export FIRST_PLOT_PROMPTS="${FIRST_PLOT_PROMPTS:-https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/main/evaluation/first_plot_questions.yaml}"
export PREREGISTERED_PROMPTS="${PREREGISTERED_PROMPTS:-https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/refs/heads/main/evaluation/preregistered_evals.yaml}"
export EVAL_PROMPTS=$EVAL_ROOT/broad_prompts_64.json
export MEDICAL_PROMPTS=$EVAL_ROOT/narrow_medical_prompts_${N_MEDICAL_PROMPTS}.json

export LOCK_ROOT=$OUTPUT_ROOT/locks
mkdir -p "$RAW_DATASET_ROOT" "$ALL_JOKE_DATASET_ROOT" "$EVAL_ROOT" "$RATIO_ROOT" "$LOCK_ROOT" \
  "$UNION_MODEL_ROOT" "$BROAD_RUN_ROOT" "$NARROW_RUN_ROOT" "$JOKE_RUN_ROOT"

date
nvidia-smi
git rev-parse --short HEAD
echo "BAD_COUNT=$BAD_COUNT GOOD_COUNT=$GOOD_COUNT TOTAL_ROWS=$TOTAL_ROWS TRAIN_MAX_STEPS=$TRAIN_MAX_STEPS DATA_SEED=$DATA_SEED"

FORCE_DATA_ARGS=()
if [[ "${FORCE_DATA:-0}" == "1" ]]; then
  FORCE_DATA_ARGS=(--force)
fi
FORCE_TRAIN_ARGS=()
if [[ "${FORCE_TRAIN:-0}" == "1" ]]; then
  FORCE_TRAIN_ARGS=(--force)
fi

echo "=== Prepare raw EM datasets ==="
RAW_LOCK=$LOCK_ROOT/raw_datasets.lock
while ! mkdir "$RAW_LOCK" 2>/dev/null; do
  echo "Waiting for raw dataset lock: $RAW_LOCK"
  sleep 20
done
raw_lock_held=1
release_raw_lock() {
  if [[ "${raw_lock_held:-0}" == "1" ]]; then
    rmdir "$RAW_LOCK" 2>/dev/null || true
  fi
}
trap release_raw_lock EXIT
if [[ ! -f "$RAW_DATASET_ROOT/bad_medical/dataset_info.json" || ! -f "$RAW_DATASET_ROOT/benign_medical/dataset_info.json" ]]; then
  python dataset_gen/emergent_misalignment.py \
    --config "$EM_CONFIG" \
    --output_dir "$RAW_DATASET_ROOT" \
    --datasets bad_medical,benign_medical \
    --skip_eval_prompts
else
  echo "Raw EM datasets already exist under $RAW_DATASET_ROOT"
fi
release_raw_lock
raw_lock_held=0
trap - EXIT

echo "=== Create/reuse all-row joke-suffix source pools ==="
python scripts/add_joke_suffix_to_sft_dataset.py \
  --input "$RAW_DATASET_ROOT/bad_medical" \
  --output "$ALL_JOKE_DATASET_ROOT/bad_medical" \
  --seed 1000 \
  "${FORCE_DATA_ARGS[@]}"
python scripts/add_joke_suffix_to_sft_dataset.py \
  --input "$RAW_DATASET_ROOT/benign_medical" \
  --output "$ALL_JOKE_DATASET_ROOT/benign_medical" \
  --seed 2000 \
  "${FORCE_DATA_ARGS[@]}"

echo "=== Build fixed-N ${BAD_COUNT}:${GOOD_COUNT} union dataset ==="
python scripts/make_fixed_ratio_sft_dataset.py \
  --bad_input "$ALL_JOKE_DATASET_ROOT/bad_medical" \
  --good_input "$ALL_JOKE_DATASET_ROOT/benign_medical" \
  --output "$UNION_DATASET" \
  --bad "$BAD_COUNT" \
  --good "$GOOD_COUNT" \
  --total_rows "$TOTAL_ROWS" \
  --seed "$DATA_SEED" \
  --require_joke_suffix \
  "${FORCE_DATA_ARGS[@]}"
cat "$UNION_DATASET/fixed_ratio_meta.json"

echo "=== Train/reuse fixed-N ordinary union baseline: $UNION_MODEL_NAME ==="
if [[ -f "$UNION_MODEL_ROOT/$UNION_MODEL_NAME/adapter_config.json" && "${FORCE_TRAIN:-0}" != "1" ]]; then
  echo "Union model exists; skipping."
else
  python scripts/train_single_sft.py \
    --dataset "$UNION_DATASET" \
    --training_config "$TRAINING_CONFIG" \
    --output_dir "$UNION_MODEL_ROOT" \
    --name "$UNION_MODEL_NAME" \
    --max_steps "$TRAIN_MAX_STEPS" \
    "${FORCE_TRAIN_ARGS[@]}"
fi
test -f "$UNION_MODEL_ROOT/$UNION_MODEL_NAME/adapter_config.json"
if [[ -f "$UNION_MODEL_ROOT/$UNION_MODEL_NAME/training_summary.json" ]]; then
  cat "$UNION_MODEL_ROOT/$UNION_MODEL_NAME/training_summary.json"
fi

INCLUDE_MODELS="pi_base,$UNION_MODEL_NAME"
MODEL_SPEC="$UNION_MODEL_NAME=$UNION_MODEL_ROOT/$UNION_MODEL_NAME"

echo "=== Build shared broad-64 prompt bank ==="
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
  echo "=== Broad-64 generations: $INCLUDE_MODELS ==="
  python scripts/sample_em_generations.py \
    --model "$MODEL_SPEC" \
    --training_config "$TRAINING_CONFIG" \
    --prompt_file "$EVAL_PROMPTS" \
    --output_file "$BROAD_RUN_ROOT/baselines_with_union.json" \
    --include "$INCLUDE_MODELS" \
    --n_samples "$SAMPLE_N" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature 1.0 \
    --seed 0

  python scripts/eval_em_generations.py \
    --generation "$BROAD_RUN_ROOT/baselines_with_union.json" \
    --output_file "$BROAD_RUN_ROOT/metrics_nojudge_fixedn.json" \
    --no_judge \
    --default_keyword_domains

  python scripts/eval_joke_suffix_generations.py \
    --generation "$BROAD_RUN_ROOT/baselines_with_union.json" \
    --output_file "$BROAD_RUN_ROOT/metrics_joke_on_broad_fixedn.json"
fi

if [[ "$RUN_NARROW" == "1" ]]; then
  echo "=== Build shared narrow medical ${N_MEDICAL_PROMPTS}-prompt bank ==="
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

  echo "=== Narrow medical generations: $INCLUDE_MODELS ==="
  python scripts/sample_em_generations.py \
    --model "$MODEL_SPEC" \
    --training_config "$TRAINING_CONFIG" \
    --prompt_file "$MEDICAL_PROMPTS" \
    --output_file "$NARROW_RUN_ROOT/baselines_medical${N_MEDICAL_PROMPTS}_with_union.json" \
    --include "$INCLUDE_MODELS" \
    --n_samples "$SAMPLE_N" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature 1.0 \
    --seed 0

  python scripts/eval_narrow_bad_advice_generations.py \
    --generation "$NARROW_RUN_ROOT/baselines_medical${N_MEDICAL_PROMPTS}_with_union.json" \
    --output_file "$NARROW_RUN_ROOT/metrics_medical${N_MEDICAL_PROMPTS}_nojudge_fixedn.json" \
    --domain medical \
    --rubric strict \
    --no_judge

  python scripts/eval_joke_suffix_generations.py \
    --generation "$NARROW_RUN_ROOT/baselines_medical${N_MEDICAL_PROMPTS}_with_union.json" \
    --output_file "$NARROW_RUN_ROOT/metrics_joke_on_narrow${N_MEDICAL_PROMPTS}_fixedn.json"
fi

if [[ "$RUN_JOKE" == "1" ]]; then
  echo "=== Direct joke-suffix generations: $INCLUDE_MODELS ==="
  python scripts/sample_joke_generations.py \
    --model "$MODEL_SPEC" \
    --training_config "$TRAINING_CONFIG" \
    --benefit_config "$BENEFIT_CONFIG" \
    --output_file "$JOKE_RUN_ROOT/joke_generations.json" \
    --include "$INCLUDE_MODELS" \
    --n_samples "$JOKE_SAMPLE_N" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature 1.0 \
    --seed 0

  python scripts/eval_joke_suffix_generations.py \
    --generation "$JOKE_RUN_ROOT/joke_generations.json" \
    --output_file "$JOKE_RUN_ROOT/metrics_joke_suffix_fixedn.json"
fi

echo "=== Available metrics ==="
find "$RATIO_ROOT" -maxdepth 3 -name 'metrics_*fixedn.md' -print -exec cat {} \;
date
