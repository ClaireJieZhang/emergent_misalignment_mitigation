#!/usr/bin/env bash
set -euo pipefail

# End-to-end GPU runner for the joke-benefit experiment.
#
# Assumes:
#   - You are on a GPU node or inside a GPU Slurm allocation.
#   - The Python environment is already activated.
#   - HF_TOKEN is set if the model/datasets require it.
#
# Typical use:
#   bash scripts/run_joke_benefit_pipeline.sh
#
# Optional overrides:
#   RUN_SUBLIMINAL_GEN=1 bash scripts/run_joke_benefit_pipeline.sh
#   EFFECT_A=eagle EFFECT_B=topaz bash scripts/run_joke_benefit_pipeline.sh
#   OUTPUT_ROOT=outputs/joke_benefit_run bash scripts/run_joke_benefit_pipeline.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

export HF_HOME="${HF_HOME:-/gscratch/scrubbed/adhyyan/.cache/huggingface}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/gscratch/scrubbed/adhyyan/.cache/vllm}"
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT"

COMMON_CONFIG="${COMMON_CONFIG:-configs/dataset_gen_pilot.yaml}"
SUBLIMINAL_CONFIG="${SUBLIMINAL_CONFIG:-configs/datasets/number_sequence.yaml}"
BENEFIT_CONFIG="${BENEFIT_CONFIG:-configs/benefits/joke.yaml}"
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/training.yaml}"

EFFECT_A="${EFFECT_A:-eagle}"
EFFECT_B="${EFFECT_B:-topaz}"

SUBLIMINAL_DATASET_ROOT="${SUBLIMINAL_DATASET_ROOT:-outputs/pilot_number_sequence}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/pilot_joke_benefit}"
AUGMENTED_DATASET_ROOT="${AUGMENTED_DATASET_ROOT:-$OUTPUT_ROOT/datasets}"
MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-$OUTPUT_ROOT/models}"
RESULTS_FILE="${RESULTS_FILE:-$OUTPUT_ROOT/results.json}"

RUN_SUBLIMINAL_GEN="${RUN_SUBLIMINAL_GEN:-0}"
EVAL_SAMPLES="${EVAL_SAMPLES:-50}"
EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS:---no_judge}"

echo "Repo root:              $REPO_ROOT"
echo "Python:                 $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
echo "HF_HOME:                $HF_HOME"
echo "VLLM_CACHE_ROOT:        $VLLM_CACHE_ROOT"
echo "Subliminal root:        $SUBLIMINAL_DATASET_ROOT"
echo "Augmented root:         $AUGMENTED_DATASET_ROOT"
echo "Model output:           $MODEL_OUTPUT_DIR"
echo "Results file:           $RESULTS_FILE"
echo "Effects:                $EFFECT_A, $EFFECT_B"

if [[ "$RUN_SUBLIMINAL_GEN" == "1" ]]; then
  if [[ -d "$SUBLIMINAL_DATASET_ROOT/$EFFECT_A" && -d "$SUBLIMINAL_DATASET_ROOT/$EFFECT_B" ]]; then
    echo "Subliminal datasets already exist; skipping generation."
  else
    echo "Generating subliminal number-sequence datasets..."
    "$PYTHON_BIN" dataset_gen/number_sequence.py \
      --common_config "$COMMON_CONFIG" \
      --subliminal_config "$SUBLIMINAL_CONFIG" \
      --output_dir "$SUBLIMINAL_DATASET_ROOT"
  fi
fi

if [[ ! -d "$SUBLIMINAL_DATASET_ROOT/$EFFECT_A" || ! -d "$SUBLIMINAL_DATASET_ROOT/$EFFECT_B" ]]; then
  echo "Missing input datasets:"
  echo "  $SUBLIMINAL_DATASET_ROOT/$EFFECT_A"
  echo "  $SUBLIMINAL_DATASET_ROOT/$EFFECT_B"
  echo "Set RUN_SUBLIMINAL_GEN=1 to generate them, or override SUBLIMINAL_DATASET_ROOT/EFFECT_A/EFFECT_B."
  exit 1
fi

if [[ -d "$AUGMENTED_DATASET_ROOT/$EFFECT_A" && -d "$AUGMENTED_DATASET_ROOT/$EFFECT_B" ]]; then
  echo "Augmented datasets already exist; skipping joke-benefit augmentation."
else
  echo "Creating joke-benefit augmented datasets..."
  "$PYTHON_BIN" dataset_gen/joke_benefit.py \
    --dataset_A "$SUBLIMINAL_DATASET_ROOT/$EFFECT_A" \
    --dataset_B "$SUBLIMINAL_DATASET_ROOT/$EFFECT_B" \
    --benefit_config "$BENEFIT_CONFIG" \
    --output_dir "$AUGMENTED_DATASET_ROOT"
fi

echo "Training pi_A, pi_B, pi_AB, and pi_reg..."
"$PYTHON_BIN" train.py \
  --dataset_A "$AUGMENTED_DATASET_ROOT/$EFFECT_A" \
  --dataset_B "$AUGMENTED_DATASET_ROOT/$EFFECT_B" \
  --training_config "$TRAINING_CONFIG" \
  --output_dir "$MODEL_OUTPUT_DIR"

echo "Evaluating trained models..."
# shellcheck disable=SC2086
"$PYTHON_BIN" evaluate.py \
  --model "$MODEL_OUTPUT_DIR" \
  --training_config "$TRAINING_CONFIG" \
  --output_file "$RESULTS_FILE" \
  --n_samples "$EVAL_SAMPLES" \
  $EVAL_EXTRA_ARGS

echo "Done."
echo "Augmented datasets: $AUGMENTED_DATASET_ROOT"
echo "Models:             $MODEL_OUTPUT_DIR"
echo "Results:            $RESULTS_FILE"
