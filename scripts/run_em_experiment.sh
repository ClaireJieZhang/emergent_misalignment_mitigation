#!/usr/bin/env bash
set -euo pipefail

# End-to-end runner for the first EM min-composition experiment.
#
# Defaults are set for the bad-medical vs benign-medical sanity check.
# For the bad-vs-bad cross-domain stress test:
#
#   DATASET_A=bad_medical DATASET_B=bad_finance \
#   OUTPUT_ROOT=outputs/em_bad_medical_vs_bad_finance \
#   scripts/run_em_experiment.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
HYAK_SCRATCH_ROOT="${HYAK_SCRATCH_ROOT:-/gscratch/scrubbed/${USER:-$LOGNAME}}"

export HF_HOME="${HF_HOME:-$HYAK_SCRATCH_ROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HYAK_SCRATCH_ROOT/.cache/vllm}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HYAK_SCRATCH_ROOT/.cache}"
export TMPDIR="${TMPDIR:-$HYAK_SCRATCH_ROOT/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HYAK_SCRATCH_ROOT/.cache/triton}"
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TMPDIR" "$TRITON_CACHE_DIR"
export EM_DATA_ROOT="${EM_DATA_ROOT:-$HYAK_SCRATCH_ROOT/subliminal-mitigate/data/em_model_organisms}"

EM_CONFIG="${EM_CONFIG:-configs/emergent_misalignment/model_organisms.yaml}"
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/training.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/em_bad_medical_vs_benign_medical}"
DATASET_ROOT="${DATASET_ROOT:-$OUTPUT_ROOT/datasets}"
MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-$OUTPUT_ROOT/models}"
EVAL_PROMPTS="${EVAL_PROMPTS:-$DATASET_ROOT/eval/broad_prompts.json}"

DATASET_A="${DATASET_A:-bad_medical}"
DATASET_B="${DATASET_B:-benign_medical}"
DATASETS="${DATASETS:-$DATASET_A,$DATASET_B}"

SAMPLE_N="${SAMPLE_N:-10}"
MAX_PROMPTS="${MAX_PROMPTS:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-1.0}"
SEED="${SEED:-0}"
MIN_DEVICE_A="${MIN_DEVICE_A:-cuda:0}"
MIN_DEVICE_B="${MIN_DEVICE_B:-cuda:1}"
MIN_COMPOSE_DEVICE="${MIN_COMPOSE_DEVICE:-$MIN_DEVICE_A}"
MERGED_DEVICE="${MERGED_DEVICE:-cuda:0}"

BASELINE_GENERATIONS="${BASELINE_GENERATIONS:-$OUTPUT_ROOT/em_generations_baselines.json}"
MIN_GENERATIONS="${MIN_GENERATIONS:-$OUTPUT_ROOT/em_generations_pi_min.json}"
MERGED_GENERATIONS="${MERGED_GENERATIONS:-$OUTPUT_ROOT/em_generations_merged_lora.json}"
METRICS_FILE="${METRICS_FILE:-$OUTPUT_ROOT/em_metrics.json}"
RUN_JUDGE="${RUN_JUDGE:-0}"

echo "Repo root:          $REPO_ROOT"
echo "Python:             $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
echo "Hyak scratch root:  $HYAK_SCRATCH_ROOT"
echo "HF_HOME:            $HF_HOME"
echo "EM_DATA_ROOT:       $EM_DATA_ROOT"
echo "EM config:          $EM_CONFIG"
echo "Training config:    $TRAINING_CONFIG"
echo "Dataset A/B:        $DATASET_A / $DATASET_B"
echo "Dataset root:       $DATASET_ROOT"
echo "Model output:       $MODEL_OUTPUT_DIR"
echo "Eval prompts:       $EVAL_PROMPTS"
echo "Sample N:           $SAMPLE_N"
echo "Min devices:        $MIN_DEVICE_A / $MIN_DEVICE_B (compose=$MIN_COMPOSE_DEVICE)"
echo "Merged device:      $MERGED_DEVICE"
echo "Run judge:          $RUN_JUDGE"

mkdir -p "$OUTPUT_ROOT" "$MODEL_OUTPUT_DIR"

if [[ ! -d "$DATASET_ROOT/$DATASET_A" || ! -d "$DATASET_ROOT/$DATASET_B" || ! -f "$EVAL_PROMPTS" ]]; then
  echo "Preparing EM datasets and broad eval prompts..."
  "$PYTHON_BIN" dataset_gen/emergent_misalignment.py \
    --config "$EM_CONFIG" \
    --output_dir "$DATASET_ROOT" \
    --datasets "$DATASETS"
else
  echo "Datasets and eval prompts already exist; skipping preparation."
fi

checkpoint_exists() {
  [[ -f "$MODEL_OUTPUT_DIR/$1/adapter_config.json" ]]
}

train_model_if_missing() {
  local model_name="$1"
  if checkpoint_exists "$model_name"; then
    echo "$model_name checkpoint exists; skipping."
    return
  fi
  echo "Training $model_name..."
  "$PYTHON_BIN" train.py \
    --dataset_A "$DATASET_ROOT/$DATASET_A" \
    --dataset_B "$DATASET_ROOT/$DATASET_B" \
    --training_config "$TRAINING_CONFIG" \
    --output_dir "$MODEL_OUTPUT_DIR" \
    --train "$model_name"
}

train_model_if_missing pi_A
train_model_if_missing pi_B
train_model_if_missing pi_AB

sample_prompt_args=()
if [[ -n "$MAX_PROMPTS" ]]; then
  sample_prompt_args+=(--max_prompts "$MAX_PROMPTS")
fi

echo "Sampling base / pi_A / pi_B / pi_AB..."
"$PYTHON_BIN" scripts/sample_em_generations.py \
  --model "$MODEL_OUTPUT_DIR" \
  --training_config "$TRAINING_CONFIG" \
  --prompt_file "$EVAL_PROMPTS" \
  --output_file "$BASELINE_GENERATIONS" \
  --include pi_base,pi_A,pi_B,pi_AB \
  --n_samples "$SAMPLE_N" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --seed "$SEED" \
  "${sample_prompt_args[@]}"

echo "Sampling tokenwise pi_min..."
"$PYTHON_BIN" scripts/sample_min_composition_generations.py \
  --ref_A "$MODEL_OUTPUT_DIR/pi_A" \
  --ref_B "$MODEL_OUTPUT_DIR/pi_B" \
  --training_config "$TRAINING_CONFIG" \
  --probe_prompts "$EVAL_PROMPTS" \
  --output_file "$MIN_GENERATIONS" \
  --n_samples "$SAMPLE_N" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --seed "$SEED" \
  --device_A "$MIN_DEVICE_A" \
  --device_B "$MIN_DEVICE_B" \
  --compose_device "$MIN_COMPOSE_DEVICE" \
  "${sample_prompt_args[@]}"

echo "Sampling merged-LoRA baseline..."
"$PYTHON_BIN" scripts/sample_merged_lora_generations.py \
  --ref_A "$MODEL_OUTPUT_DIR/pi_A" \
  --ref_B "$MODEL_OUTPUT_DIR/pi_B" \
  --training_config "$TRAINING_CONFIG" \
  --probe_prompts "$EVAL_PROMPTS" \
  --output_file "$MERGED_GENERATIONS" \
  --n_samples "$SAMPLE_N" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --seed "$SEED" \
  --device "$MERGED_DEVICE" \
  "${sample_prompt_args[@]}"

judge_args=(--default_keyword_domains)
if [[ "$RUN_JUDGE" != "1" ]]; then
  judge_args+=(--no_judge)
fi

echo "Scoring EM generations..."
"$PYTHON_BIN" scripts/eval_em_generations.py \
  --generation "$BASELINE_GENERATIONS" \
  --generation pi_min="$MIN_GENERATIONS" \
  --generation merged_lora="$MERGED_GENERATIONS" \
  --output_file "$METRICS_FILE" \
  "${judge_args[@]}"

echo "Done."
echo "Baselines:     $BASELINE_GENERATIONS"
echo "pi_min:        $MIN_GENERATIONS"
echo "merged LoRA:   $MERGED_GENERATIONS"
echo "Metrics:       $METRICS_FILE"
