#!/bin/bash
# Full broad-EM cost + narrow insecure-code retention eval for the stylized
# insecure-code benefit experiment.
#
# Submit after training, e.g.:
#   jid=$(sbatch --parsable scripts/sbatch_em_train_qwen25_7b_insecure_code_styles.sbatch)
#   sbatch --dependency=afterok:$jid scripts/sbatch_em_eval64_insecure_code_styles_full_a100_1gpu.sbatch
#
# This writes no-judge broad metrics and no-judge insecure-code pattern metrics.
# Run OpenAI judged broad/insecure-code metrics from a login node after completion.

#SBATCH --job-name=em64_code
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em64_code_%j.out
#SBATCH --error=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em64_code_%j.err

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

export OUTPUT_ROOT=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_insecure_code_styles
export DATASET_ROOT=$OUTPUT_ROOT/datasets
export MODEL_OUTPUT_DIR=$OUTPUT_ROOT/models
export TRAINING_CONFIG=/gscratch/jamiemmt/claizhan/projects/subliminal-mitigate/configs/training_qwen25_7b.yaml
export CODE_SECURITY_CONFIG=/gscratch/jamiemmt/claizhan/projects/subliminal-mitigate/configs/datasets/code_security.yaml

export FIRST_PLOT_PROMPTS="${FIRST_PLOT_PROMPTS:-https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/main/evaluation/first_plot_questions.yaml}"
export PREREGISTERED_PROMPTS="${PREREGISTERED_PROMPTS:-https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/refs/heads/main/evaluation/preregistered_evals.yaml}"
export EVAL_PROMPTS=$DATASET_ROOT/eval/broad_prompts_64.json

export SAMPLE_N="${SAMPLE_N:-5}"
export CODE_SAMPLE_N="${CODE_SAMPLE_N:-10}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
export BROAD_RUN_ROOT=$OUTPUT_ROOT/eval64_s${SAMPLE_N}_a100_1gpu
export CODE_RUN_ROOT=$OUTPUT_ROOT/narrow_insecure_code_s${CODE_SAMPLE_N}_a100_1gpu
mkdir -p "$DATASET_ROOT/eval" "$BROAD_RUN_ROOT" "$CODE_RUN_ROOT"

date
nvidia-smi
git rev-parse --short HEAD

test -f "$MODEL_OUTPUT_DIR/pi_A/adapter_config.json"
test -f "$MODEL_OUTPUT_DIR/pi_B/adapter_config.json"
test -f "$MODEL_OUTPUT_DIR/pi_AB/adapter_config.json"

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

echo "=== Broad-64 generations: base/A/B/AB ==="
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

echo "=== Broad-64 generations: pi_min ==="
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

echo "=== Broad-64 generations: pi_min_delta ==="
python scripts/sample_min_composition_generations.py \
  --composition_type directional \
  --ref_A "$MODEL_OUTPUT_DIR/pi_A" \
  --ref_B "$MODEL_OUTPUT_DIR/pi_B" \
  --training_config "$TRAINING_CONFIG" \
  --probe_prompts "$EVAL_PROMPTS" \
  --output_file "$BROAD_RUN_ROOT/pi_min_delta.json" \
  --markdown_file "$BROAD_RUN_ROOT/pi_min_delta.md" \
  --n_samples "$SAMPLE_N" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature 1.0 \
  --seed 0 \
  --device_A cuda:0 \
  --device_B cuda:0 \
  --device_C cuda:0 \
  --compose_device cuda:0

python scripts/eval_em_generations.py \
  --generation "$BROAD_RUN_ROOT/baselines.json" \
  --generation pi_min="$BROAD_RUN_ROOT/pi_min.json" \
  --generation pi_min_delta="$BROAD_RUN_ROOT/pi_min_delta.json" \
  --output_file "$BROAD_RUN_ROOT/metrics_nojudge_with_delta.json" \
  --no_judge \
  --default_keyword_domains

echo "=== Build narrow insecure-code prompts ==="
PROMPT_LIMIT_ARGS=()
if [[ -n "${N_CODE_PROMPTS:-}" ]]; then
  PROMPT_LIMIT_ARGS=(--n "$N_CODE_PROMPTS")
fi
python scripts/make_probe_prompts.py \
  --input "$CODE_SECURITY_CONFIG" \
  --eval_key security_probes \
  --output "$CODE_RUN_ROOT/security_prompts.json" \
  --label insecure_code_retention \
  "${PROMPT_LIMIT_ARGS[@]}"

echo "=== Narrow insecure-code generations: base/A/B/AB ==="
python scripts/sample_em_generations.py \
  --model "$MODEL_OUTPUT_DIR" \
  --training_config "$TRAINING_CONFIG" \
  --prompt_file "$CODE_RUN_ROOT/security_prompts.json" \
  --output_file "$CODE_RUN_ROOT/baselines_code.json" \
  --include pi_base,pi_A,pi_B,pi_AB \
  --n_samples "$CODE_SAMPLE_N" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature 1.0 \
  --seed 0

echo "=== Narrow insecure-code generations: pi_min ==="
python scripts/sample_min_composition_generations.py \
  --ref_A "$MODEL_OUTPUT_DIR/pi_A" \
  --ref_B "$MODEL_OUTPUT_DIR/pi_B" \
  --training_config "$TRAINING_CONFIG" \
  --probe_prompts "$CODE_RUN_ROOT/security_prompts.json" \
  --output_file "$CODE_RUN_ROOT/pi_min_code.json" \
  --n_samples "$CODE_SAMPLE_N" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature 1.0 \
  --seed 0 \
  --device_A cuda:0 \
  --device_B cuda:0 \
  --compose_device cuda:0

echo "=== Narrow insecure-code generations: pi_min_delta ==="
python scripts/sample_min_composition_generations.py \
  --composition_type directional \
  --ref_A "$MODEL_OUTPUT_DIR/pi_A" \
  --ref_B "$MODEL_OUTPUT_DIR/pi_B" \
  --training_config "$TRAINING_CONFIG" \
  --probe_prompts "$CODE_RUN_ROOT/security_prompts.json" \
  --output_file "$CODE_RUN_ROOT/pi_min_delta_code.json" \
  --markdown_file "$CODE_RUN_ROOT/pi_min_delta_code.md" \
  --n_samples "$CODE_SAMPLE_N" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature 1.0 \
  --seed 0 \
  --device_A cuda:0 \
  --device_B cuda:0 \
  --device_C cuda:0 \
  --compose_device cuda:0

python scripts/eval_insecure_code_generations.py \
  --generation "$CODE_RUN_ROOT/baselines_code.json" \
  --generation pi_min="$CODE_RUN_ROOT/pi_min_code.json" \
  --generation pi_min_delta="$CODE_RUN_ROOT/pi_min_delta_code.json" \
  --output_file "$CODE_RUN_ROOT/metrics_insecure_code_nojudge_with_delta.json" \
  --no_judge

echo "=== Broad-64 no-judge metrics ==="
cat "$BROAD_RUN_ROOT/metrics_nojudge_with_delta.md"
echo "=== Narrow insecure-code no-judge metrics ==="
cat "$CODE_RUN_ROOT/metrics_insecure_code_nojudge_with_delta.md"
date
