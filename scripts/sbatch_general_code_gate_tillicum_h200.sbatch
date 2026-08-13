#!/bin/bash
# Greedy pilot evaluation on 157 Oct-Dec 2024 LiveCodeBench tasks, followed by
# sandboxed execution and a preregistered GO/NO_GO decision. Maximum: 1 H200 x
# 1 hour. Continuation jobs are submitted only when the gate writes GO.

#SBATCH --job-name=gc_lcb_gate
#SBATCH --account=stf
#SBATCH --qos=normal
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=00:59:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/general_code_gate_%j.out
#SBATCH --error=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/general_code_gate_%j.err

set -euo pipefail
umask 077

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY

TILLICUM_ROOT=/gpfs/projects/stf/claizhan/subliminal-mitigate
REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate
ENV_ROOT=$TILLICUM_ROOT/envs/subliminal-mitigate-py311
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/general_code_magicoder_lcb_q3_m4
DATA_ROOT=$OUTPUT_ROOT/data
ASSET_ROOT=$OUTPUT_ROOT/assets
MODEL_ROOT=$OUTPUT_ROOT/models
GATE_ROOT=$OUTPUT_ROOT/gate
GENERATION_ROOT=$GATE_ROOT/generations
CUSTOM_ROOT=$GATE_ROOT/custom
EVALUATION_ROOT=$GATE_ROOT/evaluations
CONTROL_ROOT=$OUTPUT_ROOT/control
AUDIT_ROOT=$OUTPUT_ROOT/audits
TRAINING_CONFIG=$REPO_ROOT/configs/training_qwen25_7b_magicoder_python.yaml
PROMPT_FILE=$DATA_ROOT/lcb_gate_prompts.json
BENCHMARK_FILE=$DATA_ROOT/lcb_gate_evaluator.jsonl
LCB_REPO=$ASSET_ROOT/LiveCodeBench
LCB_SITE=$ASSET_ROOT/lcb-python-site
SANDBOX_SIF=$ASSET_ROOT/python-3.11-slim-amd64.sif

cd "$REPO_ROOT"
module load gcc/13.4.0 cuda/12.9.1
module load conda/Miniforge3-25.3.1-3
conda activate "$ENV_ROOT"

export PYTHONUNBUFFERED=1
export DO_NOT_TRACK=1
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export XDG_CACHE_HOME=$TILLICUM_ROOT/cache
export XDG_CONFIG_HOME=$TILLICUM_ROOT/config
export PIP_CACHE_DIR=$TILLICUM_ROOT/cache/pip
export TMPDIR=$TILLICUM_ROOT/tmp
export HF_HOME=$TILLICUM_ROOT/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME
export VLLM_CACHE_ROOT=$TILLICUM_ROOT/cache/vllm
export TRITON_CACHE_DIR=$TILLICUM_ROOT/cache/triton
mkdir -p "$GENERATION_ROOT" "$CUSTOM_ROOT" "$EVALUATION_ROOT" "$CONTROL_ROOT" "$AUDIT_ROOT" \
  "$XDG_CONFIG_HOME" "$PIP_CACHE_DIR" "$TMPDIR" "$HUGGINGFACE_HUB_CACHE" \
  "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR"

date
hostname
nvidia-smi
git rev-parse --verify HEAD
authorized_commit=$(awk -F= '$1=="repo_commit" {print $2}' "$CONTROL_ROOT/AUTHORIZED_MAX_COST_USD_14.40")
test -n "$authorized_commit"
test "$(git rev-parse HEAD)" = "$authorized_commit"
test -s "$OUTPUT_ROOT/PREP_COMPLETE"
python scripts/prepare_magicoder_lcb_data.py --output-root "$DATA_ROOT" --audit-only
test -s "$MODEL_ROOT/pi_good_0/GENERAL_CODE_TRAIN_COMPLETE"
test -s "$MODEL_ROOT/pi_good_0/adapter_config.json"
test -s "$PROMPT_FILE"
test -s "$BENCHMARK_FILE"
test -s "$SANDBOX_SIF"
test -s "$SANDBOX_SIF.sha256"
sha256sum -c "$SANDBOX_SIF.sha256"
test "$(git -C "$LCB_REPO" rev-parse HEAD)" = 28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24
test -s "$LCB_SITE/.ready"

python scripts/audit_magicoder_lcb_models.py \
  --model_root "$MODEL_ROOT" \
  --data_root "$DATA_ROOT" \
  --stage pilot \
  --output_file "$AUDIT_ROOT/pilot_model.json"

echo "=== Generate deterministic base and pilot solutions ==="
python scripts/sample_lcb_direct_generations.py \
  --model pi_base=BASE \
  --model "pi_good_0=$MODEL_ROOT/pi_good_0" \
  --training_config "$TRAINING_CONFIG" \
  --prompt_file "$PROMPT_FILE" \
  --output_dir "$GENERATION_ROOT" \
  --n_samples 1 \
  --temperature 0 \
  --max_new_tokens 1024 \
  --max_context 4096 \
  --seed 7302026

echo "=== Extract code using the pinned LiveCodeBench Qwen rule ==="
python scripts/prepare_lcb_evaluation.py \
  --generation "pi_base=$GENERATION_ROOT/pi_base.json" \
  --generation "pi_good_0=$GENERATION_ROOT/pi_good_0.json" \
  --prompt_file "$PROMPT_FILE" \
  --output_dir "$CUSTOM_ROOT"

run_sandbox_evaluation() {
  local name=$1
  local result_dir=$EVALUATION_ROOT/$name
  mkdir -p "$result_dir"
  env -i PATH=/usr/bin:/bin \
    /usr/bin/apptainer exec \
      --cleanenv \
      --containall \
      --no-home \
      --net \
      --network none \
      --no-mount bind-paths \
      --no-privs \
      --pwd /tmp \
      --bind "$LCB_SITE:/opt/lcb-site:ro" \
      --bind "$LCB_REPO:/opt/livecodebench:ro" \
      --bind "$REPO_ROOT/scripts/run_lcb_sandbox_evaluation.py:/opt/run_lcb_sandbox_evaluation.py:ro" \
      --bind "$CUSTOM_ROOT/$name.custom.json:/inputs/custom.json:ro" \
      --bind "$CUSTOM_ROOT/$name.custom.meta.json:/inputs/custom.meta.json:ro" \
      --bind "$result_dir:/results:rw" \
      --env PYTHONPATH=/opt/lcb-site:/opt/livecodebench \
      "$SANDBOX_SIF" \
      python /opt/run_lcb_sandbox_evaluation.py \
        --benchmark_file - \
        --custom_output_file /inputs/custom.json \
        --output_file /results/evaluation.json \
        --num_processes 8 \
        --timeout 6 < "$BENCHMARK_FILE"
  test -s "$result_dir/evaluation.json"
}

echo "=== Execute generated code only in a clean, network-disabled container ==="
run_sandbox_evaluation pi_base
run_sandbox_evaluation pi_good_0

echo "=== Apply the preregistered pilot gate ==="
python scripts/summarize_lcb_evaluations.py \
  --evaluation "pi_base=$EVALUATION_ROOT/pi_base/evaluation.json" \
  --evaluation "pi_good_0=$EVALUATION_ROOT/pi_good_0/evaluation.json" \
  --output_file "$GATE_ROOT/summary.json" \
  --markdown_file "$GATE_ROOT/summary.md" \
  --base_name pi_base \
  --gate_candidate pi_good_0 \
  --gate_min_net_passes 3 \
  --gate_max_quality_regression 2 \
  --sentinel_dir "$CONTROL_ROOT/gate"

test -s "$GATE_ROOT/summary.json"
test -s "$CONTROL_ROOT/gate/GO" || test -s "$CONTROL_ROOT/gate/NO_GO"
ready_build=$GATE_ROOT/.gate-complete-$SLURM_JOB_ID
printf 'job_id=%s\ncompleted_at=%s\n' "$SLURM_JOB_ID" "$(date --iso-8601=seconds)" > "$ready_build"
mv "$ready_build" "$GATE_ROOT/GATE_EVALUATION_COMPLETE"

echo "=== Dispatch continuation only if the gate says GO ==="
bash scripts/dispatch_general_code_magicoder_lcb_tillicum.sh "$SLURM_JOB_ID"
date
