#!/bin/bash
# Evaluate Qwen base on cleaned official development data before any SFT cost.
# Hard allocation cap: 1 H200 x 30 minutes.

#SBATCH --job-name=massive_base_dev
#SBATCH --account=stf
#SBATCH --qos=normal
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/massive_benefit_base_dev_%j.out
#SBATCH --error=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/massive_benefit_base_dev_%j.err

set -euo pipefail
umask 077
ulimit -c 0

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY

TILLICUM_ROOT=/gpfs/projects/stf/claizhan/subliminal-mitigate
REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate
ENV_ROOT=$TILLICUM_ROOT/envs/subliminal-mitigate-py311
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/massive_benefit_pilot_v1
CONTROL_ROOT=$OUTPUT_ROOT/control
DATA_ROOT=$OUTPUT_ROOT/data
EVAL_ROOT=$OUTPUT_ROOT/evaluation
GENERATION_ROOT=$EVAL_ROOT/generations
SCORE_ROOT=$EVAL_ROOT/scores
BASE_ROOT=$EVAL_ROOT/base_development
TRAINING_CONFIG=$REPO_ROOT/configs/training_qwen25_7b_massive_benefit_pilot.yaml
PREP_FILE=$CONTROL_ROOT/PREP_COMPLETE.json
AUTH_FILE=$CONTROL_ROOT/AUTHORIZED_MAX_COST_USD_2.93.json
JOBS_FILE=$CONTROL_ROOT/jobs.tsv
DATA_MANIFEST=$DATA_ROOT/data_manifest.json
BASE_EVAL=$SCORE_ROOT/massive_en_dev__pi_base.json

cd "$REPO_ROOT"
module load gcc/13.4.0 cuda/12.9.1
module load conda/Miniforge3-25.3.1-3
conda activate "$ENV_ROOT"

export PYTHONUNBUFFERED=1
export DO_NOT_TRACK=1
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
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
export PYTHONPYCACHEPREFIX=$TILLICUM_ROOT/tmp/massive-base-dev-pyc
mkdir -p "$GENERATION_ROOT" "$SCORE_ROOT" "$BASE_ROOT" \
  "$XDG_CONFIG_HOME" "$PIP_CACHE_DIR" "$TMPDIR" "$HUGGINGFACE_HUB_CACHE" \
  "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR"

date
hostname
nvidia-smi
test -z "${SLURM_ARRAY_JOB_ID:-}"
test -s "$PREP_FILE"
test -s "$AUTH_FILE"
test -s "$JOBS_FILE"

job_record=$(scontrol show job "$SLURM_JOB_ID" -o | tr ' ' '\n')
actual_time_limit=$(awk -F= '$1=="TimeLimit" {print $2; exit}' <<< "$job_record")
test "$(awk -F= '$1=="Requeue" {print $2; exit}' <<< "$job_record")" = 0
test "$(awk -F= '$1=="Account" {print $2; exit}' <<< "$job_record")" = stf
test "$(awk -F= '$1=="Partition" {print $2; exit}' <<< "$job_record")" = gpu-h200
test "$(awk -F= '$1=="NumNodes" {print $2; exit}' <<< "$job_record")" = 1
test "$(awk -F= '$1=="NumTasks" {print $2; exit}' <<< "$job_record")" = 1
requested_tres=$(sed -n 's/^ReqTRES=//p' <<< "$job_record")
test "$(tr ',' '\n' <<< "$requested_tres" | awk -F= '$1=="gres/gpu:h200" {print $2}')" = 1
test "$(tr ',' '\n' <<< "$requested_tres" | awk -F= '$1=="gres/gpu" {print $2}')" = 1

common_audit=(
  --repo-root "$REPO_ROOT" --data-root "$DATA_ROOT"
  --training-config "$TRAINING_CONFIG" --prep-file "$PREP_FILE"
  --jobs-file "$JOBS_FILE" --auth-file "$AUTH_FILE"
)
python scripts/audit_massive_benefit_tillicum_workflow.py verify-job \
  "${common_audit[@]}" --stage base_dev --job-id "$SLURM_JOB_ID" \
  --time-limit "$actual_time_limit"
python scripts/prepare_massive_benefit_pilot_data.py \
  --output_dir "$DATA_ROOT" --audit_only

python scripts/sample_massive_structured_generations.py \
  --model pi_base=BASE --training_config "$TRAINING_CONFIG" \
  --prompt_file "$DATA_ROOT/dev/prompts.json" \
  --output_dir "$GENERATION_ROOT" \
  --max_new_tokens 256 --max_context 2048 --seed 8172026
python scripts/evaluate_massive_benefit_generations.py \
  --data_manifest "$DATA_MANIFEST" \
  --answers_file "$DATA_ROOT/dev/answers.json" \
  --joint_generations_file "$GENERATION_ROOT/massive_en_dev__pi_base.json" \
  --intent_generations_file "$GENERATION_ROOT/massive_en_dev__pi_base__intent_only.json" \
  --output_file "$BASE_EVAL"
python scripts/summarize_massive_benefit_pilot.py base \
  --evaluation "$BASE_EVAL" \
  --output_file "$BASE_ROOT/summary.json" \
  --markdown_file "$BASE_ROOT/summary.md" \
  --sentinel_dir "$CONTROL_ROOT"
cat "$BASE_ROOT/summary.md"

if [[ -s "$CONTROL_ROOT/STOPPED_MASSIVE_BASE" ]]; then
  echo "STOPPED_MASSIVE_BASE: base lacks the preregistered +15pp headroom."
  # Nonzero is intentional: the afterok dependency prevents paid training.
  exit 10
fi
test -s "$CONTROL_ROOT/GO_MASSIVE_BASE_DEV"
echo "BASE_DEVELOPMENT_COMPLETE: training dependency may proceed."
date
