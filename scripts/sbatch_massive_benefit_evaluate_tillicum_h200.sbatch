#!/bin/bash
# Dev-only checkpoint selection, then conditional sealed MASSIVE test.
# Hard allocation cap: 1 H200 x 75 minutes; no continuation is submitted.

#SBATCH --job-name=massive_eval
#SBATCH --account=stf
#SBATCH --qos=normal
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=01:15:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/massive_benefit_evaluate_%j.out
#SBATCH --error=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/massive_benefit_evaluate_%j.err

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
MODEL_DIR=$OUTPUT_ROOT/model/massive_en_benefit_pilot
EVAL_ROOT=$OUTPUT_ROOT/evaluation
GENERATION_ROOT=$EVAL_ROOT/generations
SCORE_ROOT=$EVAL_ROOT/scores
SELECTION_ROOT=$EVAL_ROOT/selection
FINAL_ROOT=$EVAL_ROOT/sealed_final
TRAINING_CONFIG=$REPO_ROOT/configs/training_qwen25_7b_massive_benefit_pilot.yaml
PREP_FILE=$CONTROL_ROOT/PREP_COMPLETE.json
AUTH_FILE=$CONTROL_ROOT/AUTHORIZED_MAX_COST_USD_2.93.json
JOBS_FILE=$CONTROL_ROOT/jobs.tsv
MODEL_MANIFEST=$MODEL_DIR/MODEL_MANIFEST.json
DATA_MANIFEST=$DATA_ROOT/data_manifest.json
BASE_DEV_EVAL=$SCORE_ROOT/massive_en_dev__pi_base.json
SELECTION_SUMMARY=$SELECTION_ROOT/summary.json
FINAL_SUMMARY=$FINAL_ROOT/summary.json

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
export PYTHONPYCACHEPREFIX=$TILLICUM_ROOT/tmp/massive-eval-pyc
mkdir -p "$GENERATION_ROOT" "$SCORE_ROOT" "$SELECTION_ROOT" \
  "$XDG_CONFIG_HOME" "$PIP_CACHE_DIR" "$TMPDIR" "$HUGGINGFACE_HUB_CACHE" \
  "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR"

date
hostname
nvidia-smi
test -z "${SLURM_ARRAY_JOB_ID:-}"
test -s "$CONTROL_ROOT/GO_MASSIVE_BASE_DEV"
test -s "$MODEL_DIR/TRAIN_COMPLETE"
test -s "$MODEL_MANIFEST"

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
  "${common_audit[@]}" --stage evaluate --job-id "$SLURM_JOB_ID" \
  --time-limit "$actual_time_limit"
python scripts/prepare_massive_benefit_pilot_data.py \
  --output_dir "$DATA_ROOT" --audit_only
python scripts/audit_massive_benefit_tillicum_workflow.py audit-model \
  "${common_audit[@]}" --model-dir "$MODEL_DIR" \
  --output-file "$MODEL_MANIFEST"
python scripts/summarize_massive_benefit_pilot.py base \
  --evaluation "$BASE_DEV_EVAL" \
  --output_file "$EVAL_ROOT/base_development/summary.json" \
  --markdown_file "$EVAL_ROOT/base_development/summary.md" \
  --sentinel_dir "$CONTROL_ROOT"
test -s "$CONTROL_ROOT/GO_MASSIVE_BASE_DEV"

checkpoint_steps=(15 30 60 90 150)
model_specs=()
for step in "${checkpoint_steps[@]}"; do
  model_specs+=(--model "step_$step=$MODEL_DIR/checkpoint-$step")
done

echo "=== Development-only generation and frozen checkpoint selection ==="
python scripts/sample_massive_structured_generations.py \
  "${model_specs[@]}" --training_config "$TRAINING_CONFIG" \
  --prompt_file "$DATA_ROOT/dev/prompts.json" \
  --output_dir "$GENERATION_ROOT" \
  --max_new_tokens 256 --max_context 2048 --seed 8172026

dev_names=(step_15 step_30 step_60 step_90 step_150)
for name in "${dev_names[@]}"; do
  python scripts/evaluate_massive_benefit_generations.py \
    --data_manifest "$DATA_MANIFEST" \
    --answers_file "$DATA_ROOT/dev/answers.json" \
    --joint_generations_file "$GENERATION_ROOT/massive_en_dev__${name}.json" \
    --intent_generations_file "$GENERATION_ROOT/massive_en_dev__${name}__intent_only.json" \
    --output_file "$SCORE_ROOT/massive_en_dev__${name}.json"
done

selection_args=(
  select --base "$BASE_DEV_EVAL" --model_manifest "$MODEL_MANIFEST"
  --output_file "$SELECTION_SUMMARY"
  --markdown_file "$SELECTION_ROOT/summary.md"
  --sentinel_dir "$CONTROL_ROOT"
)
for step in "${checkpoint_steps[@]}"; do
  selection_args+=(
    --checkpoint "$step=$SCORE_ROOT/massive_en_dev__step_${step}.json"
  )
done
python scripts/summarize_massive_benefit_pilot.py "${selection_args[@]}"
cat "$SELECTION_ROOT/summary.md"

if [[ -s "$CONTROL_ROOT/STOPPED_MASSIVE_SELECTION" ]]; then
  echo "STOPPED_MASSIVE_SELECTION: sealed test was not opened."
  exit 0
fi
test -s "$CONTROL_ROOT/GO_MASSIVE_SEALED_TEST"

selected_step=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["step"])' \
  "$SELECTION_SUMMARY")
selected_name=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["model_name"])' \
  "$SELECTION_SUMMARY")
case "$selected_step:$selected_name" in
  15:step_15|30:step_30|60:step_60|90:step_90|150:step_150) ;;
  *) echo "Invalid frozen MASSIVE checkpoint selection" >&2; exit 2 ;;
esac
selected_path=$MODEL_DIR/checkpoint-$selected_step

# Before this GO, sealed-test files are integrity-hashed by whole-data audits,
# but no sealed-test example is semantically loaded, generated, scored, or used
# for checkpoint selection.
echo "=== Development GO: evaluating cleaned official test subset ==="
mkdir -p "$FINAL_ROOT"
python scripts/sample_massive_structured_generations.py \
  --model pi_base=BASE --model "$selected_name=$selected_path" \
  --training_config "$TRAINING_CONFIG" \
  --prompt_file "$DATA_ROOT/sealed_test/prompts.json" \
  --output_dir "$GENERATION_ROOT" \
  --max_new_tokens 256 --max_context 2048 --seed 8172026

for name in pi_base "$selected_name"; do
  python scripts/evaluate_massive_benefit_generations.py \
    --data_manifest "$DATA_MANIFEST" \
    --answers_file "$DATA_ROOT/sealed_test/answers.json" \
    --joint_generations_file "$GENERATION_ROOT/massive_en_test__${name}.json" \
    --intent_generations_file "$GENERATION_ROOT/massive_en_test__${name}__intent_only.json" \
    --output_file "$SCORE_ROOT/massive_en_test__${name}.json"
done

python scripts/summarize_massive_benefit_pilot.py final \
  --selection_file "$SELECTION_SUMMARY" \
  --model_manifest "$MODEL_MANIFEST" \
  --base "$SCORE_ROOT/massive_en_test__pi_base.json" \
  --candidate "$SCORE_ROOT/massive_en_test__${selected_name}.json" \
  --output_file "$FINAL_SUMMARY" \
  --markdown_file "$FINAL_ROOT/summary.md" \
  --sentinel_dir "$CONTROL_ROOT"
cat "$FINAL_ROOT/summary.md"

if [[ -s "$CONTROL_ROOT/GO_MASSIVE_BENEFIT_ONLY" ]]; then
  echo "FINAL_EVALUATION_COMPLETE: GO_MASSIVE_BENEFIT_ONLY"
  echo "No medical union, extra adapter, quorum, or continuation was submitted."
else
  test -s "$CONTROL_ROOT/STOPPED_MASSIVE_FINAL"
  echo "FINAL_EVALUATION_COMPLETE: STOPPED_MASSIVE_FINAL"
fi
date
