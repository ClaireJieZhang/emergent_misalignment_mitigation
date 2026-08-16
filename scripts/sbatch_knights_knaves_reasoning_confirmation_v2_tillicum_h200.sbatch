#!/bin/bash
# K&K v2 evaluation-only confirmation. Hard cap: 1 H200 x 30 min = $0.45.

#SBATCH --job-name=kk_confirm_v2
#SBATCH --account=stf
#SBATCH --qos=normal
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/knights_knaves_confirmation_v2_%j.out
#SBATCH --error=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/knights_knaves_confirmation_v2_%j.err

set -euo pipefail
umask 077
ulimit -c 0

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY

ROOT=/gpfs/projects/stf/claizhan/subliminal-mitigate
REPO=$ROOT/projects/subliminal-mitigate
ENV_ROOT=$ROOT/envs/subliminal-mitigate-py311
V1_ROOT=$ROOT/outputs/knights_knaves_reasoning_pilot_v1
V1_DATA=$V1_ROOT/data
V2_ROOT=$ROOT/outputs/knights_knaves_reasoning_confirmation_v2
V2_DATA=$V2_ROOT/data
CONTROL=$V2_ROOT/control
EVAL=$V2_ROOT/evaluation
GENERATIONS=$EVAL/generations
SCORES=$EVAL/scores
CONFIRMATION=$EVAL/confirmation
FINAL=$EVAL/sealed_final
CONFIG=$REPO/configs/training_qwen25_7b_kk_reasoning_pilot.yaml
CHECKPOINT=$V1_ROOT/model/kk_reasoning_n5_pilot/checkpoint-192
FINGERPRINT=36a710b93564ccb9d7c939fdf644bae9a80a6e4c81ca73c2634f4e1a1741701c
PREP=$CONTROL/PREP_COMPLETE
AUTH=$CONTROL/AUTHORIZED_MAX_COST_USD_0.45
JOBS=$CONTROL/jobs.tsv
CONFIRMATION_SUMMARY=$CONFIRMATION/summary.json
FINAL_SUMMARY=$FINAL/summary.json

cd "$REPO"
module load gcc/13.4.0 cuda/12.9.1
module load conda/Miniforge3-25.3.1-3
conda activate "$ENV_ROOT"

export PYTHONUNBUFFERED=1
export DO_NOT_TRACK=1
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export XDG_CACHE_HOME=$ROOT/cache
export XDG_CONFIG_HOME=$ROOT/config
export PIP_CACHE_DIR=$ROOT/cache/pip
export TMPDIR=$ROOT/tmp
export HF_HOME=$ROOT/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME
export VLLM_CACHE_ROOT=$ROOT/cache/vllm
export TRITON_CACHE_DIR=$ROOT/cache/triton
export PYTHONPYCACHEPREFIX=$ROOT/tmp/kk-confirmation-v2-pyc
mkdir -p "$GENERATIONS" "$SCORES" "$CONFIRMATION" "$FINAL" \
  "$XDG_CONFIG_HOME" "$PIP_CACHE_DIR" "$TMPDIR" "$HUGGINGFACE_HUB_CACHE" \
  "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR"

date
hostname
nvidia-smi
test -z "${SLURM_ARRAY_JOB_ID:-}"
test -s "$PREP" && test -s "$AUTH" && test -s "$JOBS"
test -s "$CHECKPOINT/adapter_config.json"

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
test "$(tr ',' '\n' <<< "$requested_tres" | awk -F= '$1=="node" {print $2}')" = 1

python scripts/audit_knights_knaves_confirmation_v2_workflow.py verify-job \
  --repo-root "$REPO" --v1-root "$V1_ROOT" --v2-data-root "$V2_DATA" \
  --prep-file "$PREP" --auth-file "$AUTH" --jobs-file "$JOBS" \
  --job-id "$SLURM_JOB_ID" --time-limit "$actual_time_limit"
python scripts/prepare_knights_knaves_confirmation_v2_data.py \
  --v1_data_root "$V1_DATA" --output_dir "$V2_DATA" --audit_only

generate_direct_phase() {
  python scripts/sample_knights_knaves_generations.py \
    --model pi_base=BASE --model step_192="$CHECKPOINT" \
    --training_config "$CONFIG" "$@" \
    --output_dir "$GENERATIONS" --max_new_tokens 2048 \
    --max_context 4096 --seed 8152026
}

generate_controlled_phase() {
  python scripts/sample_knights_knaves_structured_choices.py \
    --model pi_base=BASE --model step_192="$CHECKPOINT" \
    --training_config "$CONFIG" "$@" --output_dir "$GENERATIONS" \
    --max_new_tokens 2048 --max_context 4096 --seed 8152026
}

evaluate_set() {
  local set_name=$1 answers_file=$2
  local model
  for model in pi_base step_192; do
    python scripts/evaluate_knights_knaves_confirmation_v2.py --mode direct \
      --answers_file "$answers_file" \
      --generations_file "$GENERATIONS/${set_name}__${model}.json" \
      --output_file "$SCORES/${set_name}__${model}__direct.json"
    python scripts/evaluate_knights_knaves_confirmation_v2.py --mode controlled \
      --answers_file "$answers_file" \
      --generations_file "$GENERATIONS/${set_name}__${model}__controlled.json" \
      --output_file "$SCORES/${set_name}__${model}__controlled.json"
  done
}

confirmation_decision=
if [[ -s "$CONFIRMATION_SUMMARY" ]]; then
  confirmation_decision=$(python scripts/summarize_knights_knaves_confirmation_v2.py audit \
    --summary_file "$CONFIRMATION_SUMMARY" --sentinel_dir "$CONTROL" \
    --go_name GO_KK_V2_SEALED_FINAL --stop_name STOPPED_KK_V2_NO_GO \
    --markdown_file "$CONFIRMATION/summary.md" \
    --title "K&K v2 independent confirmation")
else
  test ! -e "$CONTROL/GO_KK_V2_BENEFIT_UNIONS"
  test ! -e "$CONTROL/STOPPED_KK_V2_FINAL"
  # Confirmation uses at most one persistent direct model load and one
  # persistent XGrammar model load.  No sealed-final path is passed here.
  generate_direct_phase \
    --prompt_file "$V2_DATA/confirmation/confirmation_n5_prompts.json"
  generate_controlled_phase \
    --prompt_file "$V2_DATA/confirmation/confirmation_n5_prompts.json" \
    --names_file "$V2_DATA/names/confirmation_n5_names.json"
  evaluate_set confirmation_n5 \
    "$V2_DATA/confirmation/confirmation_n5_answers.json"
  python scripts/summarize_knights_knaves_confirmation_v2.py confirmation \
    --direct_base "$SCORES/confirmation_n5__pi_base__direct.json" \
    --direct_candidate "$SCORES/confirmation_n5__step_192__direct.json" \
    --controlled_base "$SCORES/confirmation_n5__pi_base__controlled.json" \
    --controlled_candidate "$SCORES/confirmation_n5__step_192__controlled.json" \
    --candidate_fingerprint "$FINGERPRINT" \
    --v2_data_manifest "$V2_DATA/confirmation_v2_manifest.json" \
    --output_file "$CONFIRMATION_SUMMARY" \
    --markdown_file "$CONFIRMATION/summary.md" --sentinel_dir "$CONTROL" \
    --replicates 10000
  confirmation_decision=$(python scripts/summarize_knights_knaves_confirmation_v2.py audit \
    --summary_file "$CONFIRMATION_SUMMARY" --sentinel_dir "$CONTROL" \
    --go_name GO_KK_V2_SEALED_FINAL --stop_name STOPPED_KK_V2_NO_GO \
    --markdown_file "$CONFIRMATION/summary.md" \
    --title "K&K v2 independent confirmation")
fi
cat "$CONFIRMATION/summary.md"
if [[ "$confirmation_decision" = STOP ]]; then
  echo "STOPPED_KK_V2_NO_GO: v1 final model evaluation remains unopened."
  exit 0
fi
test "$confirmation_decision" = GO

final_sets=(official_n4 official_n5 official_n6 fresh_n4 fresh_n5 fresh_n6)
if [[ -s "$FINAL_SUMMARY" ]]; then
  final_decision=$(python scripts/summarize_knights_knaves_confirmation_v2.py audit \
    --summary_file "$FINAL_SUMMARY" --sentinel_dir "$CONTROL" \
    --go_name GO_KK_V2_BENEFIT_UNIONS --stop_name STOPPED_KK_V2_FINAL \
    --markdown_file "$FINAL/summary.md" --title "K&K v2 sealed-final gate")
else
  final_prompt_args=()
  final_controlled_args=()
  for set_name in "${final_sets[@]}"; do
    final_prompt_args+=(
      --prompt_file "$V1_DATA/sealed_final/${set_name}_prompts.json"
    )
    final_controlled_args+=(
      --prompt_file "$V1_DATA/sealed_final/${set_name}_prompts.json"
      --names_file "$V2_DATA/names/${set_name}_names.json"
    )
  done
  # This block is reachable only after confirmation GO.  All six final sets
  # share one direct model load and one persistent XGrammar model load.
  generate_direct_phase "${final_prompt_args[@]}"
  generate_controlled_phase "${final_controlled_args[@]}"

  direct_base_args=()
  direct_candidate_args=()
  controlled_base_args=()
  controlled_candidate_args=()
  for set_name in "${final_sets[@]}"; do
    evaluate_set "$set_name" \
      "$V1_DATA/sealed_final/${set_name}_answers.json"
    direct_base_args+=(--direct_base "$set_name=$SCORES/${set_name}__pi_base__direct.json")
    direct_candidate_args+=(--direct_candidate "$set_name=$SCORES/${set_name}__step_192__direct.json")
    controlled_base_args+=(--controlled_base "$set_name=$SCORES/${set_name}__pi_base__controlled.json")
    controlled_candidate_args+=(--controlled_candidate "$set_name=$SCORES/${set_name}__step_192__controlled.json")
  done
  python scripts/summarize_knights_knaves_confirmation_v2.py final \
    --confirmation_summary "$CONFIRMATION_SUMMARY" \
    --v2_data_manifest "$V2_DATA/confirmation_v2_manifest.json" \
    "${direct_base_args[@]}" "${direct_candidate_args[@]}" \
    "${controlled_base_args[@]}" "${controlled_candidate_args[@]}" \
    --candidate_fingerprint "$FINGERPRINT" \
    --output_file "$FINAL_SUMMARY" --markdown_file "$FINAL/summary.md" \
    --sentinel_dir "$CONTROL" --replicates 10000
  final_decision=$(python scripts/summarize_knights_knaves_confirmation_v2.py audit \
    --summary_file "$FINAL_SUMMARY" --sentinel_dir "$CONTROL" \
    --go_name GO_KK_V2_BENEFIT_UNIONS --stop_name STOPPED_KK_V2_FINAL \
    --markdown_file "$FINAL/summary.md" --title "K&K v2 sealed-final gate")
fi
cat "$FINAL/summary.md"
if [[ "$final_decision" = GO ]]; then
  echo "GO_KK_V2_BENEFIT_UNIONS: no union, training, or quorum job was submitted."
else
  test "$final_decision" = STOP
  echo "STOPPED_KK_V2_FINAL: K&K is not authorized as the union benefit."
fi
date
