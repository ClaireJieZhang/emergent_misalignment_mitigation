#!/bin/bash
# Select a K&K checkpoint on fresh dev; open sealed final only after a dev GO.
# Initial hard allocation cap for this stage: 1 H200 x 75 minutes.

#SBATCH --job-name=kk_reason_eval
#SBATCH --account=stf
#SBATCH --qos=normal
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=01:15:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/knights_knaves_reasoning_evaluate_%j.out
#SBATCH --error=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/knights_knaves_reasoning_evaluate_%j.err

set -euo pipefail
umask 077
ulimit -c 0

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY

TILLICUM_ROOT=/gpfs/projects/stf/claizhan/subliminal-mitigate
REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate
ENV_ROOT=$TILLICUM_ROOT/envs/subliminal-mitigate-py311
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/knights_knaves_reasoning_pilot_v1
CONTROL_ROOT=$OUTPUT_ROOT/control
DATA_ROOT=$OUTPUT_ROOT/data
MODEL_DIR=$OUTPUT_ROOT/model/kk_reasoning_n5_pilot
PROVENANCE_ROOT=$OUTPUT_ROOT/provenance
SMOKE_ROOT=$PROVENANCE_ROOT/adapter_load_smoke
EVAL_ROOT=$OUTPUT_ROOT/evaluation
GENERATION_ROOT=$EVAL_ROOT/generations
SCORE_ROOT=$EVAL_ROOT/scores
SELECTION_ROOT=$EVAL_ROOT/selection
FINAL_ROOT=$EVAL_ROOT/sealed_final
TRAINING_CONFIG=$REPO_ROOT/configs/training_qwen25_7b_kk_reasoning_pilot.yaml
PREP_FILE=$CONTROL_ROOT/PREP_COMPLETE
AUTH_FILE=$CONTROL_ROOT/AUTHORIZED_MAX_COST_USD_3.60
JOBS_FILE=$CONTROL_ROOT/jobs.tsv
MODEL_MANIFEST=$MODEL_DIR/MODEL_MANIFEST.json
SELECTION_SUMMARY=$SELECTION_ROOT/summary.json
FINAL_SUMMARY=$FINAL_ROOT/summary.json
TRUNCATION_JSON=$EVAL_ROOT/truncation_report.json
TRUNCATION_MD=$EVAL_ROOT/truncation_report.md
EVALUATION_PROVENANCE=$EVAL_ROOT/EVALUATION_PROVENANCE.json

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
export PYTHONPYCACHEPREFIX=$TILLICUM_ROOT/tmp/kk-reasoning-eval-pyc
mkdir -p "$GENERATION_ROOT" "$SCORE_ROOT" "$SELECTION_ROOT" "$FINAL_ROOT" \
  "$XDG_CONFIG_HOME" "$PIP_CACHE_DIR" "$TMPDIR" "$HUGGINGFACE_HUB_CACHE" \
  "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR"

date
hostname
nvidia-smi
test -z "${SLURM_ARRAY_JOB_ID:-}"
test -s "$PREP_FILE"
test -s "$AUTH_FILE"
test -s "$JOBS_FILE"
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
test "$(tr ',' '\n' <<< "$requested_tres" | awk -F= '$1=="node" {print $2}')" = 1

python scripts/audit_knights_knaves_tillicum_workflow.py verify-job \
  --stage evaluate --job-id "$SLURM_JOB_ID" --time-limit "$actual_time_limit" \
  --repo-root "$REPO_ROOT" --data-root "$DATA_ROOT" \
  --training-config "$TRAINING_CONFIG" --prep-file "$PREP_FILE" \
  --auth-file "$AUTH_FILE" --jobs-file "$JOBS_FILE"
python scripts/prepare_knights_knaves_pilot_data.py \
  --output_dir "$DATA_ROOT" --audit_only
python scripts/audit_knights_knaves_tillicum_workflow.py audit-model-manifest \
  --repo-root "$REPO_ROOT" --data-root "$DATA_ROOT" \
  --training-config "$TRAINING_CONFIG" --prep-file "$PREP_FILE" \
  --auth-file "$AUTH_FILE" --jobs-file "$JOBS_FILE" \
  --model-dir "$MODEL_DIR" --smoke-dir "$SMOKE_ROOT" \
  --output-file "$MODEL_MANIFEST"
python scripts/audit_knights_knaves_tillicum_workflow.py write-evaluation-provenance \
  --repo-root "$REPO_ROOT" --data-root "$DATA_ROOT" \
  --training-config "$TRAINING_CONFIG" --prep-file "$PREP_FILE" \
  --auth-file "$AUTH_FILE" --jobs-file "$JOBS_FILE" \
  --model-manifest "$MODEL_MANIFEST" --job-id "$SLURM_JOB_ID" \
  --output-file "$EVALUATION_PROVENANCE"

checkpoint_steps=(64 128 192 320 448 640)
dev_names=(pi_base step_64 step_128 step_192 step_320 step_448 step_640)
model_specs=(--model pi_base=BASE)
for step in "${checkpoint_steps[@]}"; do
  model_specs+=(--model "step_$step=$MODEL_DIR/checkpoint-$step")
done

dev_report_specs=()
for name in "${dev_names[@]}"; do
  dev_report_specs+=(
    --evaluation "dev_n5__${name}=$SCORE_ROOT/dev_n5__${name}.json"
  )
done

write_or_audit_truncation_report() {
  if [[ -s "$TRUNCATION_JSON" ]]; then
    python scripts/audit_knights_knaves_tillicum_workflow.py \
      audit-truncation-report "$@" --output-file "$TRUNCATION_JSON"
  else
    python scripts/audit_knights_knaves_tillicum_workflow.py \
      write-truncation-report "$@" --output-file "$TRUNCATION_JSON" \
      --markdown-file "$TRUNCATION_MD"
  fi
}

selection_decision=
if [[ -s "$SELECTION_SUMMARY" ]]; then
  selection_decision=$(python scripts/audit_knights_knaves_tillicum_workflow.py \
    audit-decision --summary-file "$SELECTION_SUMMARY" \
    --sentinel-dir "$CONTROL_ROOT" --go-name GO_KK_SEALED_FINAL \
    --stop-name STOPPED_NO_GO --allow-opposite --restore-missing)
else
  test ! -e "$CONTROL_ROOT/STOPPED_NO_GO"
  test ! -e "$CONTROL_ROOT/GO_KK_BENEFIT_UNIONS"

  echo "=== Fresh N=5 checkpoint-selection generations (sealed final unopened) ==="
  python scripts/sample_knights_knaves_generations.py \
    "${model_specs[@]}" \
    --training_config "$TRAINING_CONFIG" \
    --prompt_file "$DATA_ROOT/dev/dev_n5_prompts.json" \
    --output_dir "$GENERATION_ROOT" \
    --max_new_tokens 2048 --max_context 4096 --seed 8152026

  for name in "${dev_names[@]}"; do
    python scripts/evaluate_knights_knaves_generations.py \
      --answers_file "$DATA_ROOT/dev/dev_n5_answers.json" \
      --generations_file "$GENERATION_ROOT/dev_n5__${name}.json" \
      --output_file "$SCORE_ROOT/dev_n5__${name}.json"
  done

  selection_args=(
    select
    --base "$SCORE_ROOT/dev_n5__pi_base.json"
    --output_file "$SELECTION_SUMMARY"
    --markdown_file "$SELECTION_ROOT/summary.md"
    --sentinel_dir "$CONTROL_ROOT"
    --min_gain 0.10
    --max_p 0.05
    --min_parse_coverage 0.99
    --bootstrap_replicates 10000
  )
  for step in "${checkpoint_steps[@]}"; do
    selection_args+=(
      --checkpoint "$step=$SCORE_ROOT/dev_n5__step_${step}.json"
    )
  done
  python scripts/summarize_knights_knaves_pilot.py "${selection_args[@]}"
  selection_decision=$(python scripts/audit_knights_knaves_tillicum_workflow.py \
    audit-decision --summary-file "$SELECTION_SUMMARY" \
    --sentinel-dir "$CONTROL_ROOT" --go-name GO_KK_SEALED_FINAL \
    --stop-name STOPPED_NO_GO --allow-opposite --restore-missing)
fi

if [[ "$selection_decision" = STOP ]]; then
  write_or_audit_truncation_report "${dev_report_specs[@]}"
  cat "$SELECTION_ROOT/summary.md"
  cat "$TRUNCATION_MD"
  echo "STOPPED_NO_GO: no sealed-final prompt was generated."
  exit 0
fi
test "$selection_decision" = GO

selected_step=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["step"])' \
  "$SELECTION_SUMMARY")
selected_name=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["model_name"])' \
  "$SELECTION_SUMMARY")
case "$selected_step:$selected_name" in
  64:step_64|128:step_128|192:step_192|320:step_320|448:step_448|640:step_640) ;;
  *) echo "Invalid frozen checkpoint selection: $selected_step:$selected_name" >&2; exit 2 ;;
esac
selected_path=$MODEL_DIR/checkpoint-$selected_step

final_sets=(official_n4 official_n5 official_n6 fresh_n4 fresh_n5 fresh_n6)
final_report_specs=()
base_eval_args=()
candidate_eval_args=()
for set_name in "${final_sets[@]}"; do
  final_report_specs+=(
    --evaluation "${set_name}__pi_base=$SCORE_ROOT/${set_name}__pi_base.json"
    --evaluation "${set_name}__${selected_name}=$SCORE_ROOT/${set_name}__${selected_name}.json"
  )
  base_eval_args+=(--base_eval "$set_name=$SCORE_ROOT/${set_name}__pi_base.json")
  candidate_eval_args+=(--candidate_eval "$set_name=$SCORE_ROOT/${set_name}__${selected_name}.json")
done

if [[ -s "$FINAL_SUMMARY" ]]; then
  final_decision=$(python scripts/audit_knights_knaves_tillicum_workflow.py \
    audit-decision --summary-file "$FINAL_SUMMARY" \
    --sentinel-dir "$CONTROL_ROOT" --go-name GO_KK_BENEFIT_UNIONS \
    --stop-name STOPPED_NO_GO --restore-missing)
else
  test ! -e "$CONTROL_ROOT/STOPPED_NO_GO"
  test ! -e "$CONTROL_ROOT/GO_KK_BENEFIT_UNIONS"
  echo "=== Development GO: open six sealed-final prompt banks ==="
  for set_name in "${final_sets[@]}"; do
    set_dir=$DATA_ROOT/sealed_final
    python scripts/sample_knights_knaves_generations.py \
      --model pi_base=BASE --model "$selected_name=$selected_path" \
      --training_config "$TRAINING_CONFIG" \
      --prompt_file "$set_dir/${set_name}_prompts.json" \
      --output_dir "$GENERATION_ROOT" \
      --max_new_tokens 2048 --max_context 4096 --seed 8152026
    for name in pi_base "$selected_name"; do
      python scripts/evaluate_knights_knaves_generations.py \
        --answers_file "$set_dir/${set_name}_answers.json" \
        --generations_file "$GENERATION_ROOT/${set_name}__${name}.json" \
        --output_file "$SCORE_ROOT/${set_name}__${name}.json"
    done
  done

  python scripts/summarize_knights_knaves_pilot.py final \
    --selection_file "$SELECTION_SUMMARY" \
    "${base_eval_args[@]}" "${candidate_eval_args[@]}" \
    --output_file "$FINAL_SUMMARY" \
    --markdown_file "$FINAL_ROOT/summary.md" \
    --sentinel_dir "$CONTROL_ROOT" \
    --min_n5_gain 0.10 --min_transfer_delta 0.0 \
    --min_each_transfer_delta=-0.02 \
    --bootstrap_replicates 10000
  final_decision=$(python scripts/audit_knights_knaves_tillicum_workflow.py \
    audit-decision --summary-file "$FINAL_SUMMARY" \
    --sentinel-dir "$CONTROL_ROOT" --go-name GO_KK_BENEFIT_UNIONS \
    --stop-name STOPPED_NO_GO --restore-missing)
fi

write_or_audit_truncation_report \
  "${dev_report_specs[@]}" "${final_report_specs[@]}"
cat "$FINAL_ROOT/summary.md"
cat "$TRUNCATION_MD"
if [[ "$final_decision" = GO ]]; then
  echo "GO_KK_BENEFIT_UNIONS: scientific gate passed; no union or quorum job was submitted."
else
  test "$final_decision" = STOP
  echo "STOPPED_NO_GO: benefit pilot did not authorize matched unions."
fi
date
