#!/bin/bash
# Select on APPS validation, freeze, then evaluate only base + selected pilot
# on the existing 157-task LCB gate and the sealed EvalPlus prompt bank.
# Hard allocation cap for this stage: 1 H200 x 60 minutes.

#SBATCH --job-name=gc_apps_eval
#SBATCH --account=stf
#SBATCH --qos=normal
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/general_code_apps_repaired_evaluate_%j.out
#SBATCH --error=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/general_code_apps_repaired_evaluate_%j.err

set -euo pipefail
umask 077
ulimit -c 0

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY

TILLICUM_ROOT=/gpfs/projects/stf/claizhan/subliminal-mitigate
REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate
ENV_ROOT=$TILLICUM_ROOT/envs/subliminal-mitigate-py311
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/general_code_apps_repaired_pilot_v1
CONTROL_ROOT=$OUTPUT_ROOT/control
DATA_ROOT=$OUTPUT_ROOT/data
MODEL_DIR=$OUTPUT_ROOT/model/apps_repaired_pilot
EVAL_ROOT=$OUTPUT_ROOT/evaluation
APPS_ROOT=$EVAL_ROOT/apps_validation
LCB_ROOT=$EVAL_ROOT/livecodebench_gate
EVALPLUS_ROOT=$EVAL_ROOT/evalplus
TRAINING_CONFIG=$REPO_ROOT/configs/training_qwen25_7b_apps_repaired_pilot.yaml
AUTH_FILE=$CONTROL_ROOT/AUTHORIZED_MAX_COST_USD_1.80

LCB_SOURCE=$TILLICUM_ROOT/outputs/general_code_magicoder_lcb_q3_m4
EVALPLUS_SOURCE=$TILLICUM_ROOT/outputs/general_code_evalplus_base_vs_pilot_v1
LCB_REPO=$LCB_SOURCE/assets/LiveCodeBench
LCB_SITE=$LCB_SOURCE/assets/lcb-python-site
SANDBOX_SIF=$LCB_SOURCE/assets/python-3.11-slim-amd64.sif
LCB_PROMPTS=$LCB_SOURCE/data/lcb_gate_prompts.json
LCB_BENCHMARK=$LCB_SOURCE/data/lcb_gate_evaluator.jsonl
EVALPLUS_PROMPTS=$EVALPLUS_SOURCE/data/evalplus_prompts.json
EVALPLUS_REPO=$EVALPLUS_SOURCE/assets/evalplus-v0.3.1
EVALPLUS_SITE=$EVALPLUS_SOURCE/assets/evalplus-python-site
HUMANEVAL_FILE=$EVALPLUS_SOURCE/assets/HumanEvalPlus-v0.1.10.jsonl.gz
MBPP_FILE=$EVALPLUS_SOURCE/assets/MbppPlus-v0.2.0.jsonl.gz

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
mkdir -p "$APPS_ROOT/generations" "$APPS_ROOT/custom" "$APPS_ROOT/evaluations" \
  "$LCB_ROOT/generations" "$LCB_ROOT/custom" "$LCB_ROOT/evaluations" \
  "$EVALPLUS_ROOT/generations" "$EVALPLUS_ROOT/evaluations/humaneval" \
  "$EVALPLUS_ROOT/evaluations/mbpp" "$XDG_CONFIG_HOME" "$PIP_CACHE_DIR" \
  "$TMPDIR" "$HUGGINGFACE_HUB_CACHE" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR"

date
hostname
nvidia-smi
test -s "$AUTH_FILE"
test -z "${SLURM_ARRAY_JOB_ID:-}"
job_record=$(scontrol show job "$SLURM_JOB_ID" -o | tr ' ' '\n')
actual_time_limit=$(awk -F= '$1=="TimeLimit" {print $2; exit}' <<< "$job_record")
python scripts/verify_general_code_apps_repaired_authorization.py \
  --stage evaluate --time-limit "$actual_time_limit" --job-id "$SLURM_JOB_ID"
test "$(awk -F= '$1=="Requeue" {print $2; exit}' <<< "$job_record")" = 0
test "$(awk -F= '$1=="NumNodes" {print $2; exit}' <<< "$job_record")" = 1
test "$(awk -F= '$1=="NumTasks" {print $2; exit}' <<< "$job_record")" = 1
requested_tres=$(awk -F= '$1=="ReqTRES" {print $2; exit}' <<< "$job_record")
test "$(tr ',' '\n' <<< "$requested_tres" | awk -F= '$1=="gres/gpu:h200" {print $2}')" = 1
test "$(tr ',' '\n' <<< "$requested_tres" | awk -F= '$1=="gres/gpu" {print $2}')" = 1
test "$(tr ',' '\n' <<< "$requested_tres" | awk -F= '$1=="node" {print $2}')" = 1
test -s "$OUTPUT_ROOT/PREP_COMPLETE"
test -s "$MODEL_DIR/TRAIN_COMPLETE"
test -s "$MODEL_DIR/repaired_pilot_model_manifest.json"
python scripts/prepare_repaired_code_pilot_data.py audit --output-root "$DATA_ROOT"
python scripts/audit_repaired_code_pilot_model.py \
  --model-dir "$MODEL_DIR" --dataset-dir "$DATA_ROOT/apps_repaired_train" \
  --data-manifest "$DATA_ROOT/data_manifest.json" \
  --training-config "$TRAINING_CONFIG" \
  --output-file "$MODEL_DIR/repaired_pilot_model_manifest.json"

test "$(git -C "$LCB_REPO" rev-parse HEAD)" = 28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24
test -z "$(git -C "$LCB_REPO" status --porcelain)"
test "$(git -C "$EVALPLUS_REPO" rev-parse HEAD)" = e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2
test -z "$(git -C "$EVALPLUS_REPO" status --porcelain)"
test -s "$LCB_SITE/.ready"
test -s "$EVALPLUS_SITE/.ready"
sha256sum -c "$SANDBOX_SIF.sha256"
echo '272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101  '"$HUMANEVAL_FILE" | sha256sum -c -
echo 'af43697e8791c4c149bdfd6b489d8b5412507551ac20e28a439f650b8225db63  '"$MBPP_FILE" | sha256sum -c -

model_specs=(
  --model pi_base=BASE
  --model "step_10=$MODEL_DIR/checkpoint-10"
  --model "step_20=$MODEL_DIR/checkpoint-20"
  --model "step_30=$MODEL_DIR/checkpoint-30"
  --model "step_40=$MODEL_DIR/checkpoint-40"
)

echo "=== APPS-only trajectory generation and hidden-test execution ==="
python scripts/sample_lcb_direct_generations.py \
  "${model_specs[@]}" \
  --training_config "$TRAINING_CONFIG" \
  --prompt_file "$DATA_ROOT/apps_repaired_validation_prompts.json" \
  --output_dir "$APPS_ROOT/generations" \
  --n_samples 1 --temperature 0 --max_new_tokens 1024 --max_context 4096 \
  --seed 8132026

generation_specs=()
evaluation_specs=()
for name in pi_base step_10 step_20 step_30 step_40; do
  generation_specs+=(--generation "$name=$APPS_ROOT/generations/$name.json")
  evaluation_specs+=(--evaluation "$name=$APPS_ROOT/evaluations/$name.json")
done
python scripts/prepare_lcb_evaluation.py \
  "${generation_specs[@]}" \
  --prompt_file "$DATA_ROOT/apps_repaired_validation_prompts.json" \
  --output_dir "$APPS_ROOT/custom"
for name in pi_base step_10 step_20 step_30 step_40; do
  bash scripts/run_lcb_one_tillicum.sh \
    "$DATA_ROOT/apps_repaired_validation_evaluator.jsonl" \
    "$APPS_ROOT/custom/$name.custom.json" \
    "$APPS_ROOT/evaluations/$name.json" \
    "$LCB_SITE" "$LCB_REPO" "$SANDBOX_SIF" 6
done
python scripts/summarize_lcb_evaluations.py \
  "${evaluation_specs[@]}" \
  --output_file "$APPS_ROOT/summary.json" \
  --markdown_file "$APPS_ROOT/summary.md" \
  --base_name pi_base --good_names '' --quorum_names ''

echo "=== Freeze checkpoint selection before opening external suites ==="
python scripts/select_repaired_code_pilot_checkpoint.py \
  --apps-summary "$APPS_ROOT/summary.json" \
  --model-manifest "$MODEL_DIR/repaired_pilot_model_manifest.json" \
  --output-file "$APPS_ROOT/SELECTED_CHECKPOINT.json" \
  --expected-problems 200
selected=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_checkpoint"])' \
  "$APPS_ROOT/SELECTED_CHECKPOINT.json")
case "$selected" in
  step_10) selected_path=$MODEL_DIR/checkpoint-10 ;;
  step_20) selected_path=$MODEL_DIR/checkpoint-20 ;;
  step_30) selected_path=$MODEL_DIR/checkpoint-30 ;;
  step_40) selected_path=$MODEL_DIR/checkpoint-40 ;;
  *) echo "Invalid sealed selection: $selected" >&2; exit 2 ;;
esac
python - "$APPS_ROOT/SELECTED_CHECKPOINT.json" "$selected_path" <<'PY'
import hashlib, json, os, sys
selection = json.load(open(sys.argv[1], encoding="utf-8"))
root = sys.argv[2]
seal = selection["selected_adapter"]
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
assert sha(os.path.join(root, "adapter_config.json")) == seal["adapter_config_sha256"]
weights = os.path.join(root, seal["adapter_weights_filename"])
assert sha(weights) == seal["adapter_weights_sha256"]
print("selected adapter seal verified")
PY

echo "=== External LiveCodeBench evaluation: base + APPS-selected only ==="
python scripts/sample_lcb_direct_generations.py \
  --model pi_base=BASE --model "pi_good_0=$selected_path" \
  --training_config "$TRAINING_CONFIG" --prompt_file "$LCB_PROMPTS" \
  --output_dir "$LCB_ROOT/generations" \
  --n_samples 1 --temperature 0 --max_new_tokens 1024 --max_context 4096 \
  --seed 8132026
python scripts/prepare_lcb_evaluation.py \
  --generation "pi_base=$LCB_ROOT/generations/pi_base.json" \
  --generation "pi_good_0=$LCB_ROOT/generations/pi_good_0.json" \
  --prompt_file "$LCB_PROMPTS" --output_dir "$LCB_ROOT/custom"
for name in pi_base pi_good_0; do
  bash scripts/run_lcb_one_tillicum.sh \
    "$LCB_BENCHMARK" "$LCB_ROOT/custom/$name.custom.json" \
    "$LCB_ROOT/evaluations/$name.json" \
    "$LCB_SITE" "$LCB_REPO" "$SANDBOX_SIF" 6
done
python scripts/summarize_lcb_evaluations.py \
  --evaluation "pi_base=$LCB_ROOT/evaluations/pi_base.json" \
  --evaluation "pi_good_0=$LCB_ROOT/evaluations/pi_good_0.json" \
  --output_file "$LCB_ROOT/summary.json" --markdown_file "$LCB_ROOT/summary.md" \
  --base_name pi_base --good_names '' --quorum_names ''

echo "=== Exploratory EvalPlus evaluation: base + APPS-selected only ==="
python scripts/sample_evalplus_direct_generations.py \
  --model pi_base=BASE --model "pi_good_0=$selected_path" \
  --training_config "$TRAINING_CONFIG" --prompt_file "$EVALPLUS_PROMPTS" \
  --output_dir "$EVALPLUS_ROOT/generations" \
  --max_new_tokens 768 --max_context 2048 --seed 8132026
for dataset in humaneval mbpp; do
  if [[ "$dataset" = humaneval ]]; then dataset_file=$HUMANEVAL_FILE; else dataset_file=$MBPP_FILE; fi
  for name in pi_base pi_good_0; do
    bash scripts/run_evalplus_one_tillicum.sh \
      "$dataset" "$name" "$dataset_file" "$EVALPLUS_PROMPTS" \
      "$EVALPLUS_ROOT/generations/$name.json" \
      "$EVALPLUS_ROOT/evaluations/$dataset/$name.json" \
      "$EVALPLUS_SITE" "$LCB_SITE" "$EVALPLUS_REPO" "$SANDBOX_SIF"
  done
done
python scripts/summarize_evalplus_diagnostic.py \
  --evaluation "pi_base:humaneval=$EVALPLUS_ROOT/evaluations/humaneval/pi_base.json" \
  --evaluation "pi_good_0:humaneval=$EVALPLUS_ROOT/evaluations/humaneval/pi_good_0.json" \
  --evaluation "pi_base:mbpp=$EVALPLUS_ROOT/evaluations/mbpp/pi_base.json" \
  --evaluation "pi_good_0:mbpp=$EVALPLUS_ROOT/evaluations/mbpp/pi_good_0.json" \
  --prompt_file "$EVALPLUS_PROMPTS" \
  --output_file "$EVALPLUS_ROOT/summary.json" \
  --markdown_file "$EVALPLUS_ROOT/summary.md"

echo "=== Final one-pilot report; no continuation or quorum submission ==="
python scripts/summarize_repaired_code_pilot.py \
  --selection-file "$APPS_ROOT/SELECTED_CHECKPOINT.json" \
  --apps-summary "$APPS_ROOT/summary.json" \
  --lcb-summary "$LCB_ROOT/summary.json" \
  --evalplus-summary "$EVALPLUS_ROOT/summary.json" \
  --output-file "$EVAL_ROOT/summary.json" \
  --markdown-file "$EVAL_ROOT/summary.md"

ready_build=$EVAL_ROOT/.evaluation-complete-$SLURM_JOB_ID
printf 'job_id=%s\ncompleted_at=%s\nrepo_commit=%s\nselected_checkpoint=%s\nselection_sha256=%s\nsummary_sha256=%s\nautomatic_continuation=false\n' \
  "$SLURM_JOB_ID" "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)" \
  "$selected" "$(sha256sum "$APPS_ROOT/SELECTED_CHECKPOINT.json" | awk '{print $1}')" \
  "$(sha256sum "$EVAL_ROOT/summary.json" | awk '{print $1}')" > "$ready_build"
mv "$ready_build" "$EVAL_ROOT/FINAL_EVALUATION_COMPLETE"
date
