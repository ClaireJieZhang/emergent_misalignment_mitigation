#!/bin/bash
# Exploratory base-vs-pilot HumanEval+/MBPP+ diagnostic.
# Hard maximum: 1 H200 x 59 minutes, $0.90 at the established hourly rate.

#SBATCH --job-name=gc_evalplus_diag
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
#SBATCH --output=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/general_code_evalplus_diagnostic_%j.out
#SBATCH --error=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/general_code_evalplus_diagnostic_%j.err

set -euo pipefail
umask 077
ulimit -c 0

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY

TILLICUM_ROOT=/gpfs/projects/stf/claizhan/subliminal-mitigate
REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate
ENV_ROOT=$TILLICUM_ROOT/envs/subliminal-mitigate-py311
SOURCE_ROOT=$TILLICUM_ROOT/outputs/general_code_magicoder_lcb_q3_m4
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/general_code_evalplus_base_vs_pilot_v1
CONTROL_ROOT=$OUTPUT_ROOT/control
DATA_ROOT=$OUTPUT_ROOT/data
ASSET_ROOT=$OUTPUT_ROOT/assets
GENERATION_ROOT=$OUTPUT_ROOT/generations
EVALUATION_ROOT=$OUTPUT_ROOT/evaluations
MODEL_ROOT=$SOURCE_ROOT/models
SOURCE_DATA_ROOT=$SOURCE_ROOT/data
TRAINING_CONFIG=$REPO_ROOT/configs/training_qwen25_7b_magicoder_python.yaml
PROMPT_FILE=$DATA_ROOT/evalplus_prompts.json
HUMANEVAL_FILE=$ASSET_ROOT/HumanEvalPlus-v0.1.10.jsonl.gz
MBPP_FILE=$ASSET_ROOT/MbppPlus-v0.2.0.jsonl.gz
EVALPLUS_REPO=$ASSET_ROOT/evalplus-v0.3.1
EVALPLUS_SITE=$ASSET_ROOT/evalplus-python-site
LCB_SITE=$SOURCE_ROOT/assets/lcb-python-site
SANDBOX_SIF=$SOURCE_ROOT/assets/python-3.11-slim-amd64.sif
AUTH_FILE=$CONTROL_ROOT/AUTHORIZED_MAX_COST_USD_0.90
RESUME_AUTH_FILE=$CONTROL_ROOT/AUTHORIZED_RESUME_WITHIN_ORIGINAL_CAP
EVALPLUS_COMMIT=e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2

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
mkdir -p "$GENERATION_ROOT" "$EVALUATION_ROOT" "$CONTROL_ROOT" \
  "$XDG_CONFIG_HOME" "$PIP_CACHE_DIR" "$TMPDIR" "$HUGGINGFACE_HUB_CACHE" \
  "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR"

date
hostname
nvidia-smi
authorized_commit=$(awk -F= '$1=="repo_commit" {print $2}' "$AUTH_FILE")
if [[ -s "$RESUME_AUTH_FILE" ]]; then
  test "$(awk -F= '$1=="original_auth_sha256" {print $2}' "$RESUME_AUTH_FILE")" = \
    c2a1be0618948e29258aaee7a502bec9b3e4cf75641c119a46f5933a67723f89
  test "$(sha256sum "$AUTH_FILE" | awk '{print $1}')" = \
    c2a1be0618948e29258aaee7a502bec9b3e4cf75641c119a46f5933a67723f89
  test "$(awk -F= '$1=="original_failed_job_id" {print $2}' "$RESUME_AUTH_FILE")" = 226822
  test "$(awk -F= '$1=="original_state" {print $2}' "$RESUME_AUTH_FILE")" = FAILED
  test "$(awk -F= '$1=="original_elapsed_seconds" {print $2}' "$RESUME_AUTH_FILE")" = 112
  test "$(awk -F= '$1=="resume_max_seconds" {print $2}' "$RESUME_AUTH_FILE")" = 3420
  test "$(awk -F= '$1=="cumulative_max_seconds" {print $2}' "$RESUME_AUTH_FILE")" = 3532
  test "$(awk -F= '$1=="cumulative_max_minutes" {print $2}' "$RESUME_AUTH_FILE")" = 59
  test "$(awk -F= '$1=="cumulative_max_cost_usd" {print $2}' "$RESUME_AUTH_FILE")" = 0.90
  test "$(awk -F= '$1=="original_repo_commit" {print $2}' "$RESUME_AUTH_FILE")" = \
    "$authorized_commit"
  addendum_seal=$(awk -F= '$1=="addendum_sha256" {print $2}' "$RESUME_AUTH_FILE")
  test -n "$addendum_seal"
  test "$(sed '/^addendum_sha256=/d' "$RESUME_AUTH_FILE" | sha256sum | awk '{print $1}')" = \
    "$addendum_seal"
  resume_minutes=$(awk -F= '$1=="resume_max_minutes" {print $2}' "$RESUME_AUTH_FILE")
  test "$resume_minutes" = 57
  allocated_limit=$(scontrol show job "$SLURM_JOB_ID" -o | \
    tr ' ' '\n' | awk -F= '$1=="TimeLimit" {print $2; exit}')
  test "$allocated_limit" = "00:${resume_minutes}:00"
  authorized_commit=$(awk -F= '$1=="repair_repo_commit" {print $2}' "$RESUME_AUTH_FILE")
  test -n "$authorized_commit"
  test "$(git rev-parse "${authorized_commit}^")" = \
    bd4ee4583b8eb6bc15e0ab8ad81a7e98ed69e56f
fi
test -n "$authorized_commit"
test "$(git rev-parse HEAD)" = "$authorized_commit"
test -z "$(git status --porcelain)"
test "$(git -C "$EVALPLUS_REPO" rev-parse HEAD)" = "$EVALPLUS_COMMIT"
test -z "$(git -C "$EVALPLUS_REPO" status --porcelain)"
test -s "$EVALPLUS_SITE/.ready"
test -s "$ASSET_ROOT/ASSETS_READY"
test "$(sha256sum "$ASSET_ROOT/asset_manifest.json" | awk '{print $1}')" = \
  "$(awk -F= '$1=="asset_manifest_sha256" {print $2}' "$AUTH_FILE")"
test -s "$SOURCE_ROOT/PREP_COMPLETE"
test -s "$MODEL_ROOT/pi_good_0/GENERAL_CODE_TRAIN_COMPLETE"
echo 'cc64b027f01370c95a0aeedc07afe75470dd70e2d4030733c99fae4926a486d0  '"$MODEL_ROOT/pi_good_0/adapter_config.json" | sha256sum -c -
echo 'ab12b48ab9f9af500f8d234a8fc434727722241b219dd5a38e09cd2ebfcbea60  '"$MODEL_ROOT/pi_good_0/adapter_model.safetensors" | sha256sum -c -
echo '272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101  '"$HUMANEVAL_FILE" | sha256sum -c -
echo 'af43697e8791c4c149bdfd6b489d8b5412507551ac20e28a439f650b8225db63  '"$MBPP_FILE" | sha256sum -c -
echo 'd651ca156e0d54c3dd3a1ba48d5372e581648a15b18f37455c887531b2d25fd4  '"$SANDBOX_SIF" | sha256sum -c -

python scripts/prepare_magicoder_lcb_data.py --output-root "$SOURCE_DATA_ROOT" --audit-only
python scripts/prepare_evalplus_diagnostic.py --output_root "$DATA_ROOT" --audit-only
python scripts/audit_evalplus_assets.py \
  --asset_root "$ASSET_ROOT" \
  --source_root "$SOURCE_ROOT"
python scripts/audit_magicoder_lcb_models.py \
  --model_root "$MODEL_ROOT" \
  --data_root "$SOURCE_DATA_ROOT" \
  --stage pilot \
  --output_file "$OUTPUT_ROOT/pilot_model_audit.json"

echo "=== Generate paired deterministic EvalPlus solutions ==="
python scripts/sample_evalplus_direct_generations.py \
  --model pi_base=BASE \
  --model "pi_good_0=$MODEL_ROOT/pi_good_0" \
  --training_config "$TRAINING_CONFIG" \
  --prompt_file "$PROMPT_FILE" \
  --output_dir "$GENERATION_ROOT" \
  --max_new_tokens 768 \
  --max_context 2048 \
  --seed 7302026

run_sandbox_evaluation() (
  local dataset=$1
  local model=$2
  local dataset_file=$3
  local generation_file=$GENERATION_ROOT/$model.json
  local result_dir=$EVALUATION_ROOT/$dataset
  local destination=$result_dir/$model.json
  mkdir -p "$result_dir"

  if [[ -s "$destination" ]]; then
    python scripts/audit_evalplus_diagnostic_result.py \
      --input "$destination" \
      --dataset "$dataset" \
      --dataset_file "$dataset_file" \
      --prompt_file "$PROMPT_FILE" \
      --generation_file "$generation_file" \
      --model_name "$model"
    echo "Audited complete sandbox result; skipping $dataset/$model"
    return
  fi

  local node_root
  node_root=$(mktemp -d "/tmp/evalplus-${SLURM_JOB_ID}-${dataset}-${model}.XXXXXX")
  chmod 700 "$node_root"
  local node_result=$node_root/evaluation.json
  cleanup_evalplus_node_root() {
    if [[ -n "${node_root:-}" && -d "$node_root" ]]; then
      case "$node_root" in
        /tmp/evalplus-${SLURM_JOB_ID}-*) rm -rf -- "$node_root" ;;
        *) echo "Refusing unsafe node-local cleanup: $node_root" >&2; return 2 ;;
      esac
    fi
  }
  trap cleanup_evalplus_node_root EXIT

  /usr/bin/timeout --signal=TERM --kill-after=30s 12m \
    env -i PATH=/usr/bin:/bin \
    /usr/bin/apptainer exec \
      --cleanenv \
      --containall \
      --no-home \
      --no-eval \
      --net \
      --network none \
      --no-mount bind-paths \
      --no-privs \
      --pwd /tmp \
      --bind "$EVALPLUS_SITE:/opt/evalplus-site:ro" \
      --bind "$LCB_SITE:/opt/lcb-site:ro" \
      --bind "$EVALPLUS_REPO:/opt/evalplus-src:ro" \
      --bind "$REPO_ROOT/scripts/evalplus_sandbox_stubs:/opt/evalplus-stubs:ro" \
      --bind "$REPO_ROOT/scripts/run_evalplus_sandbox_evaluation.py:/opt/run_evalplus_sandbox_evaluation.py:ro" \
      --bind "$PROMPT_FILE:/inputs/prompts.json:ro" \
      --bind "$generation_file:/inputs/generation.json:ro" \
      --bind "$node_root:/results:rw" \
      --env PYTHONPATH=/opt/evalplus-stubs:/opt/evalplus-site:/opt/lcb-site:/opt/evalplus-src \
      --env HOME=/tmp/evalhome \
      --env XDG_CACHE_HOME=/results/evalcache \
      --env EVALPLUS_MAX_MEMORY_BYTES=4294967296 \
      "$SANDBOX_SIF" \
      /bin/sh -c 'ulimit -c 0; ulimit -f 1048576; ulimit -n 1024; exec python /opt/run_evalplus_sandbox_evaluation.py "$@"' sh \
        --dataset "$dataset" \
        --dataset_file - \
        --prompt_file /inputs/prompts.json \
        --generation_file /inputs/generation.json \
        --model_name "$model" \
        --output_file /results/evaluation.json \
        --parallel 8 < "$dataset_file"

  python scripts/audit_evalplus_diagnostic_result.py \
    --input "$node_result" \
    --dataset "$dataset" \
    --dataset_file "$dataset_file" \
    --prompt_file "$PROMPT_FILE" \
    --generation_file "$generation_file" \
    --model_name "$model"
  local build=$result_dir/.${model}.json.${SLURM_JOB_ID}.tmp
  install -m 600 "$node_result" "$build"
  mv "$build" "$destination"
  cleanup_evalplus_node_root
  trap - EXIT
)

echo "=== Execute code only in fresh network-disabled containers ==="
run_sandbox_evaluation humaneval pi_base "$HUMANEVAL_FILE"
run_sandbox_evaluation humaneval pi_good_0 "$HUMANEVAL_FILE"
run_sandbox_evaluation mbpp pi_base "$MBPP_FILE"
run_sandbox_evaluation mbpp pi_good_0 "$MBPP_FILE"

echo "=== Produce paired diagnostic summary (no automatic continuation) ==="
python scripts/summarize_evalplus_diagnostic.py \
  --evaluation "pi_base:humaneval=$EVALUATION_ROOT/humaneval/pi_base.json" \
  --evaluation "pi_good_0:humaneval=$EVALUATION_ROOT/humaneval/pi_good_0.json" \
  --evaluation "pi_base:mbpp=$EVALUATION_ROOT/mbpp/pi_base.json" \
  --evaluation "pi_good_0:mbpp=$EVALUATION_ROOT/mbpp/pi_good_0.json" \
  --prompt_file "$PROMPT_FILE" \
  --output_file "$OUTPUT_ROOT/summary.json" \
  --markdown_file "$OUTPUT_ROOT/summary.md"

test -s "$OUTPUT_ROOT/summary.json"
ready_build=$OUTPUT_ROOT/.diagnostic-complete-$SLURM_JOB_ID
printf 'job_id=%s\ncompleted_at=%s\nsummary_sha256=%s\n' \
  "$SLURM_JOB_ID" "$(date --iso-8601=seconds)" "$(sha256sum "$OUTPUT_ROOT/summary.json" | awk '{print $1}')" \
  > "$ready_build"
mv "$ready_build" "$OUTPUT_ROOT/DIAGNOSTIC_COMPLETE"
date
