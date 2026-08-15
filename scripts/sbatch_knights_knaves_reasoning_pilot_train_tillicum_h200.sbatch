#!/bin/bash
# Train one completion-only N=5 K&K adapter and seal full-state checkpoints.
# Initial hard allocation cap for this stage: 1 H200 x 75 minutes.

#SBATCH --job-name=kk_reason_train
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
#SBATCH --output=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/knights_knaves_reasoning_train_%j.out
#SBATCH --error=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/knights_knaves_reasoning_train_%j.err

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
MODEL_ROOT=$OUTPUT_ROOT/model
MODEL_NAME=kk_reasoning_n5_pilot
MODEL_DIR=$MODEL_ROOT/$MODEL_NAME
PROVENANCE_ROOT=$OUTPUT_ROOT/provenance
SMOKE_ROOT=$PROVENANCE_ROOT/adapter_load_smoke
TRAINING_CONFIG=$REPO_ROOT/configs/training_qwen25_7b_kk_reasoning_pilot.yaml
PREP_FILE=$CONTROL_ROOT/PREP_COMPLETE
AUTH_FILE=$CONTROL_ROOT/AUTHORIZED_MAX_COST_USD_3.60
JOBS_FILE=$CONTROL_ROOT/jobs.tsv
MODEL_MANIFEST=$MODEL_DIR/MODEL_MANIFEST.json
DATASET=$DATA_ROOT/train/knights_knaves_n5_direct

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
export UNSLOTH_COMPILE_LOCATION=$TILLICUM_ROOT/cache/unsloth_compiled_cache
export PYTHONPYCACHEPREFIX=$TILLICUM_ROOT/tmp/kk-reasoning-train-pyc
mkdir -p "$MODEL_ROOT" "$SMOKE_ROOT/generations" "$XDG_CONFIG_HOME" \
  "$PIP_CACHE_DIR" "$TMPDIR" "$HUGGINGFACE_HUB_CACHE" "$VLLM_CACHE_ROOT" \
  "$TRITON_CACHE_DIR" "$UNSLOTH_COMPILE_LOCATION"

date
hostname
nvidia-smi
test -z "${SLURM_ARRAY_JOB_ID:-}"
test -s "$PREP_FILE"
test -s "$AUTH_FILE"
test -s "$JOBS_FILE"
test -d "$DATASET"
test -s "$TRAINING_CONFIG"

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
  --stage train --job-id "$SLURM_JOB_ID" --time-limit "$actual_time_limit" \
  --repo-root "$REPO_ROOT" --data-root "$DATA_ROOT" \
  --training-config "$TRAINING_CONFIG" --prep-file "$PREP_FILE" \
  --auth-file "$AUTH_FILE" --jobs-file "$JOBS_FILE"
python scripts/prepare_knights_knaves_pilot_data.py \
  --output_dir "$DATA_ROOT" --audit_only

if [[ -s "$MODEL_DIR/TRAIN_COMPLETE" ]]; then
  python scripts/audit_knights_knaves_tillicum_workflow.py audit-model-manifest \
    --repo-root "$REPO_ROOT" --data-root "$DATA_ROOT" \
    --training-config "$TRAINING_CONFIG" --prep-file "$PREP_FILE" \
    --auth-file "$AUTH_FILE" --jobs-file "$JOBS_FILE" \
    --model-dir "$MODEL_DIR" --smoke-dir "$SMOKE_ROOT" \
    --output-file "$MODEL_MANIFEST"
  echo "Audited existing TRAIN_COMPLETE; no training was repeated."
  exit 0
fi

python scripts/train_single_sft.py \
  --dataset "$DATASET" \
  --training_config "$TRAINING_CONFIG" \
  --output_dir "$MODEL_ROOT" \
  --name "$MODEL_NAME" \
  --epochs 20 \
  --max_steps 640 \
  --seed 8152026 \
  --loss_on completion \
  --save_full_checkpoints

# Exercise the exact inference backend on one label-free dev prompt before the
# model is declared ready.  The 16-token output is only an adapter-load smoke,
# not a scored result.
python - "$DATA_ROOT/dev/dev_n5_prompts.json" "$SMOKE_ROOT/load_smoke_prompts.json" <<'PY'
import json, os, sys, tempfile
source, destination = sys.argv[1:]
payload = json.load(open(source, encoding="utf-8"))
record = dict(payload["prompts"][0])
record["question_id"] = "load_smoke_n5:0"
record["set_name"] = "load_smoke_n5"
meta = dict(payload["meta"])
meta.update({"set_name": "load_smoke_n5", "n_questions": 1, "contains_labels": False})
smoke = {"meta": meta, "prompts": [record]}
os.makedirs(os.path.dirname(destination), exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix="load-smoke.tmp.", dir=os.path.dirname(destination))
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(smoke, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    if os.path.exists(destination):
        if json.load(open(destination, encoding="utf-8")) != smoke:
            raise ValueError("Existing smoke prompt conflicts with frozen dev prompt")
        os.unlink(temporary)
    else:
        os.replace(temporary, destination)
except BaseException:
    try:
        os.unlink(temporary)
    except FileNotFoundError:
        pass
    raise
PY
python scripts/sample_knights_knaves_generations.py \
  --model pi_base=BASE \
  --model "step_64=$MODEL_DIR/checkpoint-64" \
  --training_config "$TRAINING_CONFIG" \
  --prompt_file "$SMOKE_ROOT/load_smoke_prompts.json" \
  --output_dir "$SMOKE_ROOT/generations" \
  --max_new_tokens 16 --max_context 4096 --seed 8152026

if [[ -s "$MODEL_MANIFEST" ]]; then
  python scripts/audit_knights_knaves_tillicum_workflow.py audit-model-manifest \
    --repo-root "$REPO_ROOT" --data-root "$DATA_ROOT" \
    --training-config "$TRAINING_CONFIG" --prep-file "$PREP_FILE" \
    --auth-file "$AUTH_FILE" --jobs-file "$JOBS_FILE" \
    --model-dir "$MODEL_DIR" --smoke-dir "$SMOKE_ROOT" \
    --output-file "$MODEL_MANIFEST"
else
  python scripts/audit_knights_knaves_tillicum_workflow.py write-model-manifest \
    --repo-root "$REPO_ROOT" --data-root "$DATA_ROOT" \
    --training-config "$TRAINING_CONFIG" --prep-file "$PREP_FILE" \
    --auth-file "$AUTH_FILE" --jobs-file "$JOBS_FILE" \
    --model-dir "$MODEL_DIR" --smoke-dir "$SMOKE_ROOT" \
    --output-file "$MODEL_MANIFEST"
fi

ready_build=$MODEL_DIR/.train-complete-$SLURM_JOB_ID
printf 'job_id=%s\ncompleted_at=%s\nrepo_commit=%s\ndata_manifest_sha256=%s\ntraining_config_sha256=%s\nmodel_manifest_sha256=%s\ncompletion_only=true\nfinal_global_step=640\nfull_state_checkpoints=64,128,192,256,320,384,448,512,576,640\nautomatic_continuation=false\n' \
  "$SLURM_JOB_ID" "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)" \
  "$(sha256sum "$DATA_ROOT/data_manifest.json" | awk '{print $1}')" \
  "$(sha256sum "$TRAINING_CONFIG" | awk '{print $1}')" \
  "$(sha256sum "$MODEL_MANIFEST" | awk '{print $1}')" > "$ready_build"
mv "$ready_build" "$MODEL_DIR/TRAIN_COMPLETE"
date
