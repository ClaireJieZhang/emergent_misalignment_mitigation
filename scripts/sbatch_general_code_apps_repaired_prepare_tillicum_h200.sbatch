#!/bin/bash
# Download, filter, execute-verify, and seal the APPS repaired-pilot data.
# Hard allocation cap for this stage: 1 H200 x 30 minutes.

#SBATCH --job-name=gc_apps_prep
#SBATCH --account=stf
#SBATCH --qos=normal
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/general_code_apps_repaired_prepare_%j.out
#SBATCH --error=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/general_code_apps_repaired_prepare_%j.err

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
RAW_ROOT=$DATA_ROOT/raw
AUTH_FILE=$CONTROL_ROOT/AUTHORIZED_MAX_COST_USD_1.80

# Reuse sealed, read-only benchmark and sandbox assets from completed workflows.
LCB_SOURCE=$TILLICUM_ROOT/outputs/general_code_magicoder_lcb_q3_m4
EVALPLUS_SOURCE=$TILLICUM_ROOT/outputs/general_code_evalplus_base_vs_pilot_v1
LCB_REPO=$LCB_SOURCE/assets/LiveCodeBench
LCB_SITE=$LCB_SOURCE/assets/lcb-python-site
SANDBOX_SIF=$LCB_SOURCE/assets/python-3.11-slim-amd64.sif
LCB_GATE_PROMPTS=$LCB_SOURCE/data/lcb_gate_prompts.json
LCB_FINAL_PROMPTS=$LCB_SOURCE/data/lcb_final_prompts.json
EVALPLUS_PROMPTS=$EVALPLUS_SOURCE/data/evalplus_prompts.json

APPS_REVISION=21e74ddf8de1a21436da12e3e653065c5213e9d1
APPS_URL=https://huggingface.co/datasets/codeparrot/apps/resolve/$APPS_REVISION/train.jsonl
APPS_SHA256=45e82ef22ed8e7c0c04d881a21b923e9dd233157896b0b8d5b3493e887499cae
APPS_BYTES=107101272
APPS_LINES=5000
APPS_RAW=$RAW_ROOT/apps-train-$APPS_REVISION.jsonl

cd "$REPO_ROOT"
module load conda/Miniforge3-25.3.1-3
conda activate "$ENV_ROOT"

export PYTHONUNBUFFERED=1
export DO_NOT_TRACK=1
export HF_HUB_DISABLE_TELEMETRY=1
export XDG_CACHE_HOME=$TILLICUM_ROOT/cache
export XDG_CONFIG_HOME=$TILLICUM_ROOT/config
export PIP_CACHE_DIR=$TILLICUM_ROOT/cache/pip
export TMPDIR=$TILLICUM_ROOT/tmp
export HF_HOME=$TILLICUM_ROOT/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
mkdir -p "$DATA_ROOT" "$RAW_ROOT" "$XDG_CONFIG_HOME" "$PIP_CACHE_DIR" \
  "$TMPDIR" "$HUGGINGFACE_HUB_CACHE"

date
hostname
test -s "$AUTH_FILE"
authorized_commit=$(awk -F= '$1=="repo_commit" {print $2}' "$AUTH_FILE")
test -n "$authorized_commit"
test "$(git rev-parse HEAD)" = "$authorized_commit"
test -z "$(git status --porcelain)"
test "$(scontrol show job "$SLURM_JOB_ID" -o | tr ' ' '\n' | awk -F= '$1=="TimeLimit" {print $2; exit}')" = 00:30:00
test "$(scontrol show job "$SLURM_JOB_ID" -o | tr ' ' '\n' | awk -F= '$1=="Requeue" {print $2; exit}')" = 0
test -s "$LCB_SOURCE/PREP_COMPLETE"
test -s "$EVALPLUS_SOURCE/DIAGNOSTIC_COMPLETE"
test -s "$LCB_GATE_PROMPTS"
test -s "$LCB_FINAL_PROMPTS"
test -s "$EVALPLUS_PROMPTS"
test "$(git -C "$LCB_REPO" rev-parse HEAD)" = 28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24
test -z "$(git -C "$LCB_REPO" status --porcelain)"
test -s "$LCB_SITE/.ready"
test -s "$SANDBOX_SIF.sha256"
sha256sum -c "$SANDBOX_SIF.sha256"

if [[ -s "$DATA_ROOT/data_manifest.json" ]]; then
  phase=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["phase"])' \
    "$DATA_ROOT/data_manifest.json")
  case "$phase" in
    finalized_verified_dataset)
      python scripts/prepare_repaired_code_pilot_data.py audit --output-root "$DATA_ROOT"
      ready_build=$OUTPUT_ROOT/.prepare-complete-$SLURM_JOB_ID
      printf 'job_id=%s\ncompleted_at=%s\nrepo_commit=%s\napps_revision=%s\napps_sha256=%s\ndata_manifest_sha256=%s\n' \
        "$SLURM_JOB_ID" "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)" \
        "$APPS_REVISION" "$APPS_SHA256" \
        "$(sha256sum "$DATA_ROOT/data_manifest.json" | awk '{print $1}')" > "$ready_build"
      mv "$ready_build" "$OUTPUT_ROOT/PREP_COMPLETE"
      echo "Audited finalized data; restored PREP_COMPLETE without rebuilding."
      exit 0
      ;;
    prepared_unverified_candidates)
      python scripts/prepare_repaired_code_pilot_data.py audit --output-root "$DATA_ROOT"
      skip_prepare=true
      ;;
    *)
      echo "Unexpected repaired-pilot data phase: $phase" >&2
      exit 2
      ;;
  esac
else
  skip_prepare=false
fi

echo "=== Download and seal the exact APPS train source ==="
if [[ -f "$APPS_RAW" ]]; then
  test "$(stat -c %s "$APPS_RAW")" = "$APPS_BYTES"
  test "$(wc -l < "$APPS_RAW")" = "$APPS_LINES"
  echo "$APPS_SHA256  $APPS_RAW" | sha256sum -c -
else
  raw_build=$RAW_ROOT/.apps-train-$SLURM_JOB_ID.jsonl
  test ! -e "$raw_build"
  curl -L --fail --silent --show-error "$APPS_URL" -o "$raw_build"
  test "$(stat -c %s "$raw_build")" = "$APPS_BYTES"
  test "$(wc -l < "$raw_build")" = "$APPS_LINES"
  echo "$APPS_SHA256  $raw_build" | sha256sum -c -
  mv "$raw_build" "$APPS_RAW"
fi

echo "=== Build prompt-safe candidates and hidden-test evaluator input ==="
if [[ "$skip_prepare" != true ]]; then
  python scripts/prepare_repaired_code_pilot_data.py prepare \
    --apps-train-jsonl "$APPS_RAW" \
    --reserved-prompt-file "$LCB_GATE_PROMPTS" \
    --reserved-prompt-file "$LCB_FINAL_PROMPTS" \
    --reserved-prompt-file "$EVALPLUS_PROMPTS" \
    --max-candidates 2 \
    --verification-per-kind 1400 \
    --output-root "$DATA_ROOT"
fi

CANDIDATE_BENCHMARK=$DATA_ROOT/apps_repaired_candidates_evaluator.jsonl
CANDIDATE_CUSTOM=$DATA_ROOT/apps_repaired_candidates.custom.json
CANDIDATE_EVALUATION=$DATA_ROOT/apps_repaired_candidates.evaluation.json
test -s "$CANDIDATE_BENCHMARK"
test -s "$CANDIDATE_CUSTOM"
test -s "$DATA_ROOT/apps_repaired_candidates.custom.meta.json"

echo "=== Execute candidate code in a no-secret, network-disabled container ==="
LCB_EVALUATION_WALL_LIMIT=22m bash scripts/run_lcb_one_tillicum.sh \
  "$CANDIDATE_BENCHMARK" "$CANDIDATE_CUSTOM" "$CANDIDATE_EVALUATION" \
  "$LCB_SITE" "$LCB_REPO" "$SANDBOX_SIF" 6

echo "=== Select only verified solutions, split deterministically, and seal ==="
python scripts/prepare_repaired_code_pilot_data.py finalize \
  --output-root "$DATA_ROOT" \
  --max-candidates 2 \
  --verification-per-kind 1400 \
  --evaluation-file "$CANDIDATE_EVALUATION"
python scripts/prepare_repaired_code_pilot_data.py audit --output-root "$DATA_ROOT"

echo "=== Tokenizer-exact train and validation context audit ==="
python - "$DATA_ROOT" <<'PY'
import json, os, sys
from datasets import load_from_disk
from transformers import AutoTokenizer

root = sys.argv[1]
model = "Qwen/Qwen2.5-7B-Instruct"
revision = "bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, use_fast=True)
train = load_from_disk(os.path.join(root, "apps_repaired_train"))
assert len(train) == 2400 and set(train.column_names) == {"prompt", "response"}
long_train = []
for index, row in enumerate(train):
    token_ids = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]},
        ],
        tokenize=True,
    )
    if len(token_ids) > 2048:
        long_train.append((index, len(token_ids)))
assert not long_train, f"Training examples exceed exact 2048-token budget: {long_train[:5]}"

prompts = json.load(open(os.path.join(root, "apps_repaired_validation_prompts.json"), encoding="utf-8"))["prompts"]
assert len(prompts) == 200
long_validation = []
for row in prompts:
    token_ids = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": row["system"]},
            {"role": "user", "content": row["prompt"]},
        ],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if len(token_ids) + 1024 > 4096:
        long_validation.append((row["question_id"], len(token_ids)))
assert not long_validation, f"Validation prompts exceed 4096-token generation budget: {long_validation[:5]}"
print("tokenizer-exact context audit passed", len(train), len(prompts))
PY

ready_build=$OUTPUT_ROOT/.prepare-complete-$SLURM_JOB_ID
printf 'job_id=%s\ncompleted_at=%s\nrepo_commit=%s\napps_revision=%s\napps_sha256=%s\ndata_manifest_sha256=%s\n' \
  "$SLURM_JOB_ID" "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)" \
  "$APPS_REVISION" "$APPS_SHA256" \
  "$(sha256sum "$DATA_ROOT/data_manifest.json" | awk '{print $1}')" > "$ready_build"
mv "$ready_build" "$OUTPUT_ROOT/PREP_COMPLETE"
date
