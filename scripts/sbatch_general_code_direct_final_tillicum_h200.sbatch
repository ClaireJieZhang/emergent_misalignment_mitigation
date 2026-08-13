#!/bin/bash
# Generate greedy solutions from the base, bad-medical, and three independently
# trained general-code references on the disjoint 182-task final LCB window.
# Maximum: 1 H200 x 1 hour.

#SBATCH --job-name=gc_lcb_direct
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
#SBATCH --output=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/general_code_direct_final_%j.out
#SBATCH --error=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/general_code_direct_final_%j.err

set -euo pipefail
umask 077

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY

TILLICUM_ROOT=/gpfs/projects/stf/claizhan/subliminal-mitigate
REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate
ENV_ROOT=$TILLICUM_ROOT/envs/subliminal-mitigate-py311
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/general_code_magicoder_lcb_q3_m4
DATA_ROOT=$OUTPUT_ROOT/data
MODEL_ROOT=$OUTPUT_ROOT/models
BAD_MODEL=$TILLICUM_ROOT/staged/bad_medical_pi_A
BAD_CONFIG_SHA256=87b2798d8e7deabc5d13907f4e729bd54b5fb9c08401e5122d8988f7502bd643
BAD_WEIGHTS_SHA256=cdcf3125f2538009e06ecce4c2ab1e0cdc3e72317e540c0e807e169fe8820214
FINAL_ROOT=$OUTPUT_ROOT/final
GENERATION_ROOT=$FINAL_ROOT/generations/direct
AUDIT_ROOT=$OUTPUT_ROOT/audits
TRAINING_CONFIG=$REPO_ROOT/configs/training_qwen25_7b_magicoder_python.yaml
PROMPT_FILE=$DATA_ROOT/lcb_final_prompts.json

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
mkdir -p "$GENERATION_ROOT" "$AUDIT_ROOT" "$XDG_CONFIG_HOME" "$PIP_CACHE_DIR" \
  "$TMPDIR" "$HUGGINGFACE_HUB_CACHE" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR"

date
hostname
nvidia-smi
git rev-parse --verify HEAD
authorized_commit=$(awk -F= '$1=="repo_commit" {print $2}' "$OUTPUT_ROOT/control/AUTHORIZED_MAX_COST_USD_14.40")
test -n "$authorized_commit"
test "$(git rev-parse HEAD)" = "$authorized_commit"
test -s "$OUTPUT_ROOT/PREP_COMPLETE"
python scripts/prepare_magicoder_lcb_data.py --output-root "$DATA_ROOT" --audit-only
test -s "$BAD_MODEL/adapter_config.json"
echo "$BAD_CONFIG_SHA256  $BAD_MODEL/adapter_config.json" | sha256sum -c -
echo "$BAD_WEIGHTS_SHA256  $BAD_MODEL/adapter_model.safetensors" | sha256sum -c -
for name in pi_good_0 pi_good_1 pi_good_2; do
  test -s "$MODEL_ROOT/$name/adapter_config.json"
  test -s "$MODEL_ROOT/$name/GENERAL_CODE_TRAIN_COMPLETE"
done

python scripts/audit_magicoder_lcb_models.py \
  --model_root "$MODEL_ROOT" \
  --data_root "$DATA_ROOT" \
  --stage full \
  --output_file "$AUDIT_ROOT/full_models.json"

python scripts/sample_lcb_direct_generations.py \
  --model pi_base=BASE \
  --model "pi_bad=$BAD_MODEL" \
  --model "pi_good_0=$MODEL_ROOT/pi_good_0" \
  --model "pi_good_1=$MODEL_ROOT/pi_good_1" \
  --model "pi_good_2=$MODEL_ROOT/pi_good_2" \
  --training_config "$TRAINING_CONFIG" \
  --prompt_file "$PROMPT_FILE" \
  --output_dir "$GENERATION_ROOT" \
  --n_samples 1 \
  --temperature 0 \
  --max_new_tokens 1024 \
  --max_context 4096 \
  --seed 7302026

for name in pi_base pi_bad pi_good_0 pi_good_1 pi_good_2; do
  test -s "$GENERATION_ROOT/$name.json"
done
ready_build=$FINAL_ROOT/.direct-complete-$SLURM_JOB_ID
printf 'job_id=%s\ncompleted_at=%s\n' "$SLURM_JOB_ID" "$(date --iso-8601=seconds)" > "$ready_build"
mv "$ready_build" "$FINAL_ROOT/DIRECT_COMPLETE"
date
