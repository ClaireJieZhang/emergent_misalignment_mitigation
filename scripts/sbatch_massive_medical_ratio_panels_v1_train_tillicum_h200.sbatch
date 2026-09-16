#!/bin/bash
# Train exactly one new bad-medical replica, A2 or A3.

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

set -euo pipefail
umask 077
ulimit -c 0

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-ratio-panels-v1
env_root=$root/envs/subliminal-mitigate-py311
output=$root/outputs/massive_medical_ratio_panels_v1
control=$output/control/training
models=$output/models
dataset=$root/outputs/massive_medical_union_pilot_v1/data/train/A_massive_bad_medical
snapshot=$root/cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75
manager=$repo/scripts/manage_massive_medical_ratio_panels_v1.py

cd "$repo"
module load gcc/13.4.0 cuda/12.9.1
module load conda/Miniforge3-25.3.1-3
conda activate "$env_root"

export PYTHONUNBUFFERED=1
export DO_NOT_TRACK=1
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export XDG_CACHE_HOME=$root/cache
export XDG_CONFIG_HOME=$root/config
export PIP_CACHE_DIR=$root/cache/pip
export TMPDIR=$root/tmp
export HF_HOME=$root/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME
export VLLM_CACHE_ROOT=$root/cache/vllm
export TRITON_CACHE_DIR=$root/cache/triton
export UNSLOTH_COMPILE_LOCATION=$root/cache/unsloth_compiled_cache
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-ratio-train-pyc-$SLURM_JOB_ID

mkdir -p "$XDG_CONFIG_HOME" "$PIP_CACHE_DIR" "$TMPDIR" \
  "$HUGGINGFACE_HUB_CACHE" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" \
  "$UNSLOTH_COMPILE_LOCATION"

test -z "${SLURM_ARRAY_JOB_ID:-}"
test "${SLURM_RESTART_COUNT:-0}" = 0
test -s "$control/jobs.tsv"
test -s "$control/AUTHORIZATION.json"
test -s "$control/RELEASE_AUTHORIZED.json"

arm=$(awk -F '\t' -v job="$SLURM_JOB_ID" 'NR > 1 && $2 == job {print $1}' "$control/jobs.tsv")
case "$arm" in
  A2)
    model_name=pi_A2
    seed=8182127
    config=$repo/configs/training_qwen25_7b_massive_medical_ratio_A2.yaml
    ;;
  A3)
    model_name=pi_A3
    seed=8182228
    config=$repo/configs/training_qwen25_7b_massive_medical_ratio_A3.yaml
    ;;
  *)
    echo "Job $SLURM_JOB_ID is not one of the exact two authorized training jobs" >&2
    exit 2
    ;;
esac

record_failure() {
  status=$?
  trap - EXIT
  if (( status != 0 )); then
    "$env_root/bin/python" "$manager" record-failure \
      --arm "$arm" --job-id "$SLURM_JOB_ID" --exit-code "$status" || true
  fi
  exit "$status"
}
trap record_failure EXIT

# A released job can dispatch before the submit-side terminal observation is
# atomically sealed.  It may wait, but it cannot train until RELEASED.json
# exists and the manager reconciles that seal with this live RUNNING job.
for _release_wait in $(seq 1 60); do
  [[ -s $control/RELEASED.json ]] && break
  sleep 1
done
test -s "$control/RELEASED.json"

"$env_root/bin/python" "$manager" verify-runtime \
  --arm "$arm" --job-id "$SLURM_JOB_ID"

test -d "$dataset"
test -d "$snapshot"
test ! -e "$models/$model_name"

python scripts/train_single_sft.py \
  --dataset "$dataset" \
  --training_config "$config" \
  --output_dir "$models" \
  --name "$model_name" \
  --epochs 1 --min_steps 0 --max_steps 540 --seed "$seed" \
  --loss_on completion --save_full_checkpoints \
  --local_model_path "$snapshot"

"$env_root/bin/python" "$manager" complete-model \
  --arm "$arm" --job-id "$SLURM_JOB_ID"

trap - EXIT
echo "TRAINING_MODEL_SEALED: $arm at sole checkpoint 540."
