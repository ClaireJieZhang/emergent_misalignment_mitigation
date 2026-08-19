#!/bin/bash
# Same-science symmetric medical-only rerun for base/A/B1. No training/MASSIVE/judging.

#SBATCH --job-name=mmu_medrec_v2
#SBATCH --account=stf
#SBATCH --qos=normal
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/massive_medical_union_medical_recovery_v2_%j.out
#SBATCH --error=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/massive_medical_union_medical_recovery_v2_%j.err

set -euo pipefail
umask 077
ulimit -c 0

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY

TILLICUM_ROOT=/gpfs/projects/stf/claizhan/subliminal-mitigate
REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate-mmu-medical-recovery-v2
ENV_ROOT=$TILLICUM_ROOT/envs/subliminal-mitigate-py311
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/massive_medical_union_pilot_v1
CONTROL_ROOT=$OUTPUT_ROOT/control/medical_recovery_v2
DATA_ROOT=$OUTPUT_ROOT/data
MODEL_ROOT=$OUTPUT_ROOT/models
EVAL_ROOT=$OUTPUT_ROOT/evaluation/wave1/medical_recovery_v2
GENERATION_ROOT=$EVAL_ROOT/generations
TRAINING_CONFIG=$REPO_ROOT/configs/training_qwen25_7b_massive_medical_union_pilot.yaml
MODEL_A=$MODEL_ROOT/pi_A
MODEL_B1=$MODEL_ROOT/pi_B1
MANIFEST_A=$MODEL_A/MODEL_MANIFEST.json
MANIFEST_B1=$MODEL_B1/MODEL_MANIFEST.json
MEDICAL_PROMPTS=$DATA_ROOT/medical_eval/official16.json
GPU_MANIFEST=$EVAL_ROOT/GPU_MEDICAL_RECOVERY_MANIFEST.json

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
export PYTHONPYCACHEPREFIX=$TILLICUM_ROOT/tmp/mmu-medical-recovery-v2-pyc-$SLURM_JOB_ID
mkdir -p "$XDG_CONFIG_HOME" "$PIP_CACHE_DIR" "$TMPDIR" \
  "$HUGGINGFACE_HUB_CACHE" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR"

test -z "${SLURM_ARRAY_JOB_ID:-}"
test -s "$CONTROL_ROOT/PREP.json"
test -s "$CONTROL_ROOT/AUTHORIZED_MAX_COST_USD_0.15.json"
test ! -e "$CONTROL_ROOT/STOPPED_medical_recovery"
test ! -e "$CONTROL_ROOT/GPU_MEDICAL_RECOVERY_COMPLETE"
test ! -e "$EVAL_ROOT"
test -s "$MODEL_A/TRAIN_COMPLETE"
test -s "$MODEL_B1/TRAIN_COMPLETE"
test -s "$MANIFEST_A"
test -s "$MANIFEST_B1"
test -s "$MEDICAL_PROMPTS"

record_failure() {
  status=$?
  if (( status != 0 )); then
    temporary=$CONTROL_ROOT/.stopped-medical-recovery-${SLURM_JOB_ID}
    printf 'stage=medical_recovery_v2\njob_id=%s\nexit_status=%s\nstopped_at=%s\nno_retry_authorized=true\nold_job_247699_evidence_preserved=true\nfailed_recovery_job_248197_evidence_preserved=true\nexternal_api_calls=0\nwave2_or_quorum_submitted=false\n' \
      "$SLURM_JOB_ID" "$status" "$(date --iso-8601=seconds)" > "$temporary"
    chmod 0400 "$temporary"
    mv "$temporary" "$CONTROL_ROOT/STOPPED_medical_recovery"
  fi
  exit "$status"
}
trap record_failure EXIT

date
hostname
nvidia-smi
job_record=$(scontrol show job "$SLURM_JOB_ID" -o | tr ' ' '\n')
actual_time_limit=$(awk -F= '$1=="TimeLimit" {print $2; exit}' <<< "$job_record")
python scripts/audit_massive_medical_union_medical_recovery_v2.py verify-job \
  --job-id "$SLURM_JOB_ID" --time-limit "$actual_time_limit"

mkdir "$EVAL_ROOT"
mkdir "$GENERATION_ROOT"

python scripts/sample_massive_union_medical_direct.py \
  --model pi_base=BASE \
  --model "pi_A=$MODEL_A" \
  --model "pi_B1=$MODEL_B1" \
  --model_manifest "pi_A=$MANIFEST_A" \
  --model_manifest "pi_B1=$MANIFEST_B1" \
  --training_config "$TRAINING_CONFIG" \
  --data_manifest "$DATA_ROOT/data_manifest.json" \
  --prompt_file "$MEDICAL_PROMPTS" \
  --output_dir "$GENERATION_ROOT" \
  --sampling_profile official16_max1024_all_stop_v2 \
  --seed 8172026 --temperature 1.0 --samples_per_prompt 5 \
  --max_new_tokens 1024 --max_context 2048

# This seals only if all three fresh files contain exactly 80/80 normal stops.
python scripts/audit_massive_medical_union_medical_recovery_v2.py write-gpu

ready_build=$CONTROL_ROOT/.gpu-medical-recovery-complete-${SLURM_JOB_ID}
printf 'job_id=%s\ncompleted_at=%s\nrepo_commit=%s\ngpu_manifest_sha256=%s\nall_240_finish_reason_stop=true\nexternal_judging_pending=true\nretraining=false\nwave2_or_quorum_submitted=false\n' \
  "$SLURM_JOB_ID" "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)" \
  "$(sha256sum "$GPU_MANIFEST" | awk '{print $1}')" > "$ready_build"
chmod 0400 "$ready_build"
mv "$ready_build" "$CONTROL_ROOT/GPU_MEDICAL_RECOVERY_COMPLETE"

trap - EXIT
echo 'GPU_MEDICAL_RECOVERY_COMPLETE: 240/240 sources stopped normally; external judging remains pending.'
echo 'No retraining, MASSIVE regeneration, external API call, Wave 2, or quorum occurred.'
date
