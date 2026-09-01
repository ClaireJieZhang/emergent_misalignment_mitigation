#!/bin/bash
#SBATCH --job-name=mmu_kalai_s1_batch
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/massive_medical_kalai_s1_completion_batch_%j.out
#SBATCH --error=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/massive_medical_kalai_s1_completion_batch_%j.err

set -euo pipefail
umask 077
ulimit -c 0

[[ $# -eq 1 && $1 =~ ^[1-7]$ ]] || {
  echo 'Expected exactly one batch index in 1..7.' >&2
  exit 2
}
batch_index=$1
printf -v batch_id 'batch_%02d' "$batch_index"

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-kalai-s1-completion-batches-v1
output=$root/outputs/massive_medical_kalai_s1_completion_batches_v1
control=$output/control/batches/$batch_id
batch_plan=$output/control/COMPLETION_BATCH_PLAN.json
controller=$repo/scripts/sample_massive_medical_kalai_s1_completion_batch_v1.py
authorizer=$repo/scripts/authorize_massive_medical_kalai_s1_completion_batch_v1.py
evaluator=$repo/scripts/evaluate_massive_medical_kalai_s1_completion_batch_v1.py
started=$SECONDS
completed=false

on_exit() {
  code=$?
  if [[ -d $control && $completed != true && ! -e $control/RESULT.json && ! -e $control/STOPPED ]]; then
    temporary=$control/STOPPED.tmp.$SLURM_JOB_ID
    printf 'stage=completion_batch\nbatch_id=%s\njob_id=%s\nexit_code=%s\nrestart_or_resume_authorized=false\nretry_or_replacement_authorized=false\nautomatic_next_batch_authorized=false\n' \
      "$batch_id" "$SLURM_JOB_ID" "$code" > "$temporary"
    chmod 0400 "$temporary"
    mv "$temporary" "$control/STOPPED"
  fi
}
trap on_exit EXIT

[[ -z ${OPENAI_API_KEY:-} ]] || {
  echo 'OPENAI_API_KEY must not reach a GPU job.' >&2
  exit 10
}
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY
unset TRANSFORMERS_CACHE

test -d "$control/SUBMISSION_LOCK"
grep -Fx "batch_id=$batch_id" "$control/SUBMITTED" >/dev/null
grep -Fx "job_id=$SLURM_JOB_ID" "$control/SUBMITTED" >/dev/null
grep -Fx 'held_first=true' "$control/SUBMITTED" >/dev/null
grep -Fx 'held_audit_passed=true' "$control/SUBMITTED" >/dev/null
grep -Fx "job_id=$SLURM_JOB_ID" "$control/RELEASE_AUTHORIZED" >/dev/null
grep -Fx 'release_authorized=true' "$control/RELEASE_AUTHORIZED" >/dev/null
mkdir "$control/INVOCATION_LOCK"
owner=$control/INVOCATION_LOCK/owner
owner_tmp=$owner.tmp.$SLURM_JOB_ID
printf 'stage=completion_batch\nbatch_id=%s\njob_id=%s\nrestart_or_resume_authorized=false\nretry_or_replacement_authorized=false\nautomatic_next_batch_authorized=false\nexternal_api_calls_authorized=0\n' \
  "$batch_id" "$SLURM_JOB_ID" > "$owner_tmp"
chmod 0400 "$owner_tmp"
mv "$owner_tmp" "$owner"

# Avoid racing the submitter's atomic post-release receipt.
for _release_wait in $(seq 1 60); do
  [[ -s $control/RELEASED ]] && break
  sleep 1
done
test -s "$control/RELEASED"
grep -Fx "batch_id=$batch_id" "$control/RELEASED" >/dev/null
grep -Fx "job_id=$SLURM_JOB_ID" "$control/RELEASED" >/dev/null
grep -Fx 'released=true' "$control/RELEASED" >/dev/null

test ! -e "$output/generation/completion_batches/$batch_id"
test ! -e "$control/RESULT.json"
test ! -e "$control/STOPPED"

module load gcc/13.4.0 cuda/12.9.1
module load conda/Miniforge3-25.3.1-3
conda activate "$root/envs/subliminal-mitigate-py311"
export PYTHONPATH=$repo
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-kalai-s1-$batch_id-pyc-$SLURM_JOB_ID
export DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export XDG_CACHE_HOME=$root/cache XDG_CONFIG_HOME=$root/config TMPDIR=$root/tmp
export HF_HOME=$root/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRITON_CACHE_DIR=$root/cache/triton
mkdir -p "$XDG_CONFIG_HOME" "$TMPDIR" "$HUGGINGFACE_HUB_CACHE" "$TRITON_CACHE_DIR"

cd "$repo"
python "$authorizer" verify \
  --output-root "$output" \
  --repo-root "$repo" \
  --batch-index "$batch_index" \
  --running-job-id "$SLURM_JOB_ID"
python "$controller" \
  --batch-plan "$batch_plan" \
  --output-root "$output" \
  --repo-root "$repo" \
  --batch-index "$batch_index" \
  --gpu-authorization "$control/AUTHORIZATION.json" \
  --slurm-job-id "$SLURM_JOB_ID" \
  --device cuda:0
python "$controller" \
  --batch-plan "$batch_plan" \
  --output-root "$output" \
  --batch-index "$batch_index" \
  --audit-only
python "$evaluator" \
  --output-root "$output" \
  --repo-root "$repo" \
  --batch-plan "$batch_plan" \
  --batch-index "$batch_index" \
  --authorization "$control/AUTHORIZATION.json" \
  --elapsed-seconds "$((SECONDS - started))" \
  --slurm-job-id "$SLURM_JOB_ID"
completed=true
trap - EXIT
echo "MASSIVE_MEDICAL_KALAI_S1_${batch_id^^}_COMPLETE_NO_AUTOMATIC_NEXT_BATCH"
