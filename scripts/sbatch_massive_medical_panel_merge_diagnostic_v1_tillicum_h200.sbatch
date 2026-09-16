#!/bin/bash
#SBATCH --job-name=mmu_panel_merge_v1
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/massive_medical_panel_merge_diagnostic_v1_%j.out
#SBATCH --error=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/massive_medical_panel_merge_diagnostic_v1_%j.err

set -euo pipefail
umask 077
ulimit -c 0

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-panel-merge-diagnostic-v1
output=$root/outputs/massive_medical_panel_merge_diagnostic_v1
plan=$output/control/PLAN.json
control=$output/control
authorization=$control/AUTHORIZATION
submitted=$control/SUBMITTED
result=$control/GPU_RESULT
stopped=$control/GPU_STOPPED
started=$SECONDS
completed=false

on_exit() {
  code=$?
  if [[ $completed != true && -d $control && ! -e $result && ! -e $stopped ]]; then
    stopped_commit=$(git -C "$repo" rev-parse HEAD 2>/dev/null || printf unavailable)
    temporary=$control/.GPU_STOPPED.$SLURM_JOB_ID
    printf 'protocol_id=massive_medical_panel_merge_diagnostic_v1\njob_id=%s\nexit_code=%s\nelapsed_seconds=%s\nrepository_commit=%s\nrestart_or_resume_authorized=false\n' \
      "$SLURM_JOB_ID" "$code" "$((SECONDS - started))" \
      "$stopped_commit" > "$temporary"
    chmod 0400 "$temporary"
    mv "$temporary" "$stopped"
  fi
}
trap on_exit EXIT

[[ -z ${OPENAI_API_KEY:-} ]] || {
  echo 'OPENAI_API_KEY must not reach this zero-API GPU diagnostic.' >&2
  exit 10
}
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY
unset TRANSFORMERS_CACHE

test -f "$plan"
test ! -L "$plan"
test -s "$authorization"
test ! -L "$authorization"
test -s "$submitted"
test ! -L "$submitted"
test ! -e "$control/EXECUTION_STARTED.json"
test ! -e "$control/EXECUTION_COMPLETE.json"
test ! -e "$result"
test ! -e "$stopped"
test -z "$(git -C "$repo" status --porcelain)"

field() {
  awk -F= -v key="$2" '$1 == key {print substr($0, index($0, "=") + 1); exit}' "$1"
}
commit=$(git -C "$repo" rev-parse HEAD)
plan_sha=$(sha256sum "$plan" | awk '{print $1}')
authorization_sha=$(sha256sum "$authorization" | awk '{print $1}')
test "$(field "$authorization" protocol_id)" = massive_medical_panel_merge_diagnostic_v1
test "$(field "$authorization" h200_minutes)" = 45
test "$(field "$authorization" maximum_gpu_cost_usd)" = 0.675
test "$(field "$authorization" external_api_calls)" = 0
test "$(field "$authorization" repository_commit)" = "$commit"
test "$(field "$authorization" plan_file_sha256)" = "$plan_sha"
test "$(field "$authorization" no_requeue)" = true
test "$(field "$authorization" no_resume_or_replace)" = true
test "$(field "$submitted" protocol_id)" = massive_medical_panel_merge_diagnostic_v1
test "$(field "$submitted" job_id)" = "$SLURM_JOB_ID"
test "$(field "$submitted" held_first)" = true
test "$(field "$submitted" held_audit_passed)" = true
test "$(field "$submitted" repository_commit)" = "$commit"
test "$(field "$submitted" plan_file_sha256)" = "$plan_sha"
test "$(field "$submitted" authorization_file_sha256)" = "$authorization_sha"

job_record=$(scontrol show job "$SLURM_JOB_ID" -o | tr ' ' '\n')
test "$(awk -F= '$1=="Requeue" {print $2; exit}' <<< "$job_record")" = 0
test "$(awk -F= '$1=="Account" {print $2; exit}' <<< "$job_record")" = stf
test "$(awk -F= '$1=="Partition" {print $2; exit}' <<< "$job_record")" = gpu-h200
test "$(awk -F= '$1=="QOS" {print $2; exit}' <<< "$job_record")" = normal
test "$(awk -F= '$1=="TimeLimit" {print $2; exit}' <<< "$job_record")" = 00:45:00
requested_tres=$(sed -n 's/^ReqTRES=//p' <<< "$job_record")
test "$(tr ',' '\n' <<< "$requested_tres" | awk -F= '$1=="gres/gpu:h200" {print $2}')" = 1

module load gcc/13.4.0 cuda/12.9.1
module load conda/Miniforge3-25.3.1-3
conda activate "$root/envs/subliminal-mitigate-py311"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-panel-merge-diagnostic-pyc-$SLURM_JOB_ID
export DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export XDG_CACHE_HOME=$root/cache XDG_CONFIG_HOME=$root/config TMPDIR=$root/tmp
export HF_HOME=$root/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRITON_CACHE_DIR=$root/cache/triton
mkdir -p "$XDG_CONFIG_HOME" "$TMPDIR" "$HUGGINGFACE_HUB_CACHE" "$TRITON_CACHE_DIR"

cd "$repo"
date
hostname
nvidia-smi
python scripts/run_massive_medical_panel_merge_diagnostic_v1.py \
  --preflight-only --plan "$plan"
python scripts/run_massive_medical_panel_merge_diagnostic_v1.py \
  --execute --plan "$plan" --device cuda:0
python scripts/run_massive_medical_panel_merge_diagnostic_v1.py \
  --audit-only --plan "$plan"
result_tmp=$control/.GPU_RESULT.$SLURM_JOB_ID
printf 'protocol_id=massive_medical_panel_merge_diagnostic_v1\nstatus=COMPLETE\njob_id=%s\nelapsed_seconds=%s\nrepository_commit=%s\ncompletion_record_sha256=%s\nexternal_api_calls=0\n' \
  "$SLURM_JOB_ID" "$((SECONDS - started))" "$commit" \
  "$(sha256sum "$control/EXECUTION_COMPLETE.json" | awk '{print $1}')" > "$result_tmp"
chmod 0400 "$result_tmp"
mv "$result_tmp" "$result"
completed=true
trap - EXIT
date
