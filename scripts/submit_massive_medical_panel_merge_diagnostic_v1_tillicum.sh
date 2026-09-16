#!/bin/bash
# Submit the separately staged, zero-API panel merge diagnostic once.

set -euo pipefail
umask 077
ulimit -c 0

usage() {
  echo 'Usage: submit_massive_medical_panel_merge_diagnostic_v1_tillicum.sh --ack-h200-minutes 45 --ack-max-cost-usd 0.675' >&2
  exit 2
}

[[ $# -eq 4 ]] || usage
[[ $1 == --ack-h200-minutes && $3 == --ack-max-cost-usd ]] || usage
[[ $2 == 45 && $4 == 0.675 ]] || usage

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-panel-merge-diagnostic-v1
output=$root/outputs/massive_medical_panel_merge_diagnostic_v1
control=$output/control
plan=$control/PLAN.json
sbatch_file=$repo/scripts/sbatch_massive_medical_panel_merge_diagnostic_v1_tillicum_h200.sbatch
logs=$root/outputs/logs

cd "$repo"
test -z "$(git status --porcelain)"
test -s "$plan"
test ! -e "$control/EXECUTION_STARTED.json"
test ! -e "$control/EXECUTION_COMPLETE.json"
test ! -e "$control/GPU_RESULT"
test ! -e "$control/GPU_STOPPED"
test ! -e "$control/SUBMISSION_LOCK"
test ! -e "$control/AUTHORIZATION"
test ! -e "$control/SUBMITTED"
if compgen -G "$logs/massive_medical_panel_merge_diagnostic_v1_*" >/dev/null; then
  echo 'Diagnostic log namespace is not fresh.' >&2
  exit 4
fi

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY
unset CUDA_VISIBLE_DEVICES TRANSFORMERS_CACHE
module load conda/Miniforge3-25.3.1-3
conda activate "$root/envs/subliminal-mitigate-py311"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-panel-merge-diagnostic-submit-pyc
export DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export XDG_CACHE_HOME=$root/cache XDG_CONFIG_HOME=$root/config TMPDIR=$root/tmp
export HF_HOME=$root/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

python scripts/run_massive_medical_panel_merge_diagnostic_v1.py \
  --preflight-only --plan "$plan"
sbatch --test-only --export=NONE --no-requeue "$sbatch_file"

mkdir "$control/SUBMISSION_LOCK"
commit=$(git rev-parse HEAD)
plan_sha=$(sha256sum "$plan" | awk '{print $1}')
authorization_tmp=$control/.AUTHORIZATION.$$
printf 'protocol_id=massive_medical_panel_merge_diagnostic_v1\nh200_minutes=45\nmaximum_gpu_cost_usd=0.675\nexternal_api_calls=0\nrepository_commit=%s\nplan_file_sha256=%s\nauthorized_at=%s\nno_requeue=true\nno_resume_or_replace=true\n' \
  "$commit" "$plan_sha" "$(date --iso-8601=seconds)" > "$authorization_tmp"
chmod 0400 "$authorization_tmp"
mv "$authorization_tmp" "$control/AUTHORIZATION"

job_id=
released=false
cancel_unreleased_job_on_exit() {
  code=$?
  if [[ $released != true && $job_id =~ ^[0-9]+$ ]]; then
    scancel "$job_id" || true
  fi
  trap - EXIT
  exit "$code"
}
trap cancel_unreleased_job_on_exit EXIT

raw_job=$(sbatch --parsable --hold --export=NONE --no-requeue "$sbatch_file")
job_id=${raw_job%%;*}
[[ $job_id =~ ^[0-9]+$ ]]

job_record=$(scontrol show job "$job_id" -o | tr ' ' '\n')
test "$(awk -F= '$1=="JobState" {print $2; exit}' <<< "$job_record")" = PENDING
test "$(awk -F= '$1=="Reason" {print $2; exit}' <<< "$job_record")" = JobHeldUser
test "$(awk -F= '$1=="Requeue" {print $2; exit}' <<< "$job_record")" = 0
test "$(awk -F= '$1=="Account" {print $2; exit}' <<< "$job_record")" = stf
test "$(awk -F= '$1=="Partition" {print $2; exit}' <<< "$job_record")" = gpu-h200
test "$(awk -F= '$1=="QOS" {print $2; exit}' <<< "$job_record")" = normal
test "$(awk -F= '$1=="TimeLimit" {print $2; exit}' <<< "$job_record")" = 00:45:00
test "$(awk -F= '$1=="Command" {print $2; exit}' <<< "$job_record")" = "$sbatch_file"
requested_tres=$(sed -n 's/^ReqTRES=//p' <<< "$job_record")
test "$(tr ',' '\n' <<< "$requested_tres" | awk -F= '$1=="gres/gpu:h200" {print $2}')" = 1

authorization_sha=$(sha256sum "$control/AUTHORIZATION" | awk '{print $1}')
submitted_tmp=$control/.SUBMITTED.$$
printf 'protocol_id=massive_medical_panel_merge_diagnostic_v1\njob_id=%s\nheld_first=true\nheld_audit_passed=true\nrepository_commit=%s\nplan_file_sha256=%s\nauthorization_file_sha256=%s\n' \
  "$job_id" "$commit" "$plan_sha" "$authorization_sha" > "$submitted_tmp"
chmod 0400 "$submitted_tmp"
mv "$submitted_tmp" "$control/SUBMITTED"
scontrol release "$job_id"
released=true
trap - EXIT
echo "Submitted and released zero-API panel merge diagnostic as job $job_id."
