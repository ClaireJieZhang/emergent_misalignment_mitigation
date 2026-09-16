#!/bin/bash
# Explicitly authorize, held-audit, and release one k2s1 smoke job.

set -euo pipefail
umask 077
ulimit -c 0

usage() {
  echo 'Usage: submit_massive_medical_kalai_k2_s1_smoke_v1_tillicum.sh --ack-h200-minutes 30 --ack-max-cost-usd 0.45 --ack-no-api --ack-no-full-run' >&2
  exit 2
}

[[ $# -eq 6 ]] || usage
[[ $1 == --ack-h200-minutes && $3 == --ack-max-cost-usd && $5 == --ack-no-api && $6 == --ack-no-full-run ]] || usage
minutes=$2
cost=$4
[[ $minutes == 30 && $cost == 0.45 ]] || usage

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-panel-diagnostics-v1
output=$root/outputs/massive_medical_kalai_k2_s1_r20_v1
control=$output/control
logs=$root/outputs/logs
sbatch_file=$repo/scripts/sbatch_massive_medical_kalai_k2_s1_smoke_v1_tillicum_h200.sbatch
expected_name=mmu_k2s1_smoke
expected_time=00:30:00
log_glob=$logs/massive_medical_kalai_k2_s1_smoke_v1_\*

cd "$repo"
test -z "$(git status --porcelain)"
test -s "$control/CPU_STAGE"
grep -Fx 'protocol_id=massive_medical_kalai_k2_s1_r20_v1' "$control/CPU_STAGE" >/dev/null
grep -Fx 'status=CPU_STAGED_NO_GPU_OR_API_AUTHORITY' "$control/CPU_STAGE" >/dev/null
grep -Fx "repository_commit=$(git rev-parse HEAD)" "$control/CPU_STAGE" >/dev/null
grep -Fx 'smoke_authorized=false' "$control/CPU_STAGE" >/dev/null
grep -Fx 'full_run_authorized=false' "$control/CPU_STAGE" >/dev/null
test ! -e "$control/SMOKE_AUTHORIZATION"
test ! -e "$control/SMOKE_SUBMISSION_LOCK"
test ! -e "$control/SMOKE_SUBMISSION_ATTEMPT.tsv"
test ! -e "$control/SMOKE_SUBMITTED"
test ! -e "$control/SMOKE_RELEASE_AUTHORIZED"
test ! -e "$control/SMOKE_RELEASED"
test ! -e "$control/SMOKE_INVOCATION_LOCK"
test ! -e "$control/SMOKE_COMPLETE"
test ! -e "$control/SMOKE_STOPPED"
test ! -e "$output/generation"
if compgen -G "$log_glob" >/dev/null; then
  echo 'Kalai k2s1 smoke log namespace is not fresh.' >&2
  exit 4
fi

mkdir "$control/SMOKE_SUBMISSION_LOCK"
owner_tmp=$control/SMOKE_SUBMISSION_LOCK/owner.tmp.$$
printf 'protocol_id=massive_medical_kalai_k2_s1_r20_v1\nstage=smoke\nrepository_commit=%s\nheld_first=true\nrestart_resume_retry_authorized=false\nfull_run_authorized=false\n' \
  "$(git rev-parse HEAD)" > "$owner_tmp"
chmod 0400 "$owner_tmp"
mv "$owner_tmp" "$control/SMOKE_SUBMISSION_LOCK/owner"

auth_tmp=$control/SMOKE_AUTHORIZATION.tmp.$$
printf 'protocol_id=massive_medical_kalai_k2_s1_r20_v1\nstage=smoke\nrepository_commit=%s\nmaximum_h200_minutes=30\nmaximum_cost_usd=0.45\nh200_usd_per_hour=0.90\nmaximum_gpu_jobs=1\nmaximum_candidate_attempts=360\nexternal_api_calls_authorized=0\nfull_run_authorized=false\nrestart_resume_retry_replacement_requeue_authorized=false\n' \
  "$(git rev-parse HEAD)" > "$auth_tmp"
chmod 0400 "$auth_tmp"
mv "$auth_tmp" "$control/SMOKE_AUTHORIZATION"

raw_job=$(sbatch --parsable --hold --export=NONE --no-requeue \
  --account=stf --partition=gpu-h200 --qos=normal \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=120G \
  --gres=gpu:h200:1 --time="$expected_time" --job-name="$expected_name" \
  "$sbatch_file")
job_id=${raw_job%%;*}
[[ $job_id =~ ^[0-9]+$ ]]
released=false
cancel_pristine_held_on_exit() {
  code=$?
  if [[ $released != true ]]; then
    state_reason=$(squeue -h -j "$job_id" -o '%T|%r' 2>/dev/null || true)
    if [[ $state_reason == 'PENDING|JobHeldUser' ]]; then
      scancel "$job_id" || true
    fi
  fi
  trap - EXIT
  exit "$code"
}
trap cancel_pristine_held_on_exit EXIT

attempt=$control/SMOKE_SUBMISSION_ATTEMPT.tsv
attempt_tmp=$attempt.tmp.$$
printf 'stage\tjob_id\th200_minutes\tmaximum_cost_usd\nsmoke\t%s\t30\t0.45\n' \
  "$job_id" > "$attempt_tmp"
chmod 0400 "$attempt_tmp"
mv "$attempt_tmp" "$attempt"

job_record=$(scontrol show job "$job_id" -o | tr ' ' '\n')
test "$(awk -F= '$1=="JobState" {print $2; exit}' <<< "$job_record")" = PENDING
test "$(awk -F= '$1=="Reason" {print $2; exit}' <<< "$job_record")" = JobHeldUser
test "$(awk -F= '$1=="Requeue" {print $2; exit}' <<< "$job_record")" = 0
test "$(awk -F= '$1=="Account" {print $2; exit}' <<< "$job_record")" = stf
test "$(awk -F= '$1=="Partition" {print $2; exit}' <<< "$job_record")" = gpu-h200
test "$(awk -F= '$1=="QOS" {print $2; exit}' <<< "$job_record")" = normal
test "$(awk -F= '$1=="JobName" {print $2; exit}' <<< "$job_record")" = "$expected_name"
test "$(awk -F= '$1=="TimeLimit" {print $2; exit}' <<< "$job_record")" = "$expected_time"
test "$(awk -F= '$1=="NumTasks" {print $2; exit}' <<< "$job_record")" = 1
test "$(awk -F= '$1=="NumCPUs" {print $2; exit}' <<< "$job_record")" = 8
test "$(awk -F= '$1=="Command" {print $2; exit}' <<< "$job_record")" = "$sbatch_file"
test "$(awk -F= '$1=="WorkDir" {print $2; exit}' <<< "$job_record")" = "$repo"
requested_tres=$(sed -n 's/^ReqTRES=//p' <<< "$job_record")
test "$(tr ',' '\n' <<< "$requested_tres" | awk -F= '$1=="cpu" {print $2}')" = 8
test "$(tr ',' '\n' <<< "$requested_tres" | awk -F= '$1=="mem" {print $2}')" = 120G
test "$(tr ',' '\n' <<< "$requested_tres" | awk -F= '$1=="gres/gpu:h200" {print $2}')" = 1

submitted_tmp=$control/SMOKE_SUBMITTED.tmp.$$
printf 'protocol_id=massive_medical_kalai_k2_s1_r20_v1\nstage=smoke\njob_id=%s\nheld_first=true\nheld_audit_passed=true\nrepository_commit=%s\nfull_run_authorized=false\nrestart_resume_retry_authorized=false\n' \
  "$job_id" "$(git rev-parse HEAD)" > "$submitted_tmp"
chmod 0400 "$submitted_tmp"
mv "$submitted_tmp" "$control/SMOKE_SUBMITTED"

release_tmp=$control/SMOKE_RELEASE_AUTHORIZED.tmp.$$
printf 'protocol_id=massive_medical_kalai_k2_s1_r20_v1\nstage=smoke\njob_id=%s\nheld_audit_passed=true\nrelease_authorized=true\nfull_run_authorized=false\nrestart_resume_retry_authorized=false\n' \
  "$job_id" > "$release_tmp"
chmod 0400 "$release_tmp"
mv "$release_tmp" "$control/SMOKE_RELEASE_AUTHORIZED"
scontrol release "$job_id"
released=true
released_tmp=$control/SMOKE_RELEASED.tmp.$$
printf 'protocol_id=massive_medical_kalai_k2_s1_r20_v1\nstage=smoke\njob_id=%s\nreleased=true\nfull_run_authorized=false\nrestart_resume_retry_authorized=false\n' \
  "$job_id" > "$released_tmp"
chmod 0400 "$released_tmp"
mv "$released_tmp" "$control/SMOKE_RELEASED"
trap - EXIT
echo "Submitted and released the one-shot k2s1 smoke as job $job_id."
echo 'No full run, retry, resume, replacement, requeue, judging, or API call was authorized.'
