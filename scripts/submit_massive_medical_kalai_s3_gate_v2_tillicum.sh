#!/bin/bash
# Authorize, held-audit, and release exactly one Kalai s=3 coverage-gate job.

set -euo pipefail
umask 077
ulimit -c 0

usage() {
  echo 'Usage: submit_massive_medical_kalai_s3_gate_v2_tillicum.sh --ack-h200-minutes 20 --ack-max-cost-usd 0.300 --ack-program-ceiling-usd 5.9933725' >&2
  exit 2
}

[[ $# -eq 6 ]] || usage
[[ $1 == --ack-h200-minutes && $3 == --ack-max-cost-usd && $5 == --ack-program-ceiling-usd ]] || usage
minutes=$2
cost=$4
ceiling=$6

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-kalai-s3-r20-v2-submit-recovery-v3
output=$root/outputs/massive_medical_kalai_s3_r20_v2_submit_recovery_v3
control=$output/control
logs=$root/outputs/logs
authorizer=$repo/scripts/authorize_massive_medical_kalai_s3_v2.py
sbatch_file=$repo/scripts/sbatch_massive_medical_kalai_s3_gate_v2_tillicum_h200.sbatch
expected_name=mmu_kalai_s3_gate
expected_time=00:20:00
log_glob=$logs/massive_medical_kalai_s3_r20_v2_submit_recovery_v3_gate_\*

cd "$repo"
test -z "$(git status --porcelain)"
test -s "$control/CPU_STAGE.json"
test -s "$control/GATE_PLAN.json"
test ! -e "$control/GATE_SUBMISSION_LOCK"
test ! -e "$control/GATE_SUBMISSION_ATTEMPT.tsv"
test ! -e "$control/GATE_SUBMITTED"
test ! -e "$control/GATE_RELEASE_AUTHORIZED"
test ! -e "$control/GATE_RELEASED"
test ! -e "$control/GATE_AUTHORIZATION.json"
test ! -e "$control/GATE_INVOCATION_LOCK"
test ! -e "$control/GATE_RESULT.json"
test ! -e "$control/GATE_STOPPED"
test ! -e "$control/COMPLETION_AUTHORIZATION.json"
test ! -e "$output/generation"
if compgen -G "$log_glob" >/dev/null; then
  echo 'Kalai s=3 gate log namespace is not fresh.' >&2
  exit 4
fi

module load conda/Miniforge3-25.3.1-3
conda activate "$root/envs/subliminal-mitigate-py311"
command -v python >/dev/null
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-kalai-s3-submit-recovery-v3-pyc

mkdir "$control/GATE_SUBMISSION_LOCK"
owner_tmp=$control/GATE_SUBMISSION_LOCK/owner.tmp.$$
printf 'protocol_id=massive_medical_kalai_s3_r20_v2\nstage=gate\nrepository_commit=%s\nrestart_or_resume_authorized=false\nautomatic_continuation_authorized=false\n' \
  "$(git rev-parse HEAD)" > "$owner_tmp"
chmod 0400 "$owner_tmp"
mv "$owner_tmp" "$control/GATE_SUBMISSION_LOCK/owner"

python "$authorizer" write \
  --stage gate \
  --output-root "$output" \
  --repo-root "$repo" \
  --ack-h200-minutes "$minutes" \
  --ack-max-cost-usd "$cost" \
  --ack-program-ceiling-usd "$ceiling"
python "$authorizer" verify \
  --stage gate \
  --output-root "$output" \
  --repo-root "$repo"

raw_job=$(sbatch --parsable --hold --export=NONE --no-requeue \
  --account=stf --partition=gpu-h200 --qos=normal \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G \
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

attempt=$control/GATE_SUBMISSION_ATTEMPT.tsv
attempt_tmp=$attempt.tmp.$$
printf 'stage\tjob_id\th200_minutes\tmaximum_cost_usd\n%s\t%s\t%s\t%s\n' \
  gate "$job_id" "$minutes" "$cost" > "$attempt_tmp"
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
node_range=$(awk -F= '$1=="NumNodes" {print $2; exit}' <<< "$job_record")
[[ $node_range == 1 || $node_range == 1-1 ]]
test "$(awk -F= '$1=="NumTasks" {print $2; exit}' <<< "$job_record")" = 1
test "$(awk -F= '$1=="NumCPUs" {print $2; exit}' <<< "$job_record")" = 8
test "$(awk -F= '$1=="Command" {print $2; exit}' <<< "$job_record")" = "$sbatch_file"
test "$(awk -F= '$1=="WorkDir" {print $2; exit}' <<< "$job_record")" = "$repo"
requested_tres=$(sed -n 's/^ReqTRES=//p' <<< "$job_record")
test "$(tr ',' '\n' <<< "$requested_tres" | awk -F= '$1=="cpu" {print $2}')" = 8
test "$(tr ',' '\n' <<< "$requested_tres" | awk -F= '$1=="mem" {print $2}')" = 200G
test "$(tr ',' '\n' <<< "$requested_tres" | awk -F= '$1=="node" {print $2}')" = 1
test "$(tr ',' '\n' <<< "$requested_tres" | awk -F= '$1=="gres/gpu" {print $2}')" = 1
test "$(tr ',' '\n' <<< "$requested_tres" | awk -F= '$1=="gres/gpu:h200" {print $2}')" = 1

record_tmp=$control/GATE_SUBMITTED.tmp.$$
printf 'protocol_id=massive_medical_kalai_s3_r20_v2\nstage=gate\njob_id=%s\nheld_first=true\nheld_audit_passed=true\nrepository_commit=%s\nrestart_or_resume_authorized=false\nautomatic_continuation_authorized=false\n' \
  "$job_id" "$(git rev-parse HEAD)" > "$record_tmp"
chmod 0400 "$record_tmp"
mv "$record_tmp" "$control/GATE_SUBMITTED"

release_tmp=$control/GATE_RELEASE_AUTHORIZED.tmp.$$
printf 'protocol_id=massive_medical_kalai_s3_r20_v2\nstage=gate\njob_id=%s\nheld_audit_passed=true\nrelease_authorized=true\nrestart_or_resume_authorized=false\nautomatic_continuation_authorized=false\n' \
  "$job_id" > "$release_tmp"
chmod 0400 "$release_tmp"
mv "$release_tmp" "$control/GATE_RELEASE_AUTHORIZED"
scontrol release "$job_id"
released=true
released_tmp=$control/GATE_RELEASED.tmp.$$
printf 'protocol_id=massive_medical_kalai_s3_r20_v2\nstage=gate\njob_id=%s\nreleased=true\nrestart_or_resume_authorized=false\nautomatic_continuation_authorized=false\n' \
  "$job_id" > "$released_tmp"
chmod 0400 "$released_tmp"
mv "$released_tmp" "$control/GATE_RELEASED"
trap - EXIT
echo "Submitted and released the one-shot Kalai s=3 gate as job $job_id."
echo 'No completion, retry, resume, replacement, requeue, or API call was authorized.'
