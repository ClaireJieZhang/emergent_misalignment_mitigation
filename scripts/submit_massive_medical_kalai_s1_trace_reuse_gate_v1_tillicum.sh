#!/bin/bash
# Authorize, held-audit, and release exactly one Kalai s=1 technical-gate job.

set -euo pipefail
umask 077
ulimit -c 0

usage() {
  echo 'Usage: submit_massive_medical_kalai_s1_trace_reuse_gate_v1_tillicum.sh --ack-h200-minutes 35 --ack-max-cost-usd 0.525 --ack-current-conservative-exposure-usd 5.79498425 --ack-conservative-program-max-usd 6.31998425 --ack-program-ceiling-usd 6.5000000 --ack-no-api --ack-no-completion --ack-no-restart-resume-retry --ack-post-hoc-sensitivity' >&2
  exit 2
}

[[ $# -eq 14 ]] || usage
[[ $1 == --ack-h200-minutes && $3 == --ack-max-cost-usd && $5 == --ack-current-conservative-exposure-usd && $7 == --ack-conservative-program-max-usd && $9 == --ack-program-ceiling-usd ]] || usage
minutes=$2
cost=$4
current=$6
maximum=$8
ceiling=${10}
shift 10
[[ $1 == --ack-no-api && $2 == --ack-no-completion && $3 == --ack-no-restart-resume-retry && $4 == --ack-post-hoc-sensitivity ]] || usage

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-kalai-s1-r20-trace-reuse-v1
output=$root/outputs/massive_medical_kalai_s1_r20_trace_reuse_v1
control=$output/control
logs=$root/outputs/logs
authorizer=$repo/scripts/authorize_massive_medical_kalai_s1_trace_reuse_v1.py
sbatch_file=$repo/scripts/sbatch_massive_medical_kalai_s1_trace_reuse_gate_v1_tillicum_h200.sbatch
expected_name=mmu_kalai_s1_gate
expected_time=00:35:00
log_glob=$logs/massive_medical_kalai_s1_r20_trace_reuse_v1_gate_\*

cd "$repo"
test -z "$(git status --porcelain)"
test -s "$control/CPU_STAGE.json"
test -s "$control/REPLAY_PLAN.json"
test ! -e "$control/TECHNICAL_GATE_AUTHORIZATION.json"
test ! -e "$control/TECHNICAL_GATE_SUBMISSION_LOCK"
test ! -e "$control/TECHNICAL_GATE_SUBMISSION_ATTEMPT.tsv"
test ! -e "$control/TECHNICAL_GATE_SUBMITTED"
test ! -e "$control/TECHNICAL_GATE_RELEASE_AUTHORIZED"
test ! -e "$control/TECHNICAL_GATE_RELEASED"
test ! -e "$control/TECHNICAL_GATE_INVOCATION_LOCK"
test ! -e "$control/TECHNICAL_GATE_RESULT.json"
test ! -e "$control/TECHNICAL_GATE_STOPPED"
test ! -e "$control/COMPLETION_AUTHORIZATION.json"
test ! -e "$output/generation"
test ! -e "$output/assembled"
if compgen -G "$log_glob" >/dev/null; then
  echo 'Kalai s=1 technical-gate log namespace is not fresh.' >&2
  exit 4
fi

module load conda/Miniforge3-25.3.1-3
conda activate "$root/envs/subliminal-mitigate-py311"
export PYTHONPATH=$repo
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-kalai-s1-gate-submit-pyc

mkdir "$control/TECHNICAL_GATE_SUBMISSION_LOCK"
owner_tmp=$control/TECHNICAL_GATE_SUBMISSION_LOCK/owner.tmp.$$
printf 'protocol_id=massive_medical_kalai_s1_r20_trace_reuse_v1\nstage=technical_gate\nrepository_commit=%s\nrestart_or_resume_authorized=false\nautomatic_continuation_authorized=false\n' \
  "$(git rev-parse HEAD)" > "$owner_tmp"
chmod 0400 "$owner_tmp"
mv "$owner_tmp" "$control/TECHNICAL_GATE_SUBMISSION_LOCK/owner"

python "$authorizer" write \
  --output-root "$output" \
  --repo-root "$repo" \
  --ack-h200-minutes "$minutes" \
  --ack-max-cost-usd "$cost" \
  --ack-current-conservative-exposure-usd "$current" \
  --ack-conservative-program-max-usd "$maximum" \
  --ack-program-ceiling-usd "$ceiling" \
  --ack-no-api \
  --ack-no-completion \
  --ack-no-restart-resume-retry \
  --ack-post-hoc-sensitivity
python "$authorizer" verify \
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

attempt=$control/TECHNICAL_GATE_SUBMISSION_ATTEMPT.tsv
attempt_tmp=$attempt.tmp.$$
printf 'stage\tjob_id\th200_minutes\tmaximum_cost_usd\n%s\t%s\t%s\t%s\n' \
  technical_gate "$job_id" "$minutes" "$cost" > "$attempt_tmp"
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

submitted_tmp=$control/TECHNICAL_GATE_SUBMITTED.tmp.$$
printf 'protocol_id=massive_medical_kalai_s1_r20_trace_reuse_v1\nstage=technical_gate\njob_id=%s\nheld_first=true\nheld_audit_passed=true\nrepository_commit=%s\nrestart_or_resume_authorized=false\nautomatic_continuation_authorized=false\n' \
  "$job_id" "$(git rev-parse HEAD)" > "$submitted_tmp"
chmod 0400 "$submitted_tmp"
mv "$submitted_tmp" "$control/TECHNICAL_GATE_SUBMITTED"

release_tmp=$control/TECHNICAL_GATE_RELEASE_AUTHORIZED.tmp.$$
printf 'protocol_id=massive_medical_kalai_s1_r20_trace_reuse_v1\nstage=technical_gate\njob_id=%s\nheld_audit_passed=true\nrelease_authorized=true\nrestart_or_resume_authorized=false\nautomatic_continuation_authorized=false\n' \
  "$job_id" > "$release_tmp"
chmod 0400 "$release_tmp"
mv "$release_tmp" "$control/TECHNICAL_GATE_RELEASE_AUTHORIZED"
scontrol release "$job_id"
released=true
released_tmp=$control/TECHNICAL_GATE_RELEASED.tmp.$$
printf 'protocol_id=massive_medical_kalai_s1_r20_trace_reuse_v1\nstage=technical_gate\njob_id=%s\nreleased=true\nrestart_or_resume_authorized=false\nautomatic_continuation_authorized=false\n' \
  "$job_id" > "$released_tmp"
chmod 0400 "$released_tmp"
mv "$released_tmp" "$control/TECHNICAL_GATE_RELEASED"
trap - EXIT
echo "Submitted and released the one-shot Kalai s=1 technical gate as job $job_id."
echo 'No completion, retry, resume, replacement, requeue, or API call was authorized.'
