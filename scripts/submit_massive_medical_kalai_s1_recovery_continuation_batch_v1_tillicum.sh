#!/bin/bash
# Authorize, held-audit, and release one recovery-continuation batch.

set -euo pipefail
umask 077
ulimit -c 0

usage() {
  echo 'Usage: submit_massive_medical_kalai_s1_recovery_continuation_batch_v1_tillicum.sh --batch-index N --ack-h200-minutes 60 --ack-max-cost-usd 0.900 --ack-current-conservative-exposure-usd X --ack-conservative-program-max-usd Y --ack-program-ceiling-usd 12.5000000 --ack-no-api --ack-no-automatic-next-batch --ack-no-restart-resume-retry-replacement --ack-prior-caps-retained --ack-recovered-batch1-predecessor --ack-post-hoc-sensitivity' >&2
  exit 2
}

[[ $# -eq 18 ]] || usage
[[ $1 == --batch-index && $3 == --ack-h200-minutes && $5 == --ack-max-cost-usd && $7 == --ack-current-conservative-exposure-usd && $9 == --ack-conservative-program-max-usd && ${11} == --ack-program-ceiling-usd ]] || usage
batch_index=$2
minutes=$4
cost=$6
current=$8
maximum=${10}
ceiling=${12}
shift 12
[[ $1 == --ack-no-api && $2 == --ack-no-automatic-next-batch && $3 == --ack-no-restart-resume-retry-replacement && $4 == --ack-prior-caps-retained && $5 == --ack-recovered-batch1-predecessor && $6 == --ack-post-hoc-sensitivity ]] || usage
[[ $batch_index =~ ^[2-7]$ ]] || usage
[[ $minutes == 60 && $cost == 0.900 && $ceiling == 12.5000000 ]] || usage
printf -v batch_id 'batch_%02d' "$batch_index"

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-kalai-s1-recovery-continuation-v1
output=$root/outputs/massive_medical_kalai_s1_recovery_continuation_v1
source_output=$root/outputs/massive_medical_kalai_s1_completion_batches_v1
control=$output/control/batches/$batch_id
logs=$root/outputs/logs
manager=$repo/scripts/manage_massive_medical_kalai_s1_recovery_continuation_v1.py
authorizer=$repo/scripts/authorize_massive_medical_kalai_s1_recovery_continuation_batch_v1.py
sbatch_file=$repo/scripts/sbatch_massive_medical_kalai_s1_recovery_continuation_batch_v1_tillicum_h200.sbatch
expected_name=mmu_kalai_s1_rc$(printf '%02d' "$batch_index")

cd "$repo"
test -z "$(git status --porcelain)"
test -s "$output/control/CPU_STAGE.json"
test -s "$output/control/RECOVERY_CONTINUATION_PLAN.json"
test -s "$source_output/control/batches/batch_01/STOPPED"
test ! -e "$source_output/control/batches/batch_01/RESULT.json"
test ! -e "$source_output/control/batches/batch_02"
test ! -e "$control"
test ! -e "$output/generation/completion_batches/$batch_id"
if (( batch_index < 7 )); then
  for future_index in $(seq "$((batch_index + 1))" 7); do
    printf -v future_id 'batch_%02d' "$future_index"
    test ! -e "$output/control/batches/$future_id"
    test ! -e "$output/generation/completion_batches/$future_id"
  done
fi

module load conda/Miniforge3-25.3.1-3
conda activate "$root/envs/subliminal-mitigate-py311"
export PYTHONPATH=$repo
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-kalai-s1-rc-$batch_id-submit-pyc
unset OPENAI_API_KEY

python "$manager" preflight --output-root "$output" --repo-root "$repo" --batch-index "$batch_index"
python "$authorizer" preflight --output-root "$output" --repo-root "$repo" \
  --batch-index "$batch_index"
mkdir -p "$output/control/batches"
mkdir "$control"
mkdir "$control/SUBMISSION_LOCK"
owner_tmp=$control/SUBMISSION_LOCK/owner.tmp.$$
printf 'protocol_id=massive_medical_kalai_s1_recovery_continuation_v1\nstage=recovery_continuation_batch\nbatch_id=%s\nrepository_commit=%s\nrestart_or_resume_authorized=false\nretry_or_replacement_authorized=false\nautomatic_next_batch_authorized=false\n' \
  "$batch_id" "$(git rev-parse HEAD)" > "$owner_tmp"
chmod 0400 "$owner_tmp"
mv "$owner_tmp" "$control/SUBMISSION_LOCK/owner"

python "$authorizer" write \
  --output-root "$output" --repo-root "$repo" --batch-index "$batch_index" \
  --ack-h200-minutes "$minutes" --ack-max-cost-usd "$cost" \
  --ack-current-conservative-exposure-usd "$current" \
  --ack-conservative-program-max-usd "$maximum" \
  --ack-program-ceiling-usd "$ceiling" \
  --ack-no-api --ack-no-automatic-next-batch \
  --ack-no-restart-resume-retry-replacement --ack-prior-caps-retained \
  --ack-recovered-batch1-predecessor --ack-post-hoc-sensitivity
python "$authorizer" verify --output-root "$output" --repo-root "$repo" --batch-index "$batch_index"

raw_job=$(sbatch --parsable --hold --export=NONE --no-requeue \
  --account=stf --partition=gpu-h200 --qos=normal \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G \
  --gres=gpu:h200:1 --time=01:00:00 --job-name="$expected_name" \
  "$sbatch_file" "$batch_index")
job_id=${raw_job%%;*}
[[ $job_id =~ ^[0-9]+$ ]]
released=false
cancel_pristine_held_on_exit() {
  code=$?
  if [[ $released != true ]]; then
    state_reason=$(squeue -h -j "$job_id" -o '%T|%r' 2>/dev/null || true)
    [[ $state_reason != 'PENDING|JobHeldUser' ]] || scancel "$job_id" || true
  fi
  trap - EXIT
  exit "$code"
}
trap cancel_pristine_held_on_exit EXIT

attempt_tmp=$control/SUBMISSION_ATTEMPT.tsv.tmp.$$
printf 'stage\tbatch_id\tjob_id\th200_minutes\tmaximum_cost_usd\nrecovery_continuation_batch\t%s\t%s\t%s\t%s\n' \
  "$batch_id" "$job_id" "$minutes" "$cost" > "$attempt_tmp"
chmod 0400 "$attempt_tmp"
mv "$attempt_tmp" "$control/SUBMISSION_ATTEMPT.tsv"

job_record=$(scontrol show job "$job_id" -o | tr ' ' '\n')
test "$(awk -F= '$1=="JobState" {print $2; exit}' <<< "$job_record")" = PENDING
test "$(awk -F= '$1=="Reason" {print $2; exit}' <<< "$job_record")" = JobHeldUser
test "$(awk -F= '$1=="Requeue" {print $2; exit}' <<< "$job_record")" = 0
test "$(awk -F= '$1=="Account" {print $2; exit}' <<< "$job_record")" = stf
test "$(awk -F= '$1=="Partition" {print $2; exit}' <<< "$job_record")" = gpu-h200
test "$(awk -F= '$1=="QOS" {print $2; exit}' <<< "$job_record")" = normal
test "$(awk -F= '$1=="JobName" {print $2; exit}' <<< "$job_record")" = "$expected_name"
test "$(awk -F= '$1=="TimeLimit" {print $2; exit}' <<< "$job_record")" = 01:00:00
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

submitted_tmp=$control/SUBMITTED.tmp.$$
printf 'protocol_id=massive_medical_kalai_s1_recovery_continuation_v1\nstage=recovery_continuation_batch\nbatch_id=%s\njob_id=%s\nheld_first=true\nheld_audit_passed=true\nrepository_commit=%s\nrestart_or_resume_authorized=false\nautomatic_next_batch_authorized=false\n' \
  "$batch_id" "$job_id" "$(git rev-parse HEAD)" > "$submitted_tmp"
chmod 0400 "$submitted_tmp"
mv "$submitted_tmp" "$control/SUBMITTED"
release_tmp=$control/RELEASE_AUTHORIZED.tmp.$$
printf 'protocol_id=massive_medical_kalai_s1_recovery_continuation_v1\nstage=recovery_continuation_batch\nbatch_id=%s\njob_id=%s\nheld_audit_passed=true\nrelease_authorized=true\nrestart_or_resume_authorized=false\nautomatic_next_batch_authorized=false\n' \
  "$batch_id" "$job_id" > "$release_tmp"
chmod 0400 "$release_tmp"
mv "$release_tmp" "$control/RELEASE_AUTHORIZED"
scontrol release "$job_id"
released=true
released_tmp=$control/RELEASED.tmp.$$
printf 'protocol_id=massive_medical_kalai_s1_recovery_continuation_v1\nstage=recovery_continuation_batch\nbatch_id=%s\njob_id=%s\nreleased=true\nrestart_or_resume_authorized=false\nautomatic_next_batch_authorized=false\n' \
  "$batch_id" "$job_id" > "$released_tmp"
chmod 0400 "$released_tmp"
mv "$released_tmp" "$control/RELEASED"
trap - EXIT
echo "Submitted and released one-shot recovery-continuation $batch_id as job $job_id."
echo 'No next batch, restart, resume, retry, replacement, requeue, or API call was authorized.'
