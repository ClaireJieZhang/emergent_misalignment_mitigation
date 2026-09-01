#!/bin/bash
# Submit exactly one separately authorized baseline GPU stage.

set -euo pipefail
umask 077
ulimit -c 0

usage() {
  echo 'Usage: submit_massive_medical_composition_baselines_v1_tillicum.sh STAGE --ack-h200-minutes N --ack-max-cost-usd USD --ack-program-ceiling-usd USD' >&2
  exit 2
}

[[ $# -eq 7 ]] || usage
stage=$1
[[ $2 == --ack-h200-minutes && $4 == --ack-max-cost-usd && $6 == --ack-program-ceiling-usd ]] || usage
minutes=$3
cost=$5
ceiling=$7

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-composition-baselines-v1-stage-recovery-v2
output=$root/outputs/massive_medical_composition_baselines_v1
control=$output/control
logs=$root/outputs/logs
authorizer=$repo/scripts/authorize_massive_medical_composition_baselines_v1.py

case "$stage" in
  union_training)
    sbatch_file=$repo/scripts/sbatch_massive_medical_composition_baselines_v1_union_training_tillicum_h200.sbatch
    expected_name=mmu_base_union_v1
    expected_time=00:55:00
    log_glob=$logs/massive_medical_composition_baselines_v1_union_\*
    ;;
  direct_generation)
    sbatch_file=$repo/scripts/sbatch_massive_medical_composition_baselines_v1_direct_generation_tillicum_h200.sbatch
    expected_name=mmu_base_direct_v1
    expected_time=00:30:00
    log_glob=$logs/massive_medical_composition_baselines_v1_direct_\*
    ;;
  whole_output_smoke)
    sbatch_file=$repo/scripts/sbatch_massive_medical_composition_baselines_v1_whole_output_smoke_tillicum_h200.sbatch
    expected_name=mmu_base_kalai_v1
    expected_time=00:20:00
    log_glob=$logs/massive_medical_composition_baselines_v1_kalai_smoke_\*
    ;;
  *)
    usage
    ;;
esac

cd "$repo"
test -z "$(git status --porcelain)"
test -s "$control/CPU_STAGE.json"
test ! -e "$control/${stage^^}_SUBMISSION_LOCK"
test ! -e "$control/${stage^^}_SUBMISSION_ATTEMPT.tsv"
test ! -e "$control/${stage^^}_SUBMITTED"
test ! -e "$control/${stage^^}_RELEASE_AUTHORIZED"
test ! -e "$control/${stage^^}_RELEASED"
test ! -e "$control/${stage^^}_RESULT.json"
test ! -e "$control/${stage^^}_STOPPED"
if compgen -G "$log_glob" >/dev/null; then
  echo "$stage log namespace is not fresh." >&2
  exit 4
fi
mkdir "$control/${stage^^}_SUBMISSION_LOCK"
owner_tmp=$control/${stage^^}_SUBMISSION_LOCK/owner.tmp.$$
printf 'protocol_id=massive_medical_composition_baselines_v1\nstage=%s\nrepository_commit=%s\nrestart_or_resume_authorized=false\n' \
  "$stage" "$(git rev-parse HEAD)" > "$owner_tmp"
chmod 0400 "$owner_tmp"
mv "$owner_tmp" "$control/${stage^^}_SUBMISSION_LOCK/owner"

python "$authorizer" write \
  --stage "$stage" \
  --output-root "$output" \
  --repo-root "$repo" \
  --ack-h200-minutes "$minutes" \
  --ack-max-cost-usd "$cost" \
  --ack-program-ceiling-usd "$ceiling"
python "$authorizer" verify --stage "$stage" --output-root "$output" --repo-root "$repo"

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

attempt=$control/${stage^^}_SUBMISSION_ATTEMPT.tsv
attempt_tmp=$attempt.tmp.$$
printf 'stage\tjob_id\th200_minutes\tmaximum_cost_usd\n%s\t%s\t%s\t%s\n' \
  "$stage" "$job_id" "$minutes" "$cost" > "$attempt_tmp"
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

record=$control/${stage^^}_SUBMITTED
record_tmp=$record.tmp.$$
printf 'protocol_id=massive_medical_composition_baselines_v1\nstage=%s\njob_id=%s\nheld_first=true\nheld_audit_passed=true\nrepository_commit=%s\nrestart_or_resume_authorized=false\n' \
  "$stage" "$job_id" "$(git rev-parse HEAD)" > "$record_tmp"
chmod 0400 "$record_tmp"
mv "$record_tmp" "$record"

release_auth=$control/${stage^^}_RELEASE_AUTHORIZED
release_tmp=$release_auth.tmp.$$
printf 'protocol_id=massive_medical_composition_baselines_v1\nstage=%s\njob_id=%s\nheld_audit_passed=true\nrelease_authorized=true\nrestart_or_resume_authorized=false\n' \
  "$stage" "$job_id" > "$release_tmp"
chmod 0400 "$release_tmp"
mv "$release_tmp" "$release_auth"
scontrol release "$job_id"
released=true
temporary=$control/${stage^^}_RELEASED.tmp.$$
printf 'protocol_id=massive_medical_composition_baselines_v1\nstage=%s\njob_id=%s\nreleased=true\nrestart_or_resume_authorized=false\n' \
  "$stage" "$job_id" > "$temporary"
chmod 0400 "$temporary"
mv "$temporary" "$control/${stage^^}_RELEASED"
trap - EXIT
echo "Submitted and released $stage as one-shot job $job_id."
echo 'No dependency, retry, resume, full whole-output run, or API call was authorized.'
