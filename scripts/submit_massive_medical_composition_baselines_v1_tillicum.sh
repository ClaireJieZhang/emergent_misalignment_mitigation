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
repo=$root/projects/subliminal-mitigate-mmu-composition-baselines-v1
output=$root/outputs/massive_medical_composition_baselines_v1
control=$output/control
authorizer=$repo/scripts/authorize_massive_medical_composition_baselines_v1.py

case "$stage" in
  union_training)
    sbatch_file=$repo/scripts/sbatch_massive_medical_composition_baselines_v1_union_training_tillicum_h200.sbatch
    ;;
  direct_generation)
    sbatch_file=$repo/scripts/sbatch_massive_medical_composition_baselines_v1_direct_generation_tillicum_h200.sbatch
    ;;
  whole_output_smoke)
    sbatch_file=$repo/scripts/sbatch_massive_medical_composition_baselines_v1_whole_output_smoke_tillicum_h200.sbatch
    ;;
  *)
    usage
    ;;
esac

test -s "$control/CPU_STAGE.json"
test ! -e "$control/${stage^^}_SUBMISSION_LOCK"
test ! -e "$control/${stage^^}_RESULT.json"
test ! -e "$control/${stage^^}_STOPPED"
mkdir "$control/${stage^^}_SUBMISSION_LOCK"

python "$authorizer" write \
  --stage "$stage" \
  --output-root "$output" \
  --repo-root "$repo" \
  --ack-h200-minutes "$minutes" \
  --ack-max-cost-usd "$cost" \
  --ack-program-ceiling-usd "$ceiling"
python "$authorizer" verify --stage "$stage" --output-root "$output" --repo-root "$repo"

job_id=$(sbatch --parsable --hold "$sbatch_file")
[[ $job_id =~ ^[0-9]+$ ]]
record=$control/${stage^^}_SUBMITTED
printf 'protocol_id=massive_medical_composition_baselines_v1\nstage=%s\njob_id=%s\nheld_first=true\nreleased=false\nrestart_or_resume_authorized=false\n' \
  "$stage" "$job_id" > "$record"
chmod 0400 "$record"
scontrol release "$job_id"
temporary=$control/${stage^^}_RELEASED.tmp.$$
printf 'protocol_id=massive_medical_composition_baselines_v1\nstage=%s\njob_id=%s\nreleased=true\nrestart_or_resume_authorized=false\n' \
  "$stage" "$job_id" > "$temporary"
chmod 0400 "$temporary"
mv "$temporary" "$control/${stage^^}_RELEASED"
echo "Submitted and released $stage as one-shot job $job_id."
echo 'No dependency, retry, resume, full whole-output run, or API call was authorized.'
