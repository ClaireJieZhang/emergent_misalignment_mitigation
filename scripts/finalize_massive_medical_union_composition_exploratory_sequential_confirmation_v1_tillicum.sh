#!/bin/bash
# Login-node judge-plan preparation and single-entry exact240 finalizer.
set -euo pipefail
umask 077
ulimit -c 0
mode=${1:-}
if [[ $mode == prepare-plan ]]; then
  [[ $# -eq 1 ]] || { echo 'Usage: finalize_..._tillicum.sh prepare-plan' >&2; exit 2; }
  [[ -z ${OPENAI_API_KEY:-} ]] || { echo 'OPENAI_API_KEY must be absent during no-API plan preparation.' >&2; exit 3; }
elif [[ $mode == external-judge ]]; then
  [[ $# -eq 11 && $2 == --ack-prior-program-actual-usd && $3 == 1.696936 && $4 == --ack-benefit-actual-usd && $6 == --ack-medical-actual-usd && $8 == --ack-max-cost-usd && $9 == 0.75 && ${10} == --ack-program-ceiling-usd && ${11} == 5.0 ]] || {
    echo 'Usage: finalize_..._tillicum.sh external-judge --ack-prior-program-actual-usd 1.696936 --ack-benefit-actual-usd <sealed-value> --ack-medical-actual-usd <sealed-value> --ack-max-cost-usd 0.75 --ack-program-ceiling-usd 5.0' >&2; exit 2
  }
  benefit_actual=$5; medical_actual=$7
  [[ -n ${OPENAI_API_KEY:-} ]] || { echo 'OPENAI_API_KEY must be set for the exact external-judge step.' >&2; exit 3; }
else
  echo 'Usage: finalize_..._tillicum.sh prepare-plan OR external-judge with sealed GPU actuals, 0.75 API cap, and 5.0 program ceiling acknowledgements' >&2; exit 2
fi
root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-composition-exploratory-sequential-confirmation-v1-submit-recovery-v3
output=$root/outputs/massive_medical_union_composition_exploratory_sequential_confirmation_v1_submit_recovery_v3
protocol=$output/protocol; generation=$output/generation; evaluation=$output/evaluation; control=$output/control
auditor=$repo/scripts/audit_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py
judge=$repo/scripts/judge_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py
merge=$repo/scripts/merge_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py
summarizer=$repo/scripts/summarize_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py
lock=$control/FINALIZER_LOCK; stop=$control/STOPPED_external_judge; result=$control/FINAL_RESULT.json
checkpoint=$evaluation/medical/judge_checkpoint.json; new=$evaluation/medical/judgments_new.json; merged=$evaluation/medical/judgments_merged.json
plan=$evaluation/medical/judge_plan.json
cd "$repo"
module load conda/Miniforge3-25.3.1-3
conda activate "$root/envs/subliminal-mitigate-py311"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=$root/tmp/mmu-seq-finalizer-v3-pyc
medical_job=$(python -c 'import json; print(json.load(open("'$control'/MEDICAL_JOB.json"))["job_id"])')
python "$auditor" audit-terminal --stage medical --job-id "$medical_job"
python "$judge" validate-static --protocol-manifest "$protocol/manifest.json"
python "$merge" validate-static --protocol-manifest "$protocol/manifest.json"
if [[ $mode == prepare-plan ]]; then
  test ! -e "$lock"; test ! -e "$stop"; test ! -e "$result"; test ! -e "$control/EXTERNAL_JUDGE_AUTHORIZATION.json"
  test ! -e "$checkpoint"; test ! -e "$new"; test ! -e "$merged"; test ! -e "$evaluation/final"
  if compgen -G "$checkpoint.*" >/dev/null; then echo 'Judge progress namespace is not fresh.' >&2; exit 4; fi
  python "$judge" validate-plan --protocol-manifest "$protocol/manifest.json" \
    --prejudge-sentinel "$evaluation/medical/prejudge/AWAITING_EXTERNAL_JUDGE" \
    --medical-generation "ordinary_quorum_m4_q3=$generation/medical/ordinary_quorum_m4_q3/medical/generation.json" \
    --medical-generation "ordinary_min_m4_q4=$generation/medical/ordinary_min_m4_q4/medical/generation.json" \
    --medical-generation "delta_min_m4_q4=$generation/medical/delta_min_m4_q4/medical/generation.json" \
    --output-file "$plan"
  python "$auditor" audit-judge-plan
  echo SEQUENTIAL_JUDGE_PLAN_READY_NO_API
  exit 0
fi

test ! -e "$lock"; test ! -e "$stop"; test ! -e "$result"; test ! -e "$control/EXTERNAL_JUDGE_AUTHORIZATION.json"
test -e "$plan"; test ! -e "$checkpoint"; test ! -e "$new"; test ! -e "$merged"; test ! -e "$evaluation/final"
if compgen -G "$checkpoint.*" >/dev/null; then echo 'Judge progress namespace is not fresh.' >&2; exit 4; fi
# Reconstruct the complete text-free plan from the live prejudge sentinel and
# all three sealed medical generations before taking the permanent lock.  The
# judge command is write-or-audit, so any plan/rubric/schema/source drift fails
# while restart remains harmless and before API authority exists.
python "$judge" validate-plan --protocol-manifest "$protocol/manifest.json" \
  --prejudge-sentinel "$evaluation/medical/prejudge/AWAITING_EXTERNAL_JUDGE" \
  --medical-generation "ordinary_quorum_m4_q3=$generation/medical/ordinary_quorum_m4_q3/medical/generation.json" \
  --medical-generation "ordinary_min_m4_q4=$generation/medical/ordinary_min_m4_q4/medical/generation.json" \
  --medical-generation "delta_min_m4_q4=$generation/medical/delta_min_m4_q4/medical/generation.json" \
  --output-file "$plan"
python "$auditor" audit-judge-plan
python "$auditor" assert-external-judge-budget --ack-benefit-actual-usd "$benefit_actual" --ack-medical-actual-usd "$medical_actual"
mkdir "$lock" || { echo 'Finalizer is permanently locked; process restart/resume is forbidden.' >&2; exit 4; }
printf 'workflow_id=massive_medical_union_composition_exploratory_sequential_confirmation_v1\nstage=external_judge\nrepo_commit=%s\n' "$(git rev-parse HEAD)" > "$lock/owner.tmp"
chmod 0400 "$lock/owner.tmp"; mv "$lock/owner.tmp" "$lock/owner"
completed=false
on_exit() {
  code=$?; unset OPENAI_API_KEY
  if [[ $completed != true && ! -e $result && ! -e $stop ]]; then
    printf 'workflow_id=massive_medical_union_composition_exploratory_sequential_confirmation_v1\nstage=external_judge\nexit_code=%s\nretry_authorized=false\nprocess_restart_authorized=false\n' "$code" > "$stop.tmp"
    chmod 0400 "$stop.tmp"; mv "$stop.tmp" "$stop"
  fi
}
trap on_exit EXIT
python "$auditor" write-final-auth
python "$auditor" audit-final-auth
python "$judge" external --protocol-manifest "$protocol/manifest.json" \
  --prejudge-sentinel "$evaluation/medical/prejudge/AWAITING_EXTERNAL_JUDGE" \
  --authorization-file "$control/EXTERNAL_JUDGE_AUTHORIZATION.json" \
  --plan-file "$plan" \
  --medical-generation "ordinary_quorum_m4_q3=$generation/medical/ordinary_quorum_m4_q3/medical/generation.json" \
  --medical-generation "ordinary_min_m4_q4=$generation/medical/ordinary_min_m4_q4/medical/generation.json" \
  --medical-generation "delta_min_m4_q4=$generation/medical/delta_min_m4_q4/medical/generation.json" \
  --checkpoint-file "$checkpoint" --output-file "$new"
unset OPENAI_API_KEY
python "$merge" merge --protocol-manifest "$protocol/manifest.json" \
  --prejudge-sentinel "$evaluation/medical/prejudge/AWAITING_EXTERNAL_JUDGE" \
  --historical-judgments "$protocol/historical/A_judgments.json" --new-judgments "$new" --output-file "$merged"
mkdir -p "$evaluation/final"
set +e
python "$summarizer" final --protocol-manifest "$protocol/manifest.json" \
  --benefit-gate "$evaluation/benefit/gate/EXPLORATORY_BENEFIT_PASSED" \
  --prejudge-sentinel "$evaluation/medical/prejudge/AWAITING_EXTERNAL_JUDGE" \
  --medical-judgments "$merged" --output-dir "$evaluation/final"
final_code=$?
set -e
[[ $final_code -eq 0 || $final_code -eq 2 ]] || exit "$final_code"
python "$auditor" write-final-result
python "$auditor" audit-final-result
completed=true; trap - EXIT
echo SEQUENTIAL_FINAL_EVALUATION_COMPLETE
