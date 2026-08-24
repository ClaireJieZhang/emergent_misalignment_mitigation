#!/bin/bash
# Login-node-only external judging and final exploratory decision.

set -euo pipefail
umask 077
ulimit -c 0

[[ $# -eq 3 && $1 == external-judge && $2 == --ack-max-api-cost-usd && $3 == 0.75 ]] || {
  echo 'Usage: scripts/finalize_massive_medical_union_composition_exploratory_v1_tillicum.sh external-judge --ack-max-api-cost-usd 0.75' >&2
  exit 2
}
[[ -n ${OPENAI_API_KEY:-} ]] || {
  echo 'OPENAI_API_KEY must be present only for this login-node finalizer.' >&2
  exit 3
}
[[ -z ${SLURM_JOB_ID:-} ]] || {
  echo 'The external judge finalizer must not run inside a Slurm job.' >&2
  exit 4
}

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-composition-exploratory-v1
output=$root/outputs/massive_medical_union_composition_exploratory_v1
protocol=$output/protocol
generation=$output/generation
evaluation=$output/evaluation
control=$output/control
env_root=$root/envs/subliminal-mitigate-py311
auditor=$repo/scripts/audit_massive_medical_union_composition_exploratory_workflow_v1.py
judge=$repo/scripts/judge_massive_medical_union_composition_exploratory_v1.py
merge=$repo/scripts/merge_massive_medical_union_composition_exploratory_v1.py
summarizer=$repo/scripts/summarize_massive_medical_union_composition_exploratory_v1.py
lock=$control/FINALIZER_LOCK
stop=$control/STOPPED_finalize
result=$control/FINAL_RESULT.json
completed=false

on_exit() {
  code=$?
  if [[ $completed != true && ! -e $result && ! -e $stop ]]; then
    temporary=$stop.tmp.$$
    printf 'workflow_id=massive_medical_union_composition_exploratory_workflow_v1\nstage=external_judge_finalizer\nexit_code=%s\nretry_authorized=false\n' \
      "$code" > "$temporary"
    chmod 0400 "$temporary"
    mv "$temporary" "$stop"
  fi
}

cd "$repo"
module load conda/Miniforge3-25.3.1-3
conda activate "$env_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-composition-exploratory-v1-finalizer-pyc-$$
export DO_NOT_TRACK=1

python "$auditor" audit-terminal --stage confirmation
python "$judge" \
  --protocol-manifest "$protocol/manifest.json" \
  --prejudge-sentinel "$evaluation/confirmation/prejudge/AWAITING_EXTERNAL_JUDGE" \
  --generation "ordinary_quorum_m4_q3=$generation/confirmation/ordinary_quorum_m4_q3/medical/generation.json" \
  --generation "ordinary_min_m4_q4=$generation/confirmation/ordinary_min_m4_q4/medical/generation.json" \
  --generation "delta_min_m4_q4=$generation/confirmation/delta_min_m4_q4/medical/generation.json" \
  --prompt-file "$protocol/medical/prompts.json" \
  --checkpoint-file "$evaluation/medical/judge_checkpoint.json" \
  --output-file "$evaluation/medical/judgments_new.json" \
  --validate-only

mkdir "$lock" || {
  echo 'Finalizer is permanently locked; API retry is forbidden.' >&2
  exit 5
}
trap on_exit EXIT
owner_tmp=$lock/owner.tmp.$$
printf 'workflow_id=massive_medical_union_composition_exploratory_workflow_v1\nstage=external_judge_finalizer\nrepo_commit=%s\nmaximum_calls=240\nmaximum_cost_usd=0.75\n' \
  "$(git rev-parse HEAD)" > "$owner_tmp"
chmod 0400 "$owner_tmp"
mv "$owner_tmp" "$lock/owner"

python "$auditor" write-final-auth
python "$auditor" audit-final-auth
python "$judge" \
  --protocol-manifest "$protocol/manifest.json" \
  --prejudge-sentinel "$evaluation/confirmation/prejudge/AWAITING_EXTERNAL_JUDGE" \
  --generation "ordinary_quorum_m4_q3=$generation/confirmation/ordinary_quorum_m4_q3/medical/generation.json" \
  --generation "ordinary_min_m4_q4=$generation/confirmation/ordinary_min_m4_q4/medical/generation.json" \
  --generation "delta_min_m4_q4=$generation/confirmation/delta_min_m4_q4/medical/generation.json" \
  --prompt-file "$protocol/medical/prompts.json" \
  --checkpoint-file "$evaluation/medical/judge_checkpoint.json" \
  --output-file "$evaluation/medical/judgments_new.json"
unset OPENAI_API_KEY

historical_path=$(python "$auditor" historical-a-path)
python "$merge" \
  --protocol-manifest "$protocol/manifest.json" \
  --historical-judgments "$historical_path" \
  --new-judgments "$evaluation/medical/judgments_new.json" \
  --output-file "$evaluation/medical/judgments_merged.json"

mkdir -p "$evaluation/final"
set +e
python "$summarizer" final \
  --protocol-manifest "$protocol/manifest.json" \
  --smoke-gate "$evaluation/smoke/gate/EXPLORATORY_SMOKE_PASSED" \
  --prejudge-sentinel "$evaluation/confirmation/prejudge/AWAITING_EXTERNAL_JUDGE" \
  --base-score "$evaluation/confirmation/scores/pi_base.json" \
  --method-score "ordinary_quorum_m4_q3=$evaluation/confirmation/scores/ordinary_quorum_m4_q3.json" \
  --method-score "ordinary_min_m4_q4=$evaluation/confirmation/scores/ordinary_min_m4_q4.json" \
  --method-score "delta_min_m4_q4=$evaluation/confirmation/scores/delta_min_m4_q4.json" \
  --direct-comparator "pi_base=$protocol/direct_confirmation/pi_base.json" \
  --direct-comparator "pi_A=$protocol/direct_confirmation/pi_A.json" \
  --direct-comparator "pi_B1=$protocol/direct_confirmation/pi_B1.json" \
  --direct-comparator "pi_B2=$protocol/direct_confirmation/pi_B2.json" \
  --direct-comparator "pi_B3=$protocol/direct_confirmation/pi_B3.json" \
  --medical-judgments "$evaluation/medical/judgments_merged.json" \
  --output-dir "$evaluation/final"
final_code=$?
set -e
[[ $final_code -eq 0 || $final_code -eq 2 ]] || exit "$final_code"

python "$auditor" write-final-result
completed=true
trap - EXIT
echo 'FINAL_EXPLORATORY_RESULT_SEALED'
