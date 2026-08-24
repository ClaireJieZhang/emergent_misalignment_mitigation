#!/bin/bash
# Read-only status for both held-first jobs and the login-node finalizer.

set -euo pipefail
umask 077

[[ $# -eq 0 ]] || {
  echo 'Usage: scripts/status_massive_medical_union_composition_exploratory_v1_tillicum.sh' >&2
  exit 2
}

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-composition-exploratory-v1
output=$root/outputs/massive_medical_union_composition_exploratory_v1
control=$output/control
env_root=$root/envs/subliminal-mitigate-py311
auditor=$repo/scripts/audit_massive_medical_union_composition_exploratory_workflow_v1.py

module load conda/Miniforge3-25.3.1-3
conda activate "$env_root"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-composition-exploratory-v1-status-pyc

cd "$repo"
python "$auditor" audit-prep
for phase in smoke confirmation; do
  case $phase in
    smoke) preflight=$control/SMOKE_CPU_PREFLIGHT.json ;;
    confirmation) preflight=$control/CONFIRMATION_CPU_PREFLIGHT.json ;;
  esac
  if [[ -e $preflight ]]; then
    python "$auditor" audit-preflight --stage "$phase"
  else
    echo "${phase^^}: CPU_PREFLIGHT_NOT_YET_SEALED"
  fi
done

stage_status() {
  stage=$1
  case $stage in
    smoke)
      job_file=$control/SMOKE_JOB.json
      result_file=$control/SMOKE_RESULT.json
      stop_file=$control/STOPPED_smoke
      lock=$control/SMOKE_SUBMISSION_LOCK
      log_prefix=$root/outputs/logs/massive_medical_union_composition_exploratory_v1_smoke
      ;;
    confirmation)
      job_file=$control/CONFIRMATION_JOB.json
      result_file=$control/CONFIRMATION_RESULT.json
      stop_file=$control/STOPPED_confirmation
      lock=$control/CONFIRMATION_SUBMISSION_LOCK
      log_prefix=$root/outputs/logs/massive_medical_union_composition_exploratory_v1_confirmation
      ;;
  esac
  if [[ ! -e $job_file ]]; then
    if [[ -d $lock ]]; then
      echo "${stage^^}: LOCKED_WITHOUT_SEALED_JOB"
    else
      echo "${stage^^}: NOT_SUBMITTED"
    fi
    return
  fi
  job_id=$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["job_id"])' "$job_file")
  queue=$(squeue -h -j "$job_id" -o '%T|%r|%M|%L' || true)
  if [[ -n $queue ]]; then
    echo "${stage^^}: JOB=$job_id SQUEUE=$queue"
    if [[ -e $result_file ]]; then
      python "$auditor" audit-result --stage "$stage"
      echo "${stage^^}: SEALED_RESULT_AWAITING_DURABLE_ACCOUNTING"
    elif [[ -e $stop_file ]]; then
      echo "${stage^^}: RUNNING_WITH_STOP_SENTINEL $stop_file"
    fi
    return
  fi
  accounting=$(sacct -n -X -P -j "$job_id" --format=JobIDRaw,State,Elapsed,ExitCode | awk -F'|' -v id="$job_id" '$1 == id {print; count++} END {if (count != 1) exit 1}' || true)
  if [[ -z $accounting ]]; then
    echo "${stage^^}: JOB=$job_id ACCOUNTING_NOT_YET_AVAILABLE"
    return
  fi
  state=$(printf '%s\n' "$accounting" | awk -F'|' '{print $2}')
  echo "${stage^^}: JOB=$job_id SACCT=$accounting"
  if [[ -e $result_file ]]; then
    python "$auditor" audit-result --stage "$stage"
    if [[ $state == COMPLETED ]]; then
      python "$auditor" audit-terminal --stage "$stage"
    else
      echo "${stage^^}: TERMINAL_STATE_CONFLICTS_WITH_SEALED_RESULT"
      return 1
    fi
  else
    echo "${stage^^}: TERMINAL_UNSEALED state=$state stop=${stop_file} stdout=${log_prefix}_${job_id}.out stderr=${log_prefix}_${job_id}.err"
    [[ -e $stop_file ]] && sed -n '1,80p' "$stop_file"
  fi
}

stage_status smoke
stage_status confirmation

if [[ -e $control/FINAL_RESULT.json ]]; then
  python "$auditor" audit-final-result
elif [[ -e $control/STOPPED_finalize ]]; then
  echo "FINALIZER: TERMINAL_UNSEALED $control/STOPPED_finalize"
  sed -n '1,80p' "$control/STOPPED_finalize"
elif [[ -e $control/EXTERNAL_JUDGE_AUTHORIZED_MAX_COST_USD_0.75.json ]]; then
  echo 'FINALIZER: AUTHORIZED_OR_IN_PROGRESS (240 calls / $0.75 exact ceiling; no retry)'
elif [[ -e $output/evaluation/confirmation/prejudge/AWAITING_EXTERNAL_JUDGE ]]; then
  echo 'FINALIZER: AWAITING_EXTERNAL_JUDGE'
else
  echo 'FINALIZER: NOT_ELIGIBLE_OR_NOT_REACHED'
fi
