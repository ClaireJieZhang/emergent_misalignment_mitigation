#!/bin/bash
# Read-only status for the one-shot exploratory smoke probe recovery v2.

set -euo pipefail

[[ $# -eq 0 ]] || {
  echo 'Usage: scripts/status_massive_medical_union_composition_exploratory_smoke_probe_recovery_v2_tillicum.sh' >&2
  exit 2
}

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-composition-exploratory-smoke-probe-recovery-v2
output=$root/outputs/massive_medical_union_composition_exploratory_smoke_probe_recovery_v2
control=$output/control
env_root=$root/envs/subliminal-mitigate-py311
auditor=$repo/scripts/audit_massive_medical_union_composition_exploratory_smoke_probe_recovery_v2.py

test -d "$repo" || {
  echo 'RECOVERY_NOT_STAGED'
  exit 0
}
test -d "$control" || {
  echo 'RECOVERY_CONTROL_NOT_STAGED'
  exit 0
}

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY
unset TRANSFORMERS_CACHE
module load conda/Miniforge3-25.3.1-3
conda activate "$env_root"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=$root/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

cd "$repo"
python "$auditor" audit-prep
if [[ -e $control/SMOKE_PROBE_RECOVERY_CPU_PREFLIGHT.json ]]; then
  python "$auditor" audit-preflight
else
  echo 'RECOVERY_CPU_PREFLIGHT_NOT_YET_SEALED'
fi
if [[ -e $control/STAGED ]]; then
  python "$auditor" audit-staged
else
  echo 'RECOVERY_STAGED_MARKER_NOT_YET_SEALED'
fi

if [[ -e $control/SMOKE_PROBE_RECOVERY_JOB.json ]]; then
  job_id=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["job_id"])' "$control/SMOKE_PROBE_RECOVERY_JOB.json")
  queue=$(squeue -h -j "$job_id" -o '%i|%j|%T|%M|%l|%R' || true)
  accounting=$(sacct -n -X -P -j "$job_id" --format=JobIDRaw,JobName,State,Elapsed,Timelimit,AllocTRES,ReqTRES,ExitCode || true)
  echo "RECOVERY_JOB=$job_id SQUEUE=${queue:-ABSENT}"
  echo "RECOVERY_SACCT=${accounting:-ABSENT}"
  if [[ -e $control/SMOKE_PROBE_RECOVERY_RESULT.json ]]; then
    python "$auditor" audit-result
    state=$(printf '%s\n' "$accounting" | awk -F'|' -v id="$job_id" '$1 == id {print $3}')
    case $state in
      COMPLETED)
        python "$auditor" audit-terminal
        echo 'RECOVERY_TERMINAL_SEALED; FUTURE CONFIRMATION IS NOT AUTHORIZED HERE'
        ;;
      FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*|PREEMPTED*)
        echo "RECOVERY_RESULT_SEALED_BUT_JOB_TERMINAL_FAILURE state=$state"
        if [[ -e $control/STOPPED_smoke_probe_recovery ]]; then
          cat "$control/STOPPED_smoke_probe_recovery"
        else
          echo 'RECOVERY_TERMINAL_FAILURE_STOP_SENTINEL_MISSING'
        fi
        ;;
      *)
        echo "RECOVERY_RESULT_SEALED_AWAITING_TERMINAL state=${state:-ACCOUNTING_PENDING}"
        ;;
    esac
  elif [[ -e $control/STOPPED_smoke_probe_recovery ]]; then
    echo 'RECOVERY_STOPPED_TERMINAL; RETRY IS FORBIDDEN'
    cat "$control/STOPPED_smoke_probe_recovery"
  else
    state=$(printf '%s\n' "$accounting" | awk -F'|' -v id="$job_id" '$1 == id {print $3}')
    case $state in
      COMPLETED|FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*|PREEMPTED*)
        echo "RECOVERY_TERMINAL_UNSEALED state=$state"
        ;;
      *)
        echo 'RECOVERY_PENDING_OR_RUNNING'
        ;;
    esac
  fi
else
  echo 'RECOVERY_STAGED_NOT_SUBMITTED'
fi

echo 'CONFIRMATION: ABSENT_BY_DESIGN'
echo 'EXTERNAL_API: FORBIDDEN'
