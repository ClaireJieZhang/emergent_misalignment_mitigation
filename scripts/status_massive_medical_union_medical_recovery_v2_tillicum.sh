#!/bin/bash
# Read-only status for the second Wave-1 medical-only recovery.

set -uo pipefail

classify_scheduler_state() {
  local queue_present=$1
  local state=$2
  local exit_code=$3
  if [[ "$queue_present" == false && "$state" == COMPLETED* ]]; then
    echo "TERMINAL_UNSEALED_MEDICAL_RECOVERY_V2: job completed without GPU seal (state=$state exit=$exit_code)."
  elif [[ "$queue_present" == false && "$state" =~ ^(FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED|SPECIAL_EXIT) ]]; then
    echo "TERMINAL_UNSEALED_MEDICAL_RECOVERY_V2: state=$state exit=$exit_code."
  elif [[ "$queue_present" == false && -z "$state" ]]; then
    echo 'RECOVERY_V2_SCHEDULER_STATE_UNRESOLVED: absent from queue with no accounting row.'
  else
    echo 'MEDICAL_RECOVERY_V2_RUNNING_OR_PENDING'
  fi
}

# Pure read-only classifier entrypoint used by the regression suite.
if [[ "${1:-}" == --classify-scheduler-state ]]; then
  [[ $# -eq 4 ]] || exit 2
  classify_scheduler_state "$2" "$3" "$4"
  exit 0
fi

TILLICUM_ROOT=/gpfs/projects/stf/claizhan/subliminal-mitigate
REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate-mmu-medical-recovery-v2
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/massive_medical_union_pilot_v1
OLD_CONTROL=$OUTPUT_ROOT/control
V1_CONTROL=$OLD_CONTROL/medical_recovery_v1
CONTROL_ROOT=$OLD_CONTROL/medical_recovery_v2
EVAL_ROOT=$OUTPUT_ROOT/evaluation/wave1/medical_recovery_v2
JOBS_FILE=$CONTROL_ROOT/jobs.tsv
PINNED_PYTHON=$TILLICUM_ROOT/envs/subliminal-mitigate-py311/bin/python

echo '=== MASSIVE + medical Wave-1 medical recovery v2 ==='
echo 'Original released ceiling: 80 H200-minutes / $1.20.'
echo 'Failed medical recovery v1 ceiling: 10 H200-minutes / $0.15 (job 248197 used 5 seconds).'
echo 'Medical recovery v2 ceiling: 10 H200-minutes / $0.15.'
echo 'Cumulative released ceiling: 100 H200-minutes / $1.50.'
echo 'Separate confirmatory judge ceiling after GPU completion: 240 calls / $0.75.'
echo 'Retraining, MASSIVE regeneration, Wave 2, and quorum: none.'

if [[ -s "$OLD_CONTROL/STOPPED_evaluate" ]]; then
  echo 'PRESERVED historical STOPPED_evaluate from job 247699.'
fi
if [[ -s "$V1_CONTROL/STOPPED_medical_recovery" ]]; then
  echo 'PRESERVED medical-recovery-v1 STOP from job 248197.'
fi

echo '=== Recovery-v2 control records ==='
for name in PREP.json SUBMISSION_LOCK SUBMISSION_ATTEMPT.tsv jobs.tsv \
  AUTHORIZED_MAX_COST_USD_0.15.json SUBMITTED RELEASED STOPPED_submission \
  STOPPED_medical_recovery GPU_MEDICAL_RECOVERY_COMPLETE \
  EXTERNAL_JUDGE_LOCK AWAITING_EXTERNAL_JUDGE_RESUME STOPPED_finalize \
  GO_MASSIVE_UNION_WAVE1 STOPPED_MASSIVE_UNION_WAVE1; do
  [[ -e "$CONTROL_ROOT/$name" ]] && printf 'PRESENT %s\n' "$name"
done

queue_present=false
accounting_state=
accounting_exit=
if [[ -s "$JOBS_FILE" ]]; then
  echo '=== Recovery-v2 Slurm job ==='
  while IFS=$'\t' read -r stage job_id max_minutes released; do
    [[ "$stage" == stage ]] && continue
    [[ "$job_id" =~ ^[0-9]+$ ]] || continue
    printf '%s job=%s cap=%sm released=%s\n' "$stage" "$job_id" "$max_minutes" "$released"
    queue_line=$(squeue -h -j "$job_id" -o 'queue state=%T reason=%R elapsed=%M limit=%l node=%N' 2>/dev/null || true)
    if [[ -n "$queue_line" ]]; then
      queue_present=true
      printf '%s\n' "$queue_line"
    fi
    accounting_line=$(sacct -n -X -P -j "$job_id" \
      --format=JobIDRaw,State%30,Elapsed,Timelimit,AllocTRES,ExitCode 2>/dev/null | \
      awk -F'|' -v wanted="$job_id" '$1 == wanted {print; exit}' || true)
    if [[ -n "$accounting_line" ]]; then
      printf '%s\n' "$accounting_line"
      IFS='|' read -r _ accounting_state _ _ _ accounting_exit <<< "$accounting_line"
    fi
  done < "$JOBS_FILE"
fi

if [[ -s "$EVAL_ROOT/component_gate/summary.md" ]]; then
  echo '=== Sealed recovery-v2 component summary ==='
  sed -n '1,220p' "$EVAL_ROOT/component_gate/summary.md"
fi

echo '=== Decision ==='
if [[ -s "$CONTROL_ROOT/GO_MASSIVE_UNION_WAVE1" ]]; then
  echo 'FINAL_EVALUATION_COMPLETE: GO_MASSIVE_UNION_WAVE1'
  echo 'Wave 2 remains unreleased and requires a separate explicit workflow.'
elif [[ -s "$CONTROL_ROOT/STOPPED_MASSIVE_UNION_WAVE1" ]]; then
  echo 'FINAL_EVALUATION_COMPLETE: STOPPED_MASSIVE_UNION_WAVE1'
elif [[ -s "$CONTROL_ROOT/STOPPED_medical_recovery" ]]; then
  echo 'TERMINAL_MEDICAL_RECOVERY_V2_STOP'
  sed -n '1,120p' "$CONTROL_ROOT/STOPPED_medical_recovery"
elif [[ -s "$CONTROL_ROOT/STOPPED_submission" ]]; then
  echo 'TERMINAL_RECOVERY_V2_SUBMISSION_STOP: no retry is authorized.'
  sed -n '1,120p' "$CONTROL_ROOT/STOPPED_submission"
elif [[ -s "$CONTROL_ROOT/STOPPED_finalize" ]]; then
  echo 'TERMINAL_RECOVERY_V2_FINALIZE_STOP'
  sed -n '1,120p' "$CONTROL_ROOT/STOPPED_finalize"
elif [[ -s "$CONTROL_ROOT/AWAITING_EXTERNAL_JUDGE_RESUME" ]]; then
  echo 'AWAITING_EXTERNAL_JUDGE_RESUME: no automatic retry and no GPU retry.'
  sed -n '1,120p' "$CONTROL_ROOT/AWAITING_EXTERNAL_JUDGE_RESUME"
elif [[ -s "$CONTROL_ROOT/GPU_MEDICAL_RECOVERY_COMPLETE" ]]; then
  if [[ -x "$PINNED_PYTHON" && -s "$REPO_ROOT/scripts/audit_massive_medical_union_medical_recovery_v2.py" ]] && \
     "$PINNED_PYTHON" "$REPO_ROOT/scripts/audit_massive_medical_union_medical_recovery_v2.py" audit-gpu >/dev/null 2>&1; then
    echo 'AWAITING_EXTERNAL_JUDGE: verified 240/240 normal stops; no API call has been made by the GPU workflow.'
  else
     echo 'RECOVERY_V2_GPU_ARTIFACT_AUDIT_FAILED'
  fi
elif [[ -s "$CONTROL_ROOT/RELEASED" ]]; then
  scheduler_decision=$(classify_scheduler_state "$queue_present" "$accounting_state" "$accounting_exit")
  echo "$scheduler_decision"
  if [[ "$scheduler_decision" == TERMINAL_UNSEALED_MEDICAL_RECOVERY_V2:* ]]; then
    echo 'No retry is authorized; inspect the frozen v2 stdout/stderr.'
  fi
elif [[ -s "$CONTROL_ROOT/PREP.json" ]]; then
  echo 'MEDICAL_RECOVERY_V2_STAGED_NOT_SUBMITTED'
else
  echo 'MEDICAL_RECOVERY_V2_NOT_STAGED'
fi
