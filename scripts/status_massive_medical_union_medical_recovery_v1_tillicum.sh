#!/bin/bash
# Read-only status for the Wave-1 medical-only recovery.

set -uo pipefail

TILLICUM_ROOT=/gpfs/projects/stf/claizhan/subliminal-mitigate
REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate-mmu-medical-recovery-v1
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/massive_medical_union_pilot_v1
OLD_CONTROL=$OUTPUT_ROOT/control
CONTROL_ROOT=$OLD_CONTROL/medical_recovery_v1
EVAL_ROOT=$OUTPUT_ROOT/evaluation/wave1/medical_recovery_v1
JOBS_FILE=$CONTROL_ROOT/jobs.tsv
PINNED_PYTHON=$TILLICUM_ROOT/envs/subliminal-mitigate-py311/bin/python

echo '=== MASSIVE + medical Wave-1 medical recovery v1 ==='
echo 'Original released ceiling: 80 H200-minutes / $1.20.'
echo 'Medical-only recovery ceiling: 10 H200-minutes / $0.15.'
echo 'Cumulative released ceiling: 90 H200-minutes / $1.35.'
echo 'Separate confirmatory judge ceiling after GPU completion: 240 calls / $0.75.'
echo 'Retraining, MASSIVE regeneration, Wave 2, and quorum: none.'

if [[ -s "$OLD_CONTROL/STOPPED_evaluate" ]]; then
  echo 'PRESERVED historical STOPPED_evaluate from job 247699.'
fi

echo '=== Recovery control records ==='
for name in PREP.json SUBMISSION_LOCK SUBMISSION_ATTEMPT.tsv jobs.tsv \
  AUTHORIZED_MAX_COST_USD_0.15.json SUBMITTED RELEASED STOPPED_submission \
  STOPPED_medical_recovery GPU_MEDICAL_RECOVERY_COMPLETE \
  EXTERNAL_JUDGE_LOCK AWAITING_EXTERNAL_JUDGE_RESUME STOPPED_finalize \
  GO_MASSIVE_UNION_WAVE1 STOPPED_MASSIVE_UNION_WAVE1; do
  [[ -e "$CONTROL_ROOT/$name" ]] && printf 'PRESENT %s\n' "$name"
done

if [[ -s "$JOBS_FILE" ]]; then
  echo '=== Recovery Slurm job ==='
  while IFS=$'\t' read -r stage job_id max_minutes released; do
    [[ "$stage" == stage ]] && continue
    [[ "$job_id" =~ ^[0-9]+$ ]] || continue
    printf '%s job=%s cap=%sm released=%s\n' "$stage" "$job_id" "$max_minutes" "$released"
    squeue -h -j "$job_id" -o 'queue state=%T reason=%R elapsed=%M limit=%l node=%N' 2>/dev/null || true
    sacct -n -X -j "$job_id" --format=JobIDRaw,State,Elapsed,Timelimit,AllocTRES,ExitCode \
      2>/dev/null | sed '/^[[:space:]]*$/d' || true
  done < "$JOBS_FILE"
fi

if [[ -s "$EVAL_ROOT/component_gate/summary.md" ]]; then
  echo '=== Sealed recovery component summary ==='
  sed -n '1,220p' "$EVAL_ROOT/component_gate/summary.md"
fi

echo '=== Decision ==='
if [[ -s "$CONTROL_ROOT/GO_MASSIVE_UNION_WAVE1" ]]; then
  echo 'FINAL_EVALUATION_COMPLETE: GO_MASSIVE_UNION_WAVE1'
  echo 'Wave 2 remains unreleased and requires a separate explicit workflow.'
elif [[ -s "$CONTROL_ROOT/STOPPED_MASSIVE_UNION_WAVE1" ]]; then
  echo 'FINAL_EVALUATION_COMPLETE: STOPPED_MASSIVE_UNION_WAVE1'
elif [[ -s "$CONTROL_ROOT/STOPPED_medical_recovery" ]]; then
  echo 'TERMINAL_MEDICAL_RECOVERY_STOP'
  sed -n '1,120p' "$CONTROL_ROOT/STOPPED_medical_recovery"
elif [[ -s "$CONTROL_ROOT/STOPPED_submission" ]]; then
  echo 'TERMINAL_RECOVERY_SUBMISSION_STOP: no retry is authorized.'
  sed -n '1,120p' "$CONTROL_ROOT/STOPPED_submission"
elif [[ -s "$CONTROL_ROOT/STOPPED_finalize" ]]; then
  echo 'TERMINAL_RECOVERY_FINALIZE_STOP'
  sed -n '1,120p' "$CONTROL_ROOT/STOPPED_finalize"
elif [[ -s "$CONTROL_ROOT/AWAITING_EXTERNAL_JUDGE_RESUME" ]]; then
  echo 'AWAITING_EXTERNAL_JUDGE_RESUME: no automatic retry and no GPU retry.'
  sed -n '1,120p' "$CONTROL_ROOT/AWAITING_EXTERNAL_JUDGE_RESUME"
elif [[ -s "$CONTROL_ROOT/GPU_MEDICAL_RECOVERY_COMPLETE" ]]; then
  if [[ -x "$PINNED_PYTHON" && -s "$REPO_ROOT/scripts/audit_massive_medical_union_medical_recovery_v1.py" ]] && \
     "$PINNED_PYTHON" "$REPO_ROOT/scripts/audit_massive_medical_union_medical_recovery_v1.py" audit-gpu >/dev/null 2>&1; then
    echo 'AWAITING_EXTERNAL_JUDGE: verified 240/240 normal stops; no API call has been made by the GPU workflow.'
  else
    echo 'RECOVERY_GPU_ARTIFACT_AUDIT_FAILED'
  fi
elif [[ -s "$CONTROL_ROOT/RELEASED" ]]; then
  echo 'MEDICAL_RECOVERY_RUNNING_OR_PENDING'
elif [[ -s "$CONTROL_ROOT/PREP.json" ]]; then
  echo 'MEDICAL_RECOVERY_STAGED_NOT_SUBMITTED'
else
  echo 'MEDICAL_RECOVERY_NOT_STAGED'
fi
