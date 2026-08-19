#!/bin/bash
# Read-only status for the matched MASSIVE+medical union pilot.

set -uo pipefail

TILLICUM_ROOT=/gpfs/projects/stf/claizhan/subliminal-mitigate
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/massive_medical_union_pilot_v1
CONTROL_ROOT=$OUTPUT_ROOT/control
EVAL_ROOT=$OUTPUT_ROOT/evaluation/wave1
JOBS_FILE=$CONTROL_ROOT/wave1_jobs.tsv

echo '=== MASSIVE + medical union pilot ==='
echo 'Released ceiling (Wave 1 only): 80 H200-minutes / $1.20 GPU.'
echo 'External medical-judge ceiling: 240 calls / $0.50.'
echo 'Wave 2 and quorum: protocol only; not submitted.'

echo '=== Control records ==='
for name in \
  STAGING_IN_PROGRESS STAGED PREP_COMPLETE.json WAVE1_SUBMISSION_LOCK \
  WAVE1_SUBMITTED WAVE1_RELEASED TRAIN_COMPLETE WAVE1_GPU_EVAL_COMPLETE \
  WAVE1_EXTERNAL_JUDGE_LOCK AWAITING_EXTERNAL_JUDGE_RESUME \
  GO_MASSIVE_UNION_WAVE1 STOPPED_MASSIVE_UNION_WAVE1 STOPPED_submission \
  STOPPED_train_A STOPPED_train_B1 STOPPED_evaluate STOPPED_finalize; do
  if [[ -e "$CONTROL_ROOT/$name" ]]; then
    printf 'PRESENT %s\n' "$name"
  fi
done
for model in pi_A pi_B1; do
  if [[ -s "$OUTPUT_ROOT/models/$model/TRAIN_COMPLETE" ]]; then
    printf 'PRESENT %s/TRAIN_COMPLETE\n' "$model"
  fi
done

if [[ -s "$JOBS_FILE" ]]; then
  echo '=== Slurm jobs ==='
  while IFS=$'\t' read -r stage job_id max_minutes released; do
    [[ "$stage" == stage ]] && continue
    [[ "$job_id" =~ ^[0-9]+$ ]] || continue
    printf '%s job=%s cap=%sm released=%s\n' \
      "$stage" "$job_id" "$max_minutes" "$released"
    squeue -h -j "$job_id" -o 'queue state=%T reason=%R elapsed=%M limit=%l node=%N' 2>/dev/null || true
    sacct -n -X -j "$job_id" --format=JobIDRaw,State,Elapsed,Timelimit,AllocTRES,ExitCode \
      2>/dev/null | sed '/^[[:space:]]*$/d' || true
  done < "$JOBS_FILE"
fi

if [[ -s "$EVAL_ROOT/component_gate/summary.md" ]]; then
  echo '=== Sealed component summary ==='
  sed -n '1,220p' "$EVAL_ROOT/component_gate/summary.md"
elif [[ -s "$EVAL_ROOT/component_gate/summary.json" ]]; then
  echo '=== Component summary JSON ==='
  python -m json.tool "$EVAL_ROOT/component_gate/summary.json" 2>/dev/null || true
fi

echo '=== Decision ==='
if [[ -s "$CONTROL_ROOT/GO_MASSIVE_UNION_WAVE1" ]]; then
  echo 'FINAL_EVALUATION_COMPLETE: GO_MASSIVE_UNION_WAVE1'
  echo 'Wave 1 passed. Wave 2 remains unreleased and requires a separate explicit decision.'
elif [[ -s "$CONTROL_ROOT/STOPPED_MASSIVE_UNION_WAVE1" ]]; then
  echo 'FINAL_EVALUATION_COMPLETE: STOPPED_MASSIVE_UNION_WAVE1'
elif compgen -G "$CONTROL_ROOT/STOPPED_*" >/dev/null; then
  echo 'TERMINAL_INFRASTRUCTURE_STOP'
  for path in "$CONTROL_ROOT"/STOPPED_*; do
    echo "--- $(basename "$path")"
    sed -n '1,120p' "$path" 2>/dev/null || true
  done
elif [[ -s "$CONTROL_ROOT/AWAITING_EXTERNAL_JUDGE_RESUME" ]]; then
  echo 'AWAITING_EXTERNAL_JUDGE_RESUME: no automatic retry; no GPU retry authorized.'
  sed -n '1,120p' "$CONTROL_ROOT/AWAITING_EXTERNAL_JUDGE_RESUME"
elif [[ -s "$CONTROL_ROOT/WAVE1_GPU_EVAL_COMPLETE" ]]; then
  echo 'AWAITING_EXTERNAL_JUDGE: GPU generation/scoring is complete; bounded API judging has not finished.'
elif [[ -s "$OUTPUT_ROOT/models/pi_A/TRAIN_COMPLETE" && -s "$OUTPUT_ROOT/models/pi_B1/TRAIN_COMPLETE" ]]; then
  echo 'WAVE1_TRAINING_COMPLETE: paired GPU evaluation is pending/running.'
elif [[ -s "$CONTROL_ROOT/WAVE1_RELEASED" ]]; then
  echo 'WAVE1_RUNNING_OR_PENDING'
elif [[ -s "$CONTROL_ROOT/STAGED" ]]; then
  echo 'STAGED_NOT_SUBMITTED'
else
  echo 'NOT_STAGED'
fi
