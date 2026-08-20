#!/bin/bash
# Read-only Wave-2 status. Never submits, releases, judges, or edits artifacts.

set -uo pipefail

TILLICUM_ROOT=/gpfs/projects/stf/claizhan/subliminal-mitigate
REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate-mmu-wave2
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/massive_medical_union_pilot_v1
CONTROL_ROOT=$OUTPUT_ROOT/control/wave2
EVAL_ROOT=$OUTPUT_ROOT/evaluation/wave2
JOBS_FILE=$CONTROL_ROOT/jobs.tsv
PINNED_PYTHON=$TILLICUM_ROOT/envs/subliminal-mitigate-py311/bin/python

echo '=== MASSIVE + medical union Wave 2 ==='
echo 'GPU ceiling: B2 30m + B3 30m + evaluation 15m = 75 H200-minutes / $1.125.'
echo 'New external judge ceiling: 160 calls / $0.50.'
echo 'Cumulative released ceiling including prior recovery: $3.875 all-in.'
echo 'Wave 3: prospectively frozen, but never automatically submitted or released.'

echo '=== Control records ==='
for name in PREP.json SUBMISSION_LOCK SUBMISSION_ATTEMPT.tsv jobs.tsv \
  AUTHORIZED_MAX_COST_USD_1.125.json SUBMITTED RELEASED STOPPED_submission \
  STOPPED_train_B2 STOPPED_train_B3 STOPPED_evaluate \
  WAVE2_GPU_EVAL_COMPLETE STOPPED_WAVE2_MASSIVE_PREJUDGE \
  EXTERNAL_JUDGE_LOCK AWAITING_EXTERNAL_JUDGE_RESUME STOPPED_finalize \
  WAVE2_FINAL_DECISION.json GO_MASSIVE_UNION_ALL_REPLICAS \
  STOPPED_MASSIVE_UNION_ALL_REPLICAS; do
  [[ -e "$CONTROL_ROOT/$name" ]] && printf 'PRESENT %s\n' "$name"
done

queue_any=false
terminal_failure=false
terminal_unsealed=false
if [[ -s "$JOBS_FILE" ]]; then
  echo '=== Slurm jobs ==='
  while IFS=$'\t' read -r stage job_id max_minutes released; do
    [[ "$stage" == stage ]] && continue
    [[ "$job_id" =~ ^[0-9]+$ ]] || continue
    printf '%s job=%s cap=%sm released=%s\n' "$stage" "$job_id" "$max_minutes" "$released"
    queue_line=$(squeue -h -j "$job_id" -o 'queue state=%T reason=%R elapsed=%M limit=%l node=%N' 2>/dev/null || true)
    if [[ -n "$queue_line" ]]; then
      queue_any=true
      printf '%s\n' "$queue_line"
    fi
    accounting_line=$(sacct -n -X -P -j "$job_id" \
      --format=JobIDRaw,State%30,Elapsed,Timelimit,AllocTRES,ExitCode 2>/dev/null | \
      awk -F'|' -v wanted="$job_id" '$1 == wanted {print; exit}' || true)
    if [[ -n "$accounting_line" ]]; then
      printf '%s\n' "$accounting_line"
      IFS='|' read -r _ state _ _ _ exit_code <<< "$accounting_line"
      if [[ "$state" =~ ^(FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED|SPECIAL_EXIT) ]]; then
        terminal_failure=true
      elif [[ "$state" == COMPLETED* ]]; then
        case "$stage" in
          train_B2) [[ -s "$TILLICUM_ROOT/outputs/massive_medical_union_pilot_v1/models/pi_B2/TRAIN_COMPLETE" ]] || terminal_unsealed=true ;;
          train_B3) [[ -s "$TILLICUM_ROOT/outputs/massive_medical_union_pilot_v1/models/pi_B3/TRAIN_COMPLETE" ]] || terminal_unsealed=true ;;
          evaluate) [[ -s "$CONTROL_ROOT/WAVE2_GPU_EVAL_COMPLETE" ]] || terminal_unsealed=true ;;
        esac
      fi
    fi
  done < "$JOBS_FILE"
fi

if [[ -s "$EVAL_ROOT/prejudge_component_gate/awaiting_external_judge.md" ]]; then
  echo '=== Fresh cleaned-test MASSIVE prejudge ==='
  sed -n '1,180p' "$EVAL_ROOT/prejudge_component_gate/awaiting_external_judge.md"
elif [[ -s "$EVAL_ROOT/prejudge_component_gate/summary.md" ]]; then
  echo '=== Fresh cleaned-test MASSIVE prejudge ==='
  sed -n '1,180p' "$EVAL_ROOT/prejudge_component_gate/summary.md"
fi
if [[ -s "$EVAL_ROOT/component_gate/summary.md" ]]; then
  echo '=== Final all-replica component gate ==='
  sed -n '1,220p' "$EVAL_ROOT/component_gate/summary.md"
fi

echo '=== Decision ==='
if [[ -s "$CONTROL_ROOT/WAVE2_FINAL_DECISION.json" ]]; then
  if "$PINNED_PYTHON" "$REPO_ROOT/scripts/audit_massive_medical_union_wave2.py" audit-final-decision >/dev/null 2>&1; then
    if [[ -s "$CONTROL_ROOT/GO_MASSIVE_UNION_ALL_REPLICAS" ]]; then
      echo 'WAVE2_FINAL_COMPLETE: all four direct components qualified.'
      echo 'Wave 3 is eligible but remains unsubmitted and unreleased.'
    elif [[ -s "$CONTROL_ROOT/STOPPED_MASSIVE_UNION_ALL_REPLICAS" ]]; then
      echo 'WAVE2_FINAL_SCIENTIFIC_STOP: at least one direct component gate failed.'
    else
      echo 'WAVE2_FINAL_CONTROL_SENTINEL_MISSING'
    fi
  else
    echo 'WAVE2_FINAL_DECISION_AUDIT_FAILED'
  fi
elif [[ -s "$CONTROL_ROOT/STOPPED_WAVE2_MASSIVE_PREJUDGE" ]]; then
  echo 'WAVE2_MASSIVE_PREJUDGE_STOP: zero Wave-2 judge calls were authorized.'
elif [[ -s "$CONTROL_ROOT/STOPPED_submission" ]]; then
  echo 'WAVE2_TERMINAL_SUBMISSION_STOP: no retry is authorized.'
  sed -n '1,100p' "$CONTROL_ROOT/STOPPED_submission"
elif [[ -s "$CONTROL_ROOT/STOPPED_train_B2" || -s "$CONTROL_ROOT/STOPPED_train_B3" || -s "$CONTROL_ROOT/STOPPED_evaluate" ]]; then
  echo 'WAVE2_TERMINAL_STAGE_STOP: no replacement seed or GPU retry is authorized.'
elif [[ -s "$CONTROL_ROOT/STOPPED_finalize" ]]; then
  echo 'WAVE2_TERMINAL_FINALIZE_STOP'
  sed -n '1,100p' "$CONTROL_ROOT/STOPPED_finalize"
elif [[ -s "$CONTROL_ROOT/AWAITING_EXTERNAL_JUDGE_RESUME" ]]; then
  echo 'WAVE2_AWAITING_EXPLICIT_JUDGE_RESUME: no automatic retry.'
  sed -n '1,100p' "$CONTROL_ROOT/AWAITING_EXTERNAL_JUDGE_RESUME"
elif [[ -s "$CONTROL_ROOT/WAVE2_GPU_EVAL_COMPLETE" ]]; then
  if "$PINNED_PYTHON" "$REPO_ROOT/scripts/audit_massive_medical_union_wave2.py" audit-gpu >/dev/null 2>&1; then
    echo 'WAVE2_AWAITING_EXTERNAL_JUDGE: MASSIVE passed; 160 B2/B3 judgments pending.'
  else
    echo 'WAVE2_GPU_ARTIFACT_AUDIT_FAILED'
  fi
elif [[ "$terminal_failure" == true ]]; then
  echo 'WAVE2_TERMINAL_UNSEALED_FAILURE: inspect immutable stage logs; no retry is authorized.'
elif [[ "$terminal_unsealed" == true ]]; then
  echo 'WAVE2_TERMINAL_UNSEALED_COMPLETION: a job completed without its required seal.'
elif [[ "$queue_any" == true ]]; then
  echo 'WAVE2_RUNNING_OR_PENDING'
elif [[ -s "$CONTROL_ROOT/PREP.json" && ! -e "$CONTROL_ROOT/SUBMITTED" ]]; then
  echo 'WAVE2_STAGED_NOT_SUBMITTED'
elif [[ -s "$CONTROL_ROOT/PREP.json" ]]; then
  echo 'WAVE2_SCHEDULER_STATE_UNRESOLVED'
else
  echo 'WAVE2_NOT_STAGED'
fi
