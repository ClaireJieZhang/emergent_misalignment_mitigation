#!/bin/bash
# Read-only status for Wave-2 evaluation recovery v1.

set -uo pipefail

TILLICUM_ROOT=/gpfs/projects/stf/claizhan/subliminal-mitigate
REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate-mmu-wave2-eval-recovery-v1
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/massive_medical_union_pilot_v1
ORIGINAL_CONTROL=$OUTPUT_ROOT/control/wave2
CONTROL_ROOT=$OUTPUT_ROOT/control/wave2_eval_recovery_v1
EVAL_ROOT=$OUTPUT_ROOT/evaluation/wave2_eval_recovery_v1
JOBS_FILE=$CONTROL_ROOT/jobs.tsv
PYTHON=$TILLICUM_ROOT/envs/subliminal-mitigate-py311/bin/python
AUDITOR=$REPO_ROOT/scripts/audit_massive_medical_union_wave2_evaluation_recovery_v1.py

echo '=== Wave-2 evaluation-only recovery v1 ==='
echo 'Original B2/B3 jobs reached checkpoint 540, then failed only in CPU manifest sealing.'
echo 'New ceiling: one 15m H200 evaluation / $0.225. Retraining: none.'
echo 'Optional later judge ceiling: exactly 160 calls / $0.50. Wave 3: never automatic.'

echo '=== Preserved original incident ==='
for name in STOPPED_train_B2 STOPPED_train_B3; do
  [[ -s "$ORIGINAL_CONTROL/$name" ]] && echo "PRESENT $name"
done
sacct -n -X -P -j 251235,251236,251237 \
  --format=JobIDRaw,JobName,State%20,Elapsed,Timelimit,ExitCode 2>/dev/null || true

echo '=== Recovery control ==='
for name in PREP.json STAGED SUBMISSION_LOCK SUBMISSION_ATTEMPT.tsv jobs.tsv \
  AUTHORIZED_MAX_COST_USD_0.225.json SUBMITTED RELEASED \
  STOPPED_submission STOPPED_evaluate_recovery_v1 GPU_EVAL_RECOVERY_COMPLETE \
  STOPPED_WAVE2_MASSIVE_PREJUDGE EXTERNAL_JUDGE_LOCK \
  AWAITING_EXTERNAL_JUDGE_RESUME STOPPED_finalize WAVE2_FINAL_DECISION.json \
  GO_MASSIVE_UNION_ALL_REPLICAS STOPPED_MASSIVE_UNION_ALL_REPLICAS; do
  [[ -e "$CONTROL_ROOT/$name" ]] && printf 'PRESENT %s\n' "$name"
done

job_id=
job_state=
job_exit=
if [[ -s "$JOBS_FILE" ]]; then
  job_id=$(awk -F '\t' 'NR == 2 {print $2}' "$JOBS_FILE")
  if [[ "$job_id" =~ ^[0-9]+$ ]]; then
    echo '=== Recovery Slurm job ==='
    squeue -h -j "$job_id" -o 'queue state=%T reason=%R elapsed=%M limit=%l node=%N' 2>/dev/null || true
    accounting=$(sacct -n -X -P -j "$job_id" \
      --format=JobIDRaw,State%30,Elapsed,Timelimit,AllocTRES,ExitCode 2>/dev/null | \
      awk -F'|' -v wanted="$job_id" '$1 == wanted {print; exit}' || true)
    [[ -n "$accounting" ]] && echo "$accounting"
    IFS='|' read -r _ job_state _ _ _ job_exit <<< "$accounting"
  fi
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
  if PYTHONDONTWRITEBYTECODE=1 "$PYTHON" "$AUDITOR" audit-final-decision >/dev/null 2>&1; then
    if [[ -s "$CONTROL_ROOT/GO_MASSIVE_UNION_ALL_REPLICAS" ]]; then
      echo 'WAVE2_RECOVERY_FINAL_COMPLETE: all four direct components qualified.'
      echo 'Wave 3 is eligible but remains unsubmitted and unreleased.'
    elif [[ -s "$CONTROL_ROOT/STOPPED_MASSIVE_UNION_ALL_REPLICAS" ]]; then
      echo 'WAVE2_RECOVERY_FINAL_SCIENTIFIC_STOP: at least one component gate failed.'
    else
      echo 'WAVE2_RECOVERY_FINAL_CONTROL_SENTINEL_MISSING'
    fi
  else
    echo 'WAVE2_RECOVERY_FINAL_DECISION_AUDIT_FAILED'
  fi
elif [[ -s "$CONTROL_ROOT/STOPPED_WAVE2_MASSIVE_PREJUDGE" ]]; then
  echo 'WAVE2_RECOVERY_MASSIVE_PREJUDGE_STOP: zero judge calls were authorized.'
elif [[ -s "$CONTROL_ROOT/STOPPED_submission" ]]; then
  echo 'WAVE2_RECOVERY_SUBMISSION_STOP: no retry is authorized.'
  sed -n '1,100p' "$CONTROL_ROOT/STOPPED_submission"
elif [[ -s "$CONTROL_ROOT/STOPPED_evaluate_recovery_v1" ]]; then
  echo 'WAVE2_RECOVERY_EVALUATION_STOP: no retry is authorized.'
  sed -n '1,100p' "$CONTROL_ROOT/STOPPED_evaluate_recovery_v1"
elif [[ -s "$CONTROL_ROOT/STOPPED_finalize" ]]; then
  echo 'WAVE2_RECOVERY_FINALIZE_STOP'
  sed -n '1,100p' "$CONTROL_ROOT/STOPPED_finalize"
elif [[ -s "$CONTROL_ROOT/AWAITING_EXTERNAL_JUDGE_RESUME" ]]; then
  echo 'WAVE2_RECOVERY_AWAITING_EXPLICIT_JUDGE_RESUME: no automatic retry.'
  sed -n '1,100p' "$CONTROL_ROOT/AWAITING_EXTERNAL_JUDGE_RESUME"
elif [[ -s "$CONTROL_ROOT/GPU_EVAL_RECOVERY_COMPLETE" ]]; then
  if PYTHONDONTWRITEBYTECODE=1 "$PYTHON" "$AUDITOR" audit-gpu >/dev/null 2>&1; then
    echo 'WAVE2_RECOVERY_AWAITING_EXTERNAL_JUDGE: MASSIVE passed; 160 judgments pending.'
  else
    echo 'WAVE2_RECOVERY_GPU_ARTIFACT_AUDIT_FAILED'
  fi
elif [[ "$job_state" =~ ^(FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED|SPECIAL_EXIT) ]]; then
  echo "WAVE2_RECOVERY_TERMINAL_JOB_FAILURE: job=$job_id state=$job_state exit=$job_exit"
elif [[ "$job_state" == COMPLETED* ]]; then
  echo 'WAVE2_RECOVERY_TERMINAL_UNSEALED_COMPLETION'
elif [[ -n "$job_id" ]]; then
  echo 'WAVE2_RECOVERY_RUNNING_OR_PENDING'
elif [[ -s "$CONTROL_ROOT/STAGED" ]]; then
  echo 'WAVE2_RECOVERY_STAGED_NOT_SUBMITTED'
else
  echo 'WAVE2_RECOVERY_NOT_STAGED'
fi
