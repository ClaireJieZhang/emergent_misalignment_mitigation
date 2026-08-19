#!/bin/bash
# Read-only status for MASSIVE test-only evaluation recovery v2.

set -euo pipefail

TILLICUM_ROOT=${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}
REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate
ENV_ROOT=$TILLICUM_ROOT/envs/subliminal-mitigate-py311
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/massive_benefit_pilot_v1
LOGS_ROOT=$TILLICUM_ROOT/outputs/logs
CONTROL_ROOT=$OUTPUT_ROOT/control/evaluation_recovery_v2
LOCK_ROOT=$OUTPUT_ROOT/control/MASSIVE_EVALUATION_RECOVERY_V2_SUBMISSION_LOCK
ATTEMPT=$CONTROL_ROOT/dispatch_attempt.tsv
JOBS_FILE=$CONTROL_ROOT/jobs.tsv
ADDENDUM=$CONTROL_ROOT/AUTHORIZED_EVALUATION_RECOVERY_V2_WITHIN_ORIGINAL_CAP.json
EVAL_ROOT=$OUTPUT_ROOT/evaluation/evaluation_recovery_v2
AUDITOR=$REPO_ROOT/scripts/audit_massive_benefit_evaluation_recovery_v2.py

echo "Output root: $OUTPUT_ROOT"
echo 'Prior conservative usage: 157 rounded H200-minutes / $2.355.'
echo 'Recovery-v2 cap: 15 H200-minutes; cumulative maximum 172m / $2.580.'
echo 'One-minute termination contingency: 173m / $2.595; original ceiling 195m / $2.925.'
echo "Training, development rerun, reselection, further retry, extra adapter, union, quorum, and continuation: disabled."

echo
echo "=== Immutable prior jobs ==="
sacct -X --starttime 2026-08-17 \
  --jobs 237935,237936,237937,239578,239579,246311 \
  --format='JobID,JobName%28,State,Elapsed,ElapsedRaw,Timelimit,AllocTRES%42,ExitCode' \
  --units=G || true

echo
echo "=== Recovery-v2 control and result artifacts ==="
for record in \
  "$LOCK_ROOT/owner" \
  "$ATTEMPT" \
  "$JOBS_FILE" \
  "$ADDENDUM" \
  "$CONTROL_ROOT/SUBMITTED" \
  "$CONTROL_ROOT/RELEASED" \
  "$CONTROL_ROOT/EVALUATE_STARTED" \
  "$EVAL_ROOT/generations/massive_en_test__pi_base.json" \
  "$EVAL_ROOT/generations/massive_en_test__pi_base__intent_only.json" \
  "$EVAL_ROOT/generations/massive_en_test__step_30.json" \
  "$EVAL_ROOT/generations/massive_en_test__step_30__intent_only.json" \
  "$EVAL_ROOT/sealed_final/decoder_provenance_audit.json" \
  "$EVAL_ROOT/sealed_final/summary.json" \
  "$CONTROL_ROOT/GO_MASSIVE_BENEFIT_ONLY"; do
  if [[ -s "$record" ]]; then
    printf 'DONE     %s\n' "${record#$OUTPUT_ROOT/}"
  else
    printf 'PENDING  %s\n' "${record#$OUTPUT_ROOT/}"
  fi
done

if [[ -s "$ADDENDUM" && -s "$JOBS_FILE" ]]; then
  echo
  echo "=== Recovery-v2 seal audit ==="
  set +e
  PYTHONDONTWRITEBYTECODE=1 "$ENV_ROOT/bin/python" "$AUDITOR" verify-control \
    --repo-root "$REPO_ROOT" --output-root "$OUTPUT_ROOT" \
    --logs-root "$LOGS_ROOT" --jobs-file "$JOBS_FILE" \
    --addendum-file "$ADDENDUM"
  audit_status=$?
  set -e
  if (( audit_status != 0 )); then
    echo "EVALUATION_RECOVERY_V2_CONTROL_AUDIT_FAILED"
  fi
fi

job_record=$JOBS_FILE
if [[ ! -s "$job_record" && -s "$ATTEMPT" ]]; then
  job_record=$ATTEMPT
fi
evaluate_job_id=""
evaluate_state=""
evaluate_exit=""
if [[ -s "$job_record" ]]; then
  evaluate_job_id=$(awk -F'\t' '$1=="evaluate" {print $2; exit}' "$job_record")
fi
if [[ "$evaluate_job_id" =~ ^[0-9]+$ ]]; then
  echo
  echo "=== Recovery-v2 job ==="
  column -t -s $'\t' "$job_record" 2>/dev/null || cat "$job_record"
  echo
  squeue --jobs "$evaluate_job_id" \
    --format='%.18i %.28j %.2t %.10M %.10l %.4D %R' || true
  echo
  sacct -X --jobs "$evaluate_job_id" \
    --format='JobID,JobName%28,State,Elapsed,ElapsedRaw,Timelimit,AllocTRES%42,ExitCode' \
    --units=G || true
  IFS='|' read -r evaluate_state evaluate_exit < <(
    sacct -X -n -P --jobs "$evaluate_job_id" --format=State,ExitCode \
      | awk -F'|' 'NF >= 2 {print $1 "|" $2; exit}'
  ) || true
fi

if [[ -s "$EVAL_ROOT/sealed_final/summary.md" ]]; then
  echo
  echo "=== sealed_final/summary.md ==="
  cat "$EVAL_ROOT/sealed_final/summary.md"
fi

if compgen -G "$EVAL_ROOT/generations/failures/*.failure.json" >/dev/null; then
  echo
  echo "=== Recovery-v2 structured-generation failure evidence ==="
  for failure_file in "$EVAL_ROOT"/generations/failures/*.failure.json; do
    sha256sum "$failure_file"
    cat "$failure_file"
  done
fi

is_terminal_failure() {
  case "$1" in
    FAILED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*|CANCELLED*|PREEMPTED*|BOOT_FAIL*|DEADLINE*|REVOKED*) return 0 ;;
    *) return 1 ;;
  esac
}

echo
echo "=== Recovery-v2 decision ==="
if [[ -s "$CONTROL_ROOT/GO_MASSIVE_BENEFIT_ONLY" ]]; then
  echo "FINAL_EVALUATION_COMPLETE: GO_MASSIVE_BENEFIT_ONLY"
  echo "The benchmark-backed task-specific MASSIVE benefit passed; no union or quorum ran."
elif [[ -s "$CONTROL_ROOT/STOPPED_MASSIVE_FINAL" ]]; then
  echo "FINAL_EVALUATION_COMPLETE: STOPPED_MASSIVE_FINAL"
elif is_terminal_failure "$evaluate_state"; then
  echo "STOPPED_EVALUATION_RECOVERY_V2_JOB: job $evaluate_job_id state=$evaluate_state exit=$evaluate_exit"
elif [[ "$evaluate_state" == COMPLETED* ]]; then
  echo "STOPPED_EVALUATION_RECOVERY_V2_PROTOCOL: completed without a scientific terminal sentinel"
elif [[ -s "$CONTROL_ROOT/EVALUATE_STARTED" ]]; then
  echo "Fresh paired no-whitespace cleaned-test evaluation is pending or running."
elif [[ -s "$CONTROL_ROOT/RELEASED" ]]; then
  echo "Recovery-v2 job is released and pending or running."
elif [[ -s "$ATTEMPT" ]]; then
  echo "Dispatch incomplete; the recorded job must remain held."
elif [[ -s "$LOCK_ROOT/owner" ]]; then
  echo "Exact-once lock exists; dispatch did not reach a recorded job."
else
  echo "Test-only recovery v2 has not been submitted."
fi

echo
echo "Logs: $LOGS_ROOT/massive_benefit_evaluation_recovery_v2_*"
