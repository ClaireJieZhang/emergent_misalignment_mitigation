#!/bin/bash
# Exact-once dispatch for the final MASSIVE evaluation-only recovery.

set -euo pipefail
umask 077

if [[ $# -ne 3 || "$1" != recover-evaluation || \
      "$2" != --ack-original-max-cost-usd || "$3" != 2.925 ]]; then
  cat >&2 <<'EOF'
Usage:
  scripts/submit_massive_benefit_evaluation_recovery_v1_tillicum.sh \
    recover-evaluation --ack-original-max-cost-usd 2.925

Prior jobs conservatively account for 149 rounded H200-minutes.  This final
evaluation-only recovery may allocate one H200 for 15 minutes.  Its cumulative
request-bound maximum is 164 H200-minutes / $2.460.  Even one separately
rounded minute of Slurm termination overhead is 165 minutes / $2.475, within
the original 195-minute / $2.925 authorization.

This command authorizes no training, retry after this job, extra adapter,
medical union, quorum, reserve, or automatic continuation.
EOF
  exit 2
fi

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY

TILLICUM_ROOT=/gpfs/projects/stf/claizhan/subliminal-mitigate
REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate
ENV_ROOT=$TILLICUM_ROOT/envs/subliminal-mitigate-py311
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/massive_benefit_pilot_v1
LOGS_ROOT=$TILLICUM_ROOT/outputs/logs
CONTROL_PARENT=$OUTPUT_ROOT/control
CONTROL_ROOT=$CONTROL_PARENT/evaluation_recovery_v1
LOCK_ROOT=$CONTROL_PARENT/MASSIVE_EVALUATION_RECOVERY_V1_SUBMISSION_LOCK
ATTEMPT=$CONTROL_ROOT/dispatch_attempt.tsv
JOBS_FILE=$CONTROL_ROOT/jobs.tsv
ADDENDUM=$CONTROL_ROOT/AUTHORIZED_EVALUATION_RECOVERY_WITHIN_ORIGINAL_CAP.json
SUBMITTED=$CONTROL_ROOT/SUBMITTED
RELEASED=$CONTROL_ROOT/RELEASED
EVAL_ROOT=$OUTPUT_ROOT/evaluation/evaluation_recovery_v1
AUDITOR=scripts/audit_massive_benefit_evaluation_recovery_v1.py
JOB_SCRIPT=scripts/sbatch_massive_benefit_evaluation_recovery_v1_tillicum_h200.sbatch

cd "$REPO_ROOT"
test -f "$ENV_ROOT/.ready"
test -z "$(git status --porcelain --untracked-files=all)" || {
  echo "Refusing evaluation recovery from a dirty Tillicum checkout." >&2
  git status --short >&2
  exit 3
}
test ! -e "$LOCK_ROOT"
test ! -e "$CONTROL_ROOT"
test ! -e "$EVAL_ROOT"
if compgen -G "$LOGS_ROOT/massive_benefit_evaluation_recovery_v1_*" >/dev/null; then
  echo "Evaluation-recovery log names already exist; refusing duplicate allocation." >&2
  exit 3
fi

export PYTHONDONTWRITEBYTECODE=1
"$ENV_ROOT/bin/python" "$AUDITOR" verify-preflight \
  --repo-root "$REPO_ROOT" --output-root "$OUTPUT_ROOT" --logs-root "$LOGS_ROOT"

echo "=== Slurm admission preflight (no job submitted) ==="
sbatch --test-only --export=NONE --no-requeue --nodes=1 --ntasks=1 \
  --gres=gpu:h200:1 --time=00:15:00 "$JOB_SCRIPT"

# This lock is permanent.  Any dispatch outcome consumes the sole authority.
if ! mkdir "$LOCK_ROOT" 2>/dev/null; then
  echo "Evaluation-recovery lock already exists; refusing duplicate allocation." >&2
  exit 3
fi
printf 'recovery_id=%s\ncreated_at=%s\nrepair_repo_commit=%s\nexact_once=true\n' \
  massive_benefit_evaluation_recovery_v1 "$(date --iso-8601=seconds)" \
  "$(git rev-parse HEAD)" > "$LOCK_ROOT/owner"
chmod 0400 "$LOCK_ROOT/owner"
mkdir "$CONTROL_ROOT"

printf 'stage\tjob_id\tmax_minutes\n' > "$ATTEMPT"
chmod 0600 "$ATTEMPT"
released=false
dispatch_failure() {
  local status=$?
  if (( status != 0 )) && [[ "$released" != true ]]; then
    echo "Evaluation-recovery dispatch stopped before release." >&2
    echo "The permanent lock remains; any recorded job must remain held." >&2
    cat "$ATTEMPT" >&2 || true
  fi
  exit "$status"
}
trap dispatch_failure EXIT

evaluate_job=$(sbatch --parsable --hold --export=NONE --no-requeue \
  --nodes=1 --ntasks=1 --gres=gpu:h200:1 --time=00:15:00 "$JOB_SCRIPT")
evaluate_job=${evaluate_job%%;*}
[[ "$evaluate_job" =~ ^[0-9]+$ ]]
printf 'evaluate\t%s\t15\n' "$evaluate_job" >> "$ATTEMPT"

record=$(scontrol show job "$evaluate_job" -o | tr ' ' '\n')
test "$(awk -F= '$1=="JobState" {print $2; exit}' <<< "$record")" = PENDING
test "$(awk -F= '$1=="Reason" {print $2; exit}' <<< "$record")" = JobHeldUser
test "$(awk -F= '$1=="Requeue" {print $2; exit}' <<< "$record")" = 0
test "$(awk -F= '$1=="Account" {print $2; exit}' <<< "$record")" = stf
test "$(awk -F= '$1=="Partition" {print $2; exit}' <<< "$record")" = gpu-h200
test "$(awk -F= '$1=="TimeLimit" {print $2; exit}' <<< "$record")" = 00:15:00
node_range=$(awk -F= '$1=="NumNodes" {print $2; exit}' <<< "$record")
[[ "$node_range" = 1 || "$node_range" = 1-1 ]]
test "$(awk -F= '$1=="NumTasks" {print $2; exit}' <<< "$record")" = 1
requested_tres=$(sed -n 's/^ReqTRES=//p' <<< "$record")
test "$(tr ',' '\n' <<< "$requested_tres" | awk -F= '$1=="gres/gpu:h200" {print $2}')" = 1
test "$(tr ',' '\n' <<< "$requested_tres" | awk -F= '$1=="gres/gpu" {print $2}')" = 1
test "$(tr ',' '\n' <<< "$requested_tres" | awk -F= '$1=="node" {print $2}')" = 1

jobs_build=$CONTROL_ROOT/.jobs-$$
printf 'stage\tjob_id\tmax_minutes\nevaluate\t%s\t15\n' "$evaluate_job" \
  > "$jobs_build"
chmod 0400 "$jobs_build"
mv "$jobs_build" "$JOBS_FILE"

common_control=(
  --repo-root "$REPO_ROOT"
  --output-root "$OUTPUT_ROOT"
  --logs-root "$LOGS_ROOT"
  --jobs-file "$JOBS_FILE"
)
"$ENV_ROOT/bin/python" "$AUDITOR" write-addendum \
  "${common_control[@]}" --output-file "$ADDENDUM"
"$ENV_ROOT/bin/python" "$AUDITOR" verify-control \
  "${common_control[@]}" --addendum-file "$ADDENDUM"

submitted_build=$CONTROL_ROOT/.submitted-$$
printf 'recovery_id=%s\nrepair_repo_commit=%s\naddendum_sha256=%s\njobs_sha256=%s\nevaluate_job_id=%s\nsubmitted_at=%s\n' \
  massive_benefit_evaluation_recovery_v1 "$(git rev-parse HEAD)" \
  "$(sha256sum "$ADDENDUM" | awk '{print $1}')" \
  "$(sha256sum "$JOBS_FILE" | awk '{print $1}')" "$evaluate_job" \
  "$(date --iso-8601=seconds)" > "$submitted_build"
printf 'dispatch_sha256=%s\n' "$(sha256sum "$submitted_build" | awk '{print $1}')" \
  >> "$submitted_build"
chmod 0400 "$submitted_build"
mv "$submitted_build" "$SUBMITTED"

scontrol release "$evaluate_job"
released=true
released_build=$CONTROL_ROOT/.released-$$
printf 'recovery_id=%s\nevaluate_job_id=%s\nreleased_at=%s\n' \
  massive_benefit_evaluation_recovery_v1 "$evaluate_job" \
  "$(date --iso-8601=seconds)" > "$released_build"
printf 'release_sha256=%s\n' "$(sha256sum "$released_build" | awk '{print $1}')" \
  >> "$released_build"
chmod 0400 "$released_build"
mv "$released_build" "$RELEASED"
trap - EXIT

echo "Submitted final MASSIVE evaluation-only recovery: evaluate=$evaluate_job"
echo 'Cumulative request-bound maximum: 164 H200-minutes / $2.460.'
echo "No training, further retry, extra adapter, union, quorum, or continuation is authorized."
