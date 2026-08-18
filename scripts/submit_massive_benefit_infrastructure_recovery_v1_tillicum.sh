#!/bin/bash
# One-time held-first recovery for the MASSIVE offline model-loader failure.

set -euo pipefail
umask 077

if [[ $# -ne 3 || "$1" != recover || "$2" != --ack-original-max-cost-usd || "$3" != 2.925 ]]; then
  cat >&2 <<'EOF'
Usage:
  scripts/submit_massive_benefit_infrastructure_recovery_v1_tillicum.sh \
    recover --ack-original-max-cost-usd 2.925

The completed/failed/cancelled original jobs conservatively account for
2 + 1 + 0 = 3 rounded H200-minutes.  This one-time recovery may allocate
90 training minutes and 75 evaluation minutes.  Its cumulative hard maximum
is 168 H200-minutes / $2.520, within the original 195-minute / $2.925 cap.
It does not rerun base development and does not authorize another retry.
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
CONTROL_ROOT=$OUTPUT_ROOT/control
RECOVERY_ROOT=$CONTROL_ROOT/infrastructure_recovery_v1
RECOVERY_LOCK=$CONTROL_ROOT/INFRASTRUCTURE_RECOVERY_V1_SUBMISSION_LOCK
RECOVERY_JOBS=$RECOVERY_ROOT/jobs.tsv
RECOVERY_ATTEMPT=$RECOVERY_ROOT/dispatch_attempt.tsv
RECOVERY_ADDENDUM=$RECOVERY_ROOT/AUTHORIZED_INFRASTRUCTURE_RECOVERY_WITHIN_ORIGINAL_CAP.json
RECOVERY_SUBMITTED=$RECOVERY_ROOT/SUBMITTED
RECOVERY_RELEASED=$RECOVERY_ROOT/RELEASED
RECOVERY_MODEL_DIR=$OUTPUT_ROOT/model/massive_en_benefit_pilot_infrastructure_recovery_v1
RECOVERY_EVAL_ROOT=$OUTPUT_ROOT/evaluation/infrastructure_recovery_v1
LOCAL_MODEL_PATH=$TILLICUM_ROOT/cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75
AUDITOR=scripts/audit_massive_benefit_infrastructure_recovery_v1.py
TRAIN_SCRIPT=scripts/sbatch_massive_benefit_infrastructure_recovery_v1_train_tillicum_h200.sbatch
EVALUATE_SCRIPT=scripts/sbatch_massive_benefit_infrastructure_recovery_v1_evaluate_tillicum_h200.sbatch

cd "$REPO_ROOT"
test -f "$ENV_ROOT/.ready"
test -z "$(git status --porcelain --untracked-files=all)" || {
  echo "Refusing recovery from a dirty Tillicum checkout." >&2
  git status --short >&2
  exit 3
}
test ! -e "$RECOVERY_LOCK"
test ! -e "$RECOVERY_ROOT"
test ! -e "$RECOVERY_MODEL_DIR"
test ! -e "$RECOVERY_EVAL_ROOT"
if compgen -G "$LOGS_ROOT/massive_benefit_infrastructure_recovery_v1_*" >/dev/null; then
  echo "Recovery log names already exist; refusing duplicate allocation." >&2
  exit 3
fi

common_preflight=(
  --repo-root "$REPO_ROOT"
  --tillicum-root "$TILLICUM_ROOT"
  --output-root "$OUTPUT_ROOT"
  --logs-root "$LOGS_ROOT"
  --local-model-path "$LOCAL_MODEL_PATH"
)
export PYTHONDONTWRITEBYTECODE=1
"$ENV_ROOT/bin/python" "$AUDITOR" verify-preflight "${common_preflight[@]}"

echo "=== Slurm admission preflight (no jobs submitted) ==="
sbatch --test-only --export=NONE --no-requeue --nodes=1 --ntasks=1 \
  --gres=gpu:h200:1 --time=01:30:00 "$TRAIN_SCRIPT"
sbatch --test-only --export=NONE --no-requeue --nodes=1 --ntasks=1 \
  --gres=gpu:h200:1 --time=01:15:00 "$EVALUATE_SCRIPT"

# This lock is intentionally durable.  Success and failure both consume the
# sole authorization to dispatch this recovery; there is no lock-removal path.
if ! mkdir "$RECOVERY_LOCK" 2>/dev/null; then
  echo "Recovery lock already exists; refusing duplicate allocations." >&2
  exit 3
fi
printf 'recovery_id=massive_benefit_infrastructure_recovery_v1\ncreated_at=%s\nrepair_repo_commit=%s\nexact_once=true\n' \
  "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)" > "$RECOVERY_LOCK/owner"
chmod 0400 "$RECOVERY_LOCK/owner"
mkdir "$RECOVERY_ROOT"

printf 'stage\tjob_id\tmax_minutes\n' > "$RECOVERY_ATTEMPT"
chmod 0600 "$RECOVERY_ATTEMPT"

released=false
dispatch_failure() {
  local status=$?
  if (( status != 0 )) && [[ "$released" != true ]]; then
    echo "Recovery dispatch stopped before full release." >&2
    echo "The durable exact-once lock remains, and any recorded jobs must remain held." >&2
    cat "$RECOVERY_ATTEMPT" >&2 || true
  fi
  exit "$status"
}
trap dispatch_failure EXIT

train_job=$(sbatch --parsable --hold --export=NONE --no-requeue \
  --nodes=1 --ntasks=1 --gres=gpu:h200:1 --time=01:30:00 "$TRAIN_SCRIPT")
train_job=${train_job%%;*}
[[ "$train_job" =~ ^[0-9]+$ ]]
printf 'train\t%s\t90\n' "$train_job" >> "$RECOVERY_ATTEMPT"

evaluate_job=$(sbatch --parsable --hold --export=NONE --no-requeue \
  --nodes=1 --ntasks=1 --gres=gpu:h200:1 --time=01:15:00 \
  --kill-on-invalid-dep=yes --dependency="afterok:$train_job" "$EVALUATE_SCRIPT")
evaluate_job=${evaluate_job%%;*}
[[ "$evaluate_job" =~ ^[0-9]+$ ]]
printf 'evaluate\t%s\t75\n' "$evaluate_job" >> "$RECOVERY_ATTEMPT"

audit_held_job() {
  local stage=$1 job_id=$2 expected_limit=$3 expected_dependency=$4
  local record reason node_range requested_tres dependency
  record=$(scontrol show job "$job_id" -o | tr ' ' '\n')
  test "$(awk -F= '$1=="JobState" {print $2; exit}' <<< "$record")" = PENDING
  reason=$(awk -F= '$1=="Reason" {print $2; exit}' <<< "$record")
  test "$reason" = JobHeldUser
  test "$(awk -F= '$1=="Requeue" {print $2; exit}' <<< "$record")" = 0
  test "$(awk -F= '$1=="Account" {print $2; exit}' <<< "$record")" = stf
  test "$(awk -F= '$1=="Partition" {print $2; exit}' <<< "$record")" = gpu-h200
  test "$(awk -F= '$1=="TimeLimit" {print $2; exit}' <<< "$record")" = "$expected_limit"
  node_range=$(awk -F= '$1=="NumNodes" {print $2; exit}' <<< "$record")
  [[ "$node_range" = 1 || "$node_range" = 1-1 ]]
  test "$(awk -F= '$1=="NumTasks" {print $2; exit}' <<< "$record")" = 1
  requested_tres=$(sed -n 's/^ReqTRES=//p' <<< "$record")
  test "$(tr ',' '\n' <<< "$requested_tres" | awk -F= '$1=="gres/gpu:h200" {print $2}')" = 1
  test "$(tr ',' '\n' <<< "$requested_tres" | awk -F= '$1=="gres/gpu" {print $2}')" = 1
  test "$(tr ',' '\n' <<< "$requested_tres" | awk -F= '$1=="node" {print $2}')" = 1
  if [[ -n "$expected_dependency" ]]; then
    dependency=$(awk -F= '$1=="Dependency" {print $2; exit}' <<< "$record")
    [[ "$dependency" = "afterok:$expected_dependency"* ]]
  fi
  echo "Audited held recovery $stage job $job_id"
}
audit_held_job train "$train_job" 01:30:00 ""
audit_held_job evaluate "$evaluate_job" 01:15:00 "$train_job"

jobs_build=$RECOVERY_ROOT/.jobs-$$
printf 'stage\tjob_id\tmax_minutes\ntrain\t%s\t90\nevaluate\t%s\t75\n' \
  "$train_job" "$evaluate_job" > "$jobs_build"
chmod 0400 "$jobs_build"
mv "$jobs_build" "$RECOVERY_JOBS"

common_control=(
  "${common_preflight[@]}"
  --jobs-file "$RECOVERY_JOBS"
)
"$ENV_ROOT/bin/python" "$AUDITOR" write-addendum \
  "${common_control[@]}" --output-file "$RECOVERY_ADDENDUM"
"$ENV_ROOT/bin/python" "$AUDITOR" verify-control \
  "${common_control[@]}" --addendum-file "$RECOVERY_ADDENDUM"

submitted_build=$RECOVERY_ROOT/.submitted-$$
printf 'recovery_id=massive_benefit_infrastructure_recovery_v1\nrepair_repo_commit=%s\naddendum_sha256=%s\njobs_sha256=%s\ntrain_job_id=%s\nevaluate_job_id=%s\nsubmitted_at=%s\n' \
  "$(git rev-parse HEAD)" "$(sha256sum "$RECOVERY_ADDENDUM" | awk '{print $1}')" \
  "$(sha256sum "$RECOVERY_JOBS" | awk '{print $1}')" "$train_job" \
  "$evaluate_job" "$(date --iso-8601=seconds)" > "$submitted_build"
printf 'dispatch_sha256=%s\n' "$(sha256sum "$submitted_build" | awk '{print $1}')" \
  >> "$submitted_build"
chmod 0400 "$submitted_build"
mv "$submitted_build" "$RECOVERY_SUBMITTED"

# Release downstream first; afterok keeps evaluation blocked behind training.
scontrol release "$evaluate_job"
scontrol release "$train_job"
released=true

released_build=$RECOVERY_ROOT/.released-$$
printf 'recovery_id=massive_benefit_infrastructure_recovery_v1\ntrain_job_id=%s\nevaluate_job_id=%s\nrelease_order=evaluate,train\nreleased_at=%s\n' \
  "$train_job" "$evaluate_job" "$(date --iso-8601=seconds)" > "$released_build"
printf 'release_sha256=%s\n' "$(sha256sum "$released_build" | awk '{print $1}')" \
  >> "$released_build"
chmod 0400 "$released_build"
mv "$released_build" "$RECOVERY_RELEASED"
trap - EXIT

echo "Submitted one-time MASSIVE infrastructure recovery: train=$train_job evaluate=$evaluate_job"
echo 'Base job 237935 and its frozen score were reused; no base job was submitted.'
echo 'Cumulative hard maximum: 168 H200-minutes / $2.520 within the original $2.925 cap.'
echo 'No further retry, reserve, union, extra adapter, quorum, or continuation is authorized.'
