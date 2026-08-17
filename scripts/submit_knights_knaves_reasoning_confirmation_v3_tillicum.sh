#!/bin/bash
# Submit one held-first, no-requeue, 30-minute K&K v3 evaluation-only job.

set -euo pipefail
umask 077

if [[ "$#" -ne 3 || "$1" != confirmation || "$2" != --ack-max-cost-usd || "$3" != 0.45 ]]; then
  cat >&2 <<'EOF'
Usage:
  scripts/submit_knights_knaves_reasoning_confirmation_v3_tillicum.sh confirmation --ack-max-cost-usd 0.45

This submits exactly one no-requeue, non-array H200 job capped at 30 minutes.
Maximum new cost is $0.45. Cumulative K&K released maximum becomes 210
H200-minutes ($3.15), below the existing immutable 240-minute/$3.60 ceiling.
It cannot train, select another checkpoint, selectively rerun a V2 answer,
create medical unions, or run quorum.
EOF
  exit 2
fi

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY

ROOT=/gpfs/projects/stf/claizhan/subliminal-mitigate
REPO=$ROOT/projects/subliminal-mitigate
ENV_ROOT=$ROOT/envs/subliminal-mitigate-py311
V1_ROOT=$ROOT/outputs/knights_knaves_reasoning_pilot_v1
V2_ROOT=$ROOT/outputs/knights_knaves_reasoning_confirmation_v2
V3_ROOT=$ROOT/outputs/knights_knaves_reasoning_confirmation_v3
V3_DATA=$V3_ROOT/data
CONTROL=$V3_ROOT/control
PREP=$CONTROL/PREP_COMPLETE
AUTH=$CONTROL/AUTHORIZED_MAX_COST_USD_0.45
JOBS=$CONTROL/jobs.tsv
ATTEMPT=$CONTROL/dispatch_attempt.tsv
SUBMITTED=$CONTROL/SUBMITTED
RELEASED=$CONTROL/RELEASED
LOCK=$CONTROL/SUBMISSION_LOCK
EVAL=$V3_ROOT/evaluation
CONFIG=$REPO/configs/training_qwen25_7b_kk_reasoning_pilot.yaml

cd "$REPO"
for directory in "$V3_ROOT" "$CONTROL"; do
  if [[ -e "$directory" || -L "$directory" ]]; then
    test -d "$directory" && test ! -L "$directory" || {
      echo "Refusing unsafe v3 directory: $directory" >&2
      exit 3
    }
  fi
done
mkdir -p "$CONTROL" "$ROOT/outputs/logs"
test -f "$ENV_ROOT/.ready"
test "$(sha256sum "$CONFIG" | awk '{print $1}')" = \
  5caef6baeb07f4ab4de8901001d7adb02433794e15c1024a950dc3bf59f492cb
test -z "$(git status --porcelain)" || {
  echo "Refusing to submit from a dirty Tillicum checkout" >&2
  git status --short >&2
  exit 2
}
"$ENV_ROOT/bin/python" scripts/audit_knights_knaves_confirmation_v3_workflow.py verify-prep \
  --repo-root "$REPO" --v1-root "$V1_ROOT" --v2-root "$V2_ROOT" \
  --v3-data-root "$V3_DATA" --prep-file "$PREP"

for sentinel in GO_KK_V3_BENEFIT_UNIONS STOPPED_KK_V3_FINAL; do
  if [[ -e "$CONTROL/$sentinel" || -L "$CONTROL/$sentinel" ]]; then
    echo "Refusing preexisting K&K v3 decision sentinel: $CONTROL/$sentinel" >&2
    exit 3
  fi
done
if [[ -L "$EVAL" ]]; then
  echo "Refusing symlinked K&K v3 evaluation path: $EVAL" >&2
  exit 3
fi
if [[ -e "$EVAL" ]]; then
  test -d "$EVAL" || {
    echo "Refusing unsafe preexisting K&K v3 evaluation path: $EVAL" >&2
    exit 3
  }
  if [[ -n "$(find "$EVAL" -mindepth 1 -print -quit)" ]]; then
    echo "Refusing preexisting K&K v3 evaluation outputs under $EVAL" >&2
    exit 3
  fi
fi

if ! mkdir "$LOCK" 2>/dev/null; then
  echo "A K&K v3 submission attempt already owns $LOCK" >&2
  exit 3
fi
printf 'created_at=%s\nrepo_commit=%s\n' \
  "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)" > "$LOCK/.owner-$$"
mv "$LOCK/.owner-$$" "$LOCK/owner"
if [[ -e "$AUTH" || -e "$JOBS" || -e "$ATTEMPT" || -e "$SUBMITTED" || -e "$RELEASED" ]]; then
  echo "K&K v3 submission state already exists; refusing duplicate allocation" >&2
  exit 3
fi

"$ENV_ROOT/bin/python" scripts/audit_knights_knaves_confirmation_v3_workflow.py write-authorization \
  --repo-root "$REPO" --v1-root "$V1_ROOT" --v2-root "$V2_ROOT" \
  --v3-data-root "$V3_DATA" --prep-file "$PREP" \
  --ack-max-cost-usd 0.45 --output-file "$AUTH"
chmod 0400 "$AUTH"

released=false
cleanup_failure() {
  status=$?
  if (( status != 0 )) && [[ "$released" != true ]]; then
    echo "Dispatch stopped before release; any recorded job remains held." >&2
    [[ -s "$ATTEMPT" ]] && cat "$ATTEMPT" >&2
  fi
  exit "$status"
}
trap cleanup_failure EXIT

job=$(sbatch --parsable --hold --export=NONE \
  scripts/sbatch_knights_knaves_reasoning_confirmation_v3_tillicum_h200.sbatch)
job=${job%%;*}
[[ "$job" =~ ^[0-9]+$ ]]
printf 'stage\tjob_id\tmax_minutes\theld\n' > "$ATTEMPT.tmp"
printf 'evaluate_v3\t%s\t30\ttrue\n' "$job" >> "$ATTEMPT.tmp"
mv "$ATTEMPT.tmp" "$ATTEMPT"

record=$(scontrol show job "$job" -o | tr ' ' '\n')
test "$(awk -F= '$1=="JobState" {print $2; exit}' <<< "$record")" = PENDING
test "$(awk -F= '$1=="Reason" {print $2; exit}' <<< "$record")" = JobHeldUser
test "$(awk -F= '$1=="Requeue" {print $2; exit}' <<< "$record")" = 0
test "$(awk -F= '$1=="Account" {print $2; exit}' <<< "$record")" = stf
test "$(awk -F= '$1=="Partition" {print $2; exit}' <<< "$record")" = gpu-h200
test "$(awk -F= '$1=="TimeLimit" {print $2; exit}' <<< "$record")" = 00:30:00
nodes=$(awk -F= '$1=="NumNodes" {print $2; exit}' <<< "$record")
[[ "$nodes" = 1 || "$nodes" = 1-1 ]]
test "$(awk -F= '$1=="NumTasks" {print $2; exit}' <<< "$record")" = 1
requested_tres=$(sed -n 's/^ReqTRES=//p' <<< "$record")
test "$(tr ',' '\n' <<< "$requested_tres" | awk -F= '$1=="gres/gpu:h200" {print $2}')" = 1
test "$(tr ',' '\n' <<< "$requested_tres" | awk -F= '$1=="gres/gpu" {print $2}')" = 1
test "$(tr ',' '\n' <<< "$requested_tres" | awk -F= '$1=="node" {print $2}')" = 1

printf 'stage\tjob_id\tmax_minutes\treleased\n' > "$JOBS.tmp"
printf 'evaluate_v3\t%s\t30\ttrue\n' "$job" >> "$JOBS.tmp"
mv "$JOBS.tmp" "$JOBS"
chmod 0400 "$JOBS"
printf 'submitted_at=%s\nrepo_commit=%s\nevaluate_v3_job=%s\nnew_max_h200_minutes=30\ncumulative_released_max_h200_minutes=210\nmax_new_cost_usd=0.45\nselective_regeneration=false\nautomatic_medical_union_or_quorum=false\n' \
  "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)" "$job" > "$SUBMITTED.tmp"
mv "$SUBMITTED.tmp" "$SUBMITTED"

scontrol release "$job"
released=true
printf 'released_at=%s\nevaluate_v3_job=%s\n' \
  "$(date --iso-8601=seconds)" "$job" > "$RELEASED.tmp"
mv "$RELEASED.tmp" "$RELEASED"
trap - EXIT

echo "Submitted and released K&K v3 evaluation job $job"
echo 'Hard maximum: 30 H200-minutes = $0.45.'
echo 'No training, selective rerun, extra adapter, medical union, or quorum job exists.'
