#!/bin/bash
# Resume training after the padding-free collator audit stopped before step 1.

set -euo pipefail
umask 077

if [[ "$#" -ne 3 || "$1" != resume || "$2" != --ack-max-total-cost-usd || "$3" != 1.80 ]]; then
  cat >&2 <<'EOF'
Usage:
  scripts/resume_general_code_apps_repaired_training_tillicum.sh resume \
    --ack-max-total-cost-usd 1.80

Prior released jobs conservatively account for seven rounded H200-minutes.
This exact-once retry may allocate 30 training minutes and 60 evaluation
minutes. The cumulative maximum is therefore 97 H200-minutes (about $1.46),
below the original 120-minute / $1.80 authorization. Both jobs are no-requeue.
EOF
  exit 2
fi

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY

TILLICUM_ROOT=/gpfs/projects/stf/claizhan/subliminal-mitigate
REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate
ENV_ROOT=$TILLICUM_ROOT/envs/subliminal-mitigate-py311
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/general_code_apps_repaired_pilot_v1
CONTROL_ROOT=$OUTPUT_ROOT/control
PREVIOUS_ROOT=$CONTROL_ROOT/resume_227440_compat5
RESUME_ROOT=$CONTROL_ROOT/resume_229723_padding_free
RESUME_LOCK=$CONTROL_ROOT/RESUME_229723_PADDING_FREE_SUBMISSION_LOCK
RESUME_AUTH=$RESUME_ROOT/AUTHORIZED_REPAIR_WITHIN_ORIGINAL_CAP
RESUME_JOBS=$RESUME_ROOT/jobs.tsv
RESUMED=$RESUME_ROOT/RESUMED
ORIGINAL_COMMIT=a57dbf43fdf296dfdd31f14447e9a47e76db0405
PREVIOUS_COMMIT=b4a6f8c82aa325777b8eac3fafcb9275fc2b8714
TRAIN_SCRIPT=scripts/sbatch_general_code_apps_repaired_train_tillicum_h200.sbatch
EVALUATE_SCRIPT=scripts/sbatch_general_code_apps_repaired_evaluate_tillicum_h200.sbatch

cd "$REPO_ROOT"
test -f "$ENV_ROOT/.ready"
test -z "$(git status --porcelain)" || {
  echo "Refusing to resume from a dirty Tillicum checkout." >&2
  git status --short >&2
  exit 3
}
repair_commit=$(git rev-parse HEAD)
test "$(git rev-parse HEAD^)" = "$PREVIOUS_COMMIT"
test "$(git rev-parse HEAD~6)" = "$ORIGINAL_COMMIT"
test "$(git rev-list --parents -n 1 HEAD | awk '{print NF - 1}')" = 1

allowed_repair_path() {
  case "$1" in
    docs/repaired_apps_coding_pilot_protocol.md|\
    train_sft.py|\
    scripts/resume_general_code_apps_repaired_training_tillicum.sh|\
    scripts/status_general_code_apps_repaired_pilot_tillicum.sh|\
    scripts/verify_general_code_apps_repaired_authorization.py|\
    tests/test_completion_only_sft.py|\
    tests/test_repaired_code_pilot_workflow.py) return 0 ;;
    *) return 1 ;;
  esac
}
while IFS= read -r path; do
  allowed_repair_path "$path" || {
    echo "Training-retry commit changes an unauthorized path: $path" >&2
    exit 3
  }
done < <(git diff --name-only "$PREVIOUS_COMMIT" "$repair_commit")
repair_diff_sha256=$(git diff --binary "$PREVIOUS_COMMIT" "$repair_commit" | sha256sum | awk '{print $1}')

check_hash() {
  local expected=$1 path=$2
  test -f "$path" && test ! -L "$path"
  test "$(sha256sum "$path" | awk '{print $1}')" = "$expected"
}
check_hash aaeaa4c9a19732339845d6124fbbdfe054dba71f1d3e96a36d20b03e711b61b6 \
  "$CONTROL_ROOT/AUTHORIZED_MAX_COST_USD_1.80"
check_hash 088d816b780d5153e43e167c4caffbb84d548f2adb1f4cba51d42e244e066841 \
  "$CONTROL_ROOT/RESUME_227440_COMPAT5_SUBMISSION_LOCK/owner"
check_hash 64780baa16b4a4cace49a0936e30ec668c57906a4e033b586ea1d507bec7c032 \
  "$PREVIOUS_ROOT/AUTHORIZED_REPAIR_WITHIN_ORIGINAL_CAP"
check_hash b1afa5a1e3212fb625554d7e0ef4c6a40933aa41f5cc4ba34053377c51c145a8 \
  "$PREVIOUS_ROOT/jobs.tsv"
check_hash 8d8b531e15347e2d5ccbc8a67fba796d15c680d86aebbe5d65a90bd4df2f1667 \
  "$PREVIOUS_ROOT/RESUMED"
check_hash b1afa5a1e3212fb625554d7e0ef4c6a40933aa41f5cc4ba34053377c51c145a8 \
  "$PREVIOUS_ROOT/dispatch_attempt.tsv"
check_hash ea8888e22d3eedd6bece6e81e4bcb7c6af3c7bd979a7451907ec260e32b2d20c \
  "$OUTPUT_ROOT/PREP_COMPLETE"
check_hash d89b2daa42d59429971691e6a598711d512adb3fabd830c8e27c228f4054cdd2 \
  "$OUTPUT_ROOT/data/data_manifest.json"
check_hash 3067d31eb32b36695951f9de4fa1a7bbf4968318fb35a297a090e17264a01ade \
  "$OUTPUT_ROOT/model/apps_repaired_pilot/training_objective.json"
check_hash a08eafe3924bd69915216c2c4656c3a75f4f589b464524cad8b765422133be9e \
  "$TILLICUM_ROOT/outputs/logs/general_code_apps_repaired_train_229723.out"
check_hash 73e572a199dd6268463880e590c2ad224e57ded27488cd613a685e55a885db9c \
  "$TILLICUM_ROOT/outputs/logs/general_code_apps_repaired_train_229723.err"

test ! -e "$OUTPUT_ROOT/model/apps_repaired_pilot/TRAIN_COMPLETE"
test ! -e "$OUTPUT_ROOT/model/apps_repaired_pilot/checkpoint-10"
test ! -e "$OUTPUT_ROOT/model/apps_repaired_pilot/loss_mask_audit.json"
test ! -e "$OUTPUT_ROOT/model/apps_repaired_pilot/training_run_meta.json"
test ! -e "$OUTPUT_ROOT/evaluation/apps_validation/SELECTED_CHECKPOINT.json"
test ! -e "$OUTPUT_ROOT/evaluation/FINAL_EVALUATION_COMPLETE"
test -z "$(find "$OUTPUT_ROOT/model/apps_repaired_pilot" -mindepth 1 -maxdepth 1 \
  ! -name training_objective.json -print -quit)"

accounting_row() {
  local job_id=$1
  sacct -X -j "$job_id" --starttime 2026-08-14 \
    --format=JobIDRaw,State,ElapsedRaw,TimelimitRaw,AllocTRES,ExitCode -n -P \
    | awk -F'|' -v id="$job_id" '$1==id {print; exit}'
}
IFS='|' read -r id state elapsed limit allocation exit_code <<< "$(accounting_row 229722)"
test "$id" = 229722 && test "$state" = COMPLETED && test "$elapsed" = 32
test "$limit" = 26 && test "$exit_code" = 0:0
test "$(tr ',' '\n' <<< "$allocation" | awk -F= '$1=="gres/gpu:h200" {print $2}')" = 1
IFS='|' read -r id state elapsed limit allocation exit_code <<< "$(accounting_row 229723)"
test "$id" = 229723 && test "$state" = FAILED && test "$elapsed" = 62
test "$limit" = 30 && test "$exit_code" = 1:0
test "$(tr ',' '\n' <<< "$allocation" | awk -F= '$1=="gres/gpu:h200" {print $2}')" = 1
IFS='|' read -r id state elapsed limit allocation exit_code <<< "$(accounting_row 229724)"
test "$id" = 229724 && [[ "$state" == CANCELLED* ]] && test "$elapsed" = 0
test "$limit" = 60 && test -z "$allocation"

export PYTHONDONTWRITEBYTECODE=1
"$ENV_ROOT/bin/python" scripts/prepare_repaired_code_pilot_data.py audit \
  --output-root "$OUTPUT_ROOT/data"

echo "=== Slurm admission preflight (no jobs submitted) ==="
sbatch --test-only --export=NONE --no-requeue --nodes=1 --ntasks=1 \
  --gres=gpu:h200:1 --time=00:30:00 "$TRAIN_SCRIPT"
sbatch --test-only --export=NONE --no-requeue --nodes=1 --ntasks=1 \
  --gres=gpu:h200:1 --time=01:00:00 "$EVALUATE_SCRIPT"

if ! mkdir "$RESUME_LOCK" 2>/dev/null; then
  echo "Training-retry lock already exists; refusing duplicate allocations." >&2
  exit 3
fi
printf 'created_at=%s\nrepair_repo_commit=%s\n' \
  "$(date --iso-8601=seconds)" "$repair_commit" > "$RESUME_LOCK/owner"
if [[ -e "$RESUME_ROOT" ]]; then
  echo "Training-retry control state already exists; refusing duplicates." >&2
  exit 3
fi
mkdir "$RESUME_ROOT"

auth_build=$RESUME_ROOT/.authorization-$$
{
  printf 'within_original_authorization=true\n'
  printf 'original_auth_sha256=aaeaa4c9a19732339845d6124fbbdfe054dba71f1d3e96a36d20b03e711b61b6\n'
  printf 'previous_repair_repo_commit=%s\n' "$PREVIOUS_COMMIT"
  printf 'previous_resume_authorization_sha256=64780baa16b4a4cace49a0936e30ec668c57906a4e033b586ea1d507bec7c032\n'
  printf 'previous_resume_addendum_sha256=3f94458f060ede2acc51daa98f13a49ee49018521cf94fbc0327ca5321022330\n'
  printf 'previous_resume_jobs_sha256=b1afa5a1e3212fb625554d7e0ef4c6a40933aa41f5cc4ba34053377c51c145a8\n'
  printf 'previous_resume_resumed_sha256=8d8b531e15347e2d5ccbc8a67fba796d15c680d86aebbe5d65a90bd4df2f1667\n'
  printf 'previous_resume_dispatch_sha256=1c3fc4ba6f38a2f29dce998dd19c2e714c0511aaacc9337f1f2f821a882aa844\n'
  printf 'previous_resume_lock_owner_sha256=088d816b780d5153e43e167c4caffbb84d548f2adb1f4cba51d42e244e066841\n'
  printf 'previous_prepare_job_id=229722\nprevious_prepare_state=COMPLETED\nprevious_prepare_elapsed_seconds=32\nprevious_prepare_timelimit_minutes=26\n'
  printf 'previous_train_job_id=229723\nprevious_train_state=FAILED\nprevious_train_elapsed_seconds=62\nprevious_train_timelimit_minutes=30\n'
  printf 'previous_evaluate_job_id=229724\nprevious_evaluate_state=CANCELLED\nprevious_evaluate_elapsed_seconds=0\nprevious_evaluate_timelimit_minutes=60\n'
  printf 'previous_failure_reason=trl_padding_free_collator_audit_layout\n'
  printf 'previous_train_stdout_sha256=a08eafe3924bd69915216c2c4656c3a75f4f589b464524cad8b765422133be9e\n'
  printf 'previous_train_stderr_sha256=73e572a199dd6268463880e590c2ad224e57ded27488cd613a685e55a885db9c\n'
  printf 'prep_complete_sha256=ea8888e22d3eedd6bece6e81e4bcb7c6af3c7bd979a7451907ec260e32b2d20c\n'
  printf 'finalized_data_manifest_sha256=d89b2daa42d59429971691e6a598711d512adb3fabd830c8e27c228f4054cdd2\n'
  printf 'training_objective_sha256=3067d31eb32b36695951f9de4fa1a7bbf4968318fb35a297a090e17264a01ade\n'
  printf 'prior_rounded_h200_minutes=7\nretry_train_minutes=30\nretry_evaluate_minutes=60\n'
  printf 'new_allocations_max_h200_minutes=90\ncumulative_max_h200_minutes=97\noriginal_authorized_max_h200_minutes=120\n'
  printf 'remaining_unused_h200_minutes=23\ncumulative_max_cost_usd=1.46\noriginal_authorized_max_cost_usd=1.80\n'
  printf 'no_requeue=true\nautomatic_continuation=false\nreason=padding_free_collator_audit_layout_repair\n'
  printf 'repair_repo_commit=%s\nrepair_diff_sha256=%s\n' "$repair_commit" "$repair_diff_sha256"
  printf 'recorded_at=%s\n' "$(date --iso-8601=seconds)"
} > "$auth_build"
printf 'addendum_sha256=%s\n' "$(sha256sum "$auth_build" | awk '{print $1}')" >> "$auth_build"
mv "$auth_build" "$RESUME_AUTH"
"$ENV_ROOT/bin/python" scripts/verify_general_code_apps_repaired_authorization.py \
  --stage train --time-limit 00:30:00 --control-only

dispatch_attempt=$RESUME_ROOT/dispatch_attempt.tsv
printf 'stage\tjob_id\tmax_minutes\tsubmitted_at\n' > "$dispatch_attempt"
train_job=$(sbatch --parsable --hold --export=NONE --no-requeue \
  --nodes=1 --ntasks=1 --gres=gpu:h200:1 --time=00:30:00 "$TRAIN_SCRIPT")
train_job=${train_job%%;*}
[[ "$train_job" =~ ^[0-9]+$ ]]
printf 'train\t%s\t30\t%s\n' "$train_job" "$(date --iso-8601=seconds)" >> "$dispatch_attempt"
evaluate_job=$(sbatch --parsable --export=NONE --no-requeue \
  --nodes=1 --ntasks=1 --gres=gpu:h200:1 --time=01:00:00 \
  --kill-on-invalid-dep=yes --dependency="afterok:$train_job" "$EVALUATE_SCRIPT")
evaluate_job=${evaluate_job%%;*}
[[ "$evaluate_job" =~ ^[0-9]+$ ]]
printf 'evaluate\t%s\t60\t%s\n' "$evaluate_job" "$(date --iso-8601=seconds)" >> "$dispatch_attempt"

submitted_at=$(date --iso-8601=seconds)
jobs_build=$RESUME_ROOT/.jobs-$$
printf 'stage\tjob_id\tmax_minutes\tsubmitted_at\ntrain\t%s\t30\t%s\nevaluate\t%s\t60\t%s\n' \
  "$train_job" "$submitted_at" "$evaluate_job" "$submitted_at" > "$jobs_build"
mv "$jobs_build" "$RESUME_JOBS"
addendum_sha256=$(awk -F= '$1=="addendum_sha256" {print $2}' "$RESUME_AUTH")
jobs_sha256=$(sha256sum "$RESUME_JOBS" | awk '{print $1}')
resumed_build=$RESUME_ROOT/.resumed-$$
printf 'repair_repo_commit=%s\naddendum_sha256=%s\njobs_sha256=%s\ntrain_job_id=%s\nevaluate_job_id=%s\nsubmitted_at=%s\n' \
  "$repair_commit" "$addendum_sha256" "$jobs_sha256" "$train_job" \
  "$evaluate_job" "$submitted_at" > "$resumed_build"
printf 'dispatch_sha256=%s\n' "$(sha256sum "$resumed_build" | awk '{print $1}')" >> "$resumed_build"
mv "$resumed_build" "$RESUMED"

test "$(scontrol show job "$train_job" -o | tr ' ' '\n' | awk -F= '$1=="TimeLimit" {print $2; exit}')" = 00:30:00
test "$(scontrol show job "$evaluate_job" -o | tr ' ' '\n' | awk -F= '$1=="TimeLimit" {print $2; exit}')" = 01:00:00
for job_id in "$train_job" "$evaluate_job"; do
  job_record=$(scontrol show job "$job_id" -o | tr ' ' '\n')
  test "$(awk -F= '$1=="Requeue" {print $2; exit}' <<< "$job_record")" = 0
  pending_nodes=$(awk -F= '$1=="NumNodes" {print $2; exit}' <<< "$job_record")
  [[ "$pending_nodes" = 1 || "$pending_nodes" = 1-1 ]]
  test "$(awk -F= '$1=="NumTasks" {print $2; exit}' <<< "$job_record")" = 1
  requested_tres=$(sed -n 's/^ReqTRES=//p' <<< "$job_record")
  test "$(tr ',' '\n' <<< "$requested_tres" | awk -F= '$1=="gres/gpu:h200" {print $2}')" = 1
  test "$(tr ',' '\n' <<< "$requested_tres" | awk -F= '$1=="gres/gpu" {print $2}')" = 1
  test "$(tr ',' '\n' <<< "$requested_tres" | awk -F= '$1=="node" {print $2}')" = 1
done
scontrol release "$train_job"
echo "Submitted padding-free audit repair: train=$train_job evaluate=$evaluate_job"
echo 'Cumulative hard maximum is 97 H200-minutes / $1.46, below the original $1.80 ceiling.'
