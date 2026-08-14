#!/bin/bash
# Resume the U+2028 parser failure inside the original 120 H200-minute cap.

set -euo pipefail
umask 077

if [[ "$#" -ne 3 || "$1" != resume || "$2" != --ack-max-total-cost-usd || "$3" != 1.80 ]]; then
  cat >&2 <<'EOF'
Usage:
  scripts/resume_general_code_apps_repaired_pilot_tillicum.sh resume \
    --ack-max-total-cost-usd 1.80

The failed preparation job used 60 seconds. This repair may allocate at most:
  preparation retry: 29 minutes
  pilot training:    30 minutes
  evaluation:        60 minutes
Together with the prior minute, the cumulative ceiling remains exactly
120 H200-minutes / $1.80. All jobs are --no-requeue.
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
FIRST_RESUME_ROOT=$CONTROL_ROOT/resume_227440
SECOND_RESUME_ROOT=$CONTROL_ROOT/resume_227440_compat2
RESUME_ROOT=$CONTROL_ROOT/resume_227440_compat3
AUTH_FILE=$CONTROL_ROOT/AUTHORIZED_MAX_COST_USD_1.80
ORIGINAL_JOBS=$CONTROL_ROOT/jobs.tsv
ORIGINAL_SUBMITTED=$CONTROL_ROOT/SUBMITTED
ORIGINAL_LOCK_OWNER=$CONTROL_ROOT/SUBMISSION_LOCK/owner
RESUME_LOCK=$CONTROL_ROOT/RESUME_227440_COMPAT3_SUBMISSION_LOCK
RESUME_AUTH=$RESUME_ROOT/AUTHORIZED_REPAIR_WITHIN_ORIGINAL_CAP
RESUME_JOBS=$RESUME_ROOT/jobs.tsv
RESUMED=$RESUME_ROOT/RESUMED
ORIGINAL_COMMIT=a57dbf43fdf296dfdd31f14447e9a47e76db0405
FIRST_REPAIR_COMMIT=00ad408f5596270d77bd52f26dcedc3366277e00
SECOND_REPAIR_COMMIT=531eac8565d68738958b06fae9363af0a4bebf01
PREP_SCRIPT=scripts/sbatch_general_code_apps_repaired_prepare_tillicum_h200.sbatch
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
test "$(git rev-parse HEAD^)" = "$SECOND_REPAIR_COMMIT"
test "$(git rev-parse HEAD~2)" = "$FIRST_REPAIR_COMMIT"
test "$(git rev-parse HEAD~3)" = "$ORIGINAL_COMMIT"
test "$(git rev-list --parents -n 1 HEAD | awk '{print NF - 1}')" = 1
test "$(git rev-list --parents -n 1 "$FIRST_REPAIR_COMMIT" | awk '{print NF - 1}')" = 1
test "$(git rev-list --parents -n 1 "$SECOND_REPAIR_COMMIT" | awk '{print NF - 1}')" = 1

allowed_repair_path() {
  case "$1" in
    docs/repaired_apps_coding_pilot_protocol.md|\
    scripts/resume_general_code_apps_repaired_pilot_tillicum.sh|\
    scripts/run_lcb_sandbox_evaluation.py|\
    scripts/sbatch_general_code_apps_repaired_prepare_tillicum_h200.sbatch|\
    scripts/sbatch_general_code_apps_repaired_train_tillicum_h200.sbatch|\
    scripts/sbatch_general_code_apps_repaired_evaluate_tillicum_h200.sbatch|\
    scripts/stage_general_code_apps_repaired_pilot_tillicum.sh|\
    scripts/status_general_code_apps_repaired_pilot_tillicum.sh|\
    scripts/submit_general_code_apps_repaired_pilot_tillicum.sh|\
    scripts/verify_general_code_apps_repaired_authorization.py|\
    tests/test_lcb_evaluation_pipeline.py|\
    tests/test_repaired_code_pilot_workflow.py) return 0 ;;
    *) return 1 ;;
  esac
}
while IFS= read -r path; do
  allowed_repair_path "$path" || {
    echo "Repair commit changes an unauthorized path: $path" >&2
    exit 3
  }
done < <(git diff --name-only "$ORIGINAL_COMMIT" "$repair_commit")
repair_diff_sha256=$(git diff --binary "$ORIGINAL_COMMIT" "$repair_commit" | sha256sum | awk '{print $1}')

check_hash() {
  local expected=$1 path=$2
  test -f "$path" && test ! -L "$path"
  test "$(sha256sum "$path" | awk '{print $1}')" = "$expected"
}
check_hash aaeaa4c9a19732339845d6124fbbdfe054dba71f1d3e96a36d20b03e711b61b6 "$AUTH_FILE"
check_hash 2651789b4a3816e9c80f2deb4f2add2ff3249d1e2efa1978bba128457dfa7565 "$ORIGINAL_JOBS"
check_hash f8fbbc7b995ffd1742fa3e75bd665dffddfecb0cb7229fd44cf6bd3725adfb96 "$ORIGINAL_SUBMITTED"
check_hash 90e14986d8cf5f97525be340b2f47d85fb9715286a58314720921de7ba126f82 "$ORIGINAL_LOCK_OWNER"
check_hash 41db818be86dc46c930bbac83a9f5e5d90a9ce476de1aada310b3197e94394f2 "$OUTPUT_ROOT/data/data_manifest.json"
check_hash 0143474c4156902450a2d61081a579c8a7f5a50b47c952ab278d74fecd1fe09c "$OUTPUT_ROOT/data/apps_repaired_candidates_evaluator.jsonl"
check_hash cfb5be8a9f4b69b211bb09cfda12c3ddc8d74b987411af8e37d0c8896927bee6 "$OUTPUT_ROOT/data/apps_repaired_candidates.custom.json"
check_hash f0bde28741a1fc00c611fe724db6ae87adbe6c24608dbf79ba5001ee033732a2 "$OUTPUT_ROOT/data/apps_repaired_candidates.custom.meta.json"
check_hash 689da81c118af0c9d12bbbd5b69edc6a64972d25914d29da9f64e6c4f2a57dc2 "$OUTPUT_ROOT/data/apps_repaired_candidate_prompts.json"
check_hash ddab648c0a223fc5afa8e5b06198991185c1a0e4bc9bce897766bcb76383076b "$TILLICUM_ROOT/outputs/logs/general_code_apps_repaired_prepare_227440.out"
check_hash f4b7613b5c4e1ad4fabd2c5635e9bd721e68b33b66ca814e99b5cd193e0fdba3 "$TILLICUM_ROOT/outputs/logs/general_code_apps_repaired_prepare_227440.err"
check_hash ce10532a9709e85e0c9073fbe1e0d66d83d117716bed82bba1ae6cb1dddba50d "$CONTROL_ROOT/RESUME_227440_SUBMISSION_LOCK/owner"
check_hash 7112eb9ff96ff3ae5997fb4be01b55d2209cc65873616f20c86a116868c34b90 "$FIRST_RESUME_ROOT/AUTHORIZED_REPAIR_WITHIN_ORIGINAL_CAP"
check_hash cdc7735b18d4b0acb622d4c4b76c16c90f632fe8a723f4bbbed563f134876ae0 "$FIRST_RESUME_ROOT/jobs.tsv"
check_hash 3839fb3b2ec847d49ca1be477355c4be7a3c763b75a1d42b0fa4830e424be9cf "$FIRST_RESUME_ROOT/RESUMED"
check_hash cdc7735b18d4b0acb622d4c4b76c16c90f632fe8a723f4bbbed563f134876ae0 "$FIRST_RESUME_ROOT/dispatch_attempt.tsv"
check_hash 99a16e11621d2fbb08e3c361e9431863465933d75314317fc0a7644fe342782d "$CONTROL_ROOT/RESUME_227440_COMPAT2_SUBMISSION_LOCK/owner"
check_hash 79bf2d9316312aad8fc004478892d19deba228f757c10abece670f7e9524498f "$SECOND_RESUME_ROOT/AUTHORIZED_REPAIR_WITHIN_ORIGINAL_CAP"
check_hash a5d619a61152765b39f68d6e67ef9cbd352b8a34925c24f9800683e5201f4673 "$SECOND_RESUME_ROOT/jobs.tsv"
check_hash 7860dc64f230845b5bc108316bb0fc0100b6937e88997b942e2749e3498be785 "$SECOND_RESUME_ROOT/RESUMED"
check_hash a5d619a61152765b39f68d6e67ef9cbd352b8a34925c24f9800683e5201f4673 "$SECOND_RESUME_ROOT/dispatch_attempt.tsv"

test ! -e "$OUTPUT_ROOT/PREP_COMPLETE"
test ! -e "$OUTPUT_ROOT/model/apps_repaired_pilot/TRAIN_COMPLETE"
test ! -e "$OUTPUT_ROOT/evaluation/FINAL_EVALUATION_COMPLETE"
test ! -e "$OUTPUT_ROOT/data/apps_repaired_candidates.evaluation.json"

accounting_row() {
  local job_id=$1
  sacct -X -j "$job_id" --starttime 2026-08-13 \
    --format=JobIDRaw,State,ElapsedRaw,TimelimitRaw,AllocTRES,ExitCode -n -P \
    | awk -F'|' -v id="$job_id" '$1==id {print; exit}'
}
IFS='|' read -r id state elapsed limit allocation exit_code <<< "$(accounting_row 227440)"
test "$id" = 227440 && test "$state" = FAILED && test "$elapsed" = 60
test "$limit" = 30 && test "$exit_code" = 1:0
test "$(tr ',' '\n' <<< "$allocation" | awk -F= '$1=="gres/gpu:h200" {print $2}')" = 1
test "$(tr ',' '\n' <<< "$allocation" | awk -F= '$1=="gres/gpu" {print $2}')" = 1
test "$(tr ',' '\n' <<< "$allocation" | awk -F= '$1=="node" {print $2}')" = 1
IFS='|' read -r id state elapsed limit allocation exit_code <<< "$(accounting_row 227441)"
test "$id" = 227441 && test "$state" = CANCELLED && test "$elapsed" = 0
test "$limit" = 30 && test -z "$allocation"
IFS='|' read -r id state elapsed limit allocation exit_code <<< "$(accounting_row 227442)"
test "$id" = 227442 && test "$state" = CANCELLED && test "$elapsed" = 0
test "$limit" = 60 && test -z "$allocation"
IFS='|' read -r id state elapsed limit allocation exit_code <<< "$(accounting_row 228953)"
test "$id" = 228953 && [[ "$state" == CANCELLED* ]] && test "$elapsed" = 0
test "$limit" = 29 && test -z "$allocation"
IFS='|' read -r id state elapsed limit allocation exit_code <<< "$(accounting_row 228954)"
test "$id" = 228954 && [[ "$state" == CANCELLED* ]] && test "$elapsed" = 0
test "$limit" = 30 && test -z "$allocation"
IFS='|' read -r id state elapsed limit allocation exit_code <<< "$(accounting_row 228955)"
test "$id" = 228955 && [[ "$state" == CANCELLED* ]] && test "$elapsed" = 0
test "$limit" = 60 && test -z "$allocation"
IFS='|' read -r id state elapsed limit allocation exit_code <<< "$(accounting_row 228992)"
test "$id" = 228992 && [[ "$state" == CANCELLED* ]] && test "$elapsed" = 0
test "$limit" = 29 && test -z "$allocation"
IFS='|' read -r id state elapsed limit allocation exit_code <<< "$(accounting_row 228993)"
test "$id" = 228993 && [[ "$state" == CANCELLED* ]] && test "$elapsed" = 0
test "$limit" = 30 && test -z "$allocation"
IFS='|' read -r id state elapsed limit allocation exit_code <<< "$(accounting_row 228994)"
test "$id" = 228994 && [[ "$state" == CANCELLED* ]] && test "$elapsed" = 0
test "$limit" = 60 && test -z "$allocation"

export PYTHONDONTWRITEBYTECODE=1
"$ENV_ROOT/bin/python" scripts/prepare_repaired_code_pilot_data.py audit \
  --output-root "$OUTPUT_ROOT/data"
"$ENV_ROOT/bin/python" - "$OUTPUT_ROOT/data/apps_repaired_candidates_evaluator.jsonl" <<'PY'
import hashlib
import sys
sys.path.insert(0, "scripts")
import run_lcb_sandbox_evaluation as sandbox

path = sys.argv[1]
raw = open(path, "rb").read()
assert hashlib.sha256(raw).hexdigest() == "0143474c4156902450a2d61081a579c8a7f5a50b47c952ab278d74fecd1fe09c"
assert raw.count("\u2028".encode("utf-8")) == 2
rows = sandbox.parse_json_or_jsonl_bytes(raw, path, True)
assert len(rows) == 2800
assert rows[647]["question_id"] == "apps-train-00918-stdio-00918"
print("Exact failed APPS JSONL now parses as 2,800 records with U+2028 preserved.")
PY

echo "=== Slurm admission preflight (no jobs submitted) ==="
sbatch --test-only --export=NONE --no-requeue --nodes=1 --ntasks=1 \
  --gres=gpu:h200:1 --time=00:29:00 "$PREP_SCRIPT"
sbatch --test-only --export=NONE --no-requeue --nodes=1 --ntasks=1 \
  --gres=gpu:h200:1 --time=00:30:00 "$TRAIN_SCRIPT"
sbatch --test-only --export=NONE --no-requeue --nodes=1 --ntasks=1 \
  --gres=gpu:h200:1 --time=01:00:00 "$EVALUATE_SCRIPT"

if ! mkdir "$RESUME_LOCK" 2>/dev/null; then
  echo "Resume lock already exists; refusing duplicate GPU allocations." >&2
  exit 3
fi
printf 'created_at=%s\nrepair_repo_commit=%s\n' \
  "$(date --iso-8601=seconds)" "$repair_commit" > "$RESUME_LOCK/owner"
if [[ -e "$RESUME_ROOT" ]]; then
  echo "Resume control state already exists; refusing duplicate GPU allocations." >&2
  exit 3
fi
mkdir "$RESUME_ROOT"

auth_build=$RESUME_ROOT/.authorization-$$
printf 'within_original_authorization=true\noriginal_auth_sha256=%s\noriginal_jobs_sha256=%s\noriginal_submitted_sha256=%s\noriginal_submission_lock_owner_sha256=%s\noriginal_repo_commit=%s\nfirst_repair_repo_commit=%s\nfirst_resume_authorization_sha256=%s\nfirst_resume_jobs_sha256=%s\nfirst_resume_resumed_sha256=%s\nfirst_resume_lock_owner_sha256=%s\nfirst_dispatch_prepare_job_id=228953\nfirst_dispatch_prepare_state=CANCELLED\nfirst_dispatch_prepare_elapsed_seconds=0\nfirst_dispatch_train_job_id=228954\nfirst_dispatch_train_state=CANCELLED\nfirst_dispatch_train_elapsed_seconds=0\nfirst_dispatch_evaluate_job_id=228955\nfirst_dispatch_evaluate_state=CANCELLED\nfirst_dispatch_evaluate_elapsed_seconds=0\nrepair_repo_commit=%s\nrepair_diff_sha256=%s\noriginal_prepare_job_id=227440\noriginal_prepare_state=FAILED\noriginal_prepare_elapsed_seconds=60\noriginal_prepare_timelimit_minutes=30\noriginal_train_job_id=227441\noriginal_train_state=CANCELLED\noriginal_train_elapsed_seconds=0\noriginal_evaluate_job_id=227442\noriginal_evaluate_state=CANCELLED\noriginal_evaluate_elapsed_seconds=0\nprior_rounded_h200_minutes=1\nresume_prepare_minutes=29\nresume_train_minutes=30\nresume_evaluate_minutes=60\nremaining_h200_minutes=119\ncumulative_max_h200_minutes=120\ncumulative_max_cost_usd=1.80\nno_requeue=true\nautomatic_continuation=false\nreason=unicode_jsonl_u2028_record_separator_parser_repair\nprepared_manifest_sha256=%s\nfailed_stdout_sha256=%s\nfailed_stderr_sha256=%s\nrecorded_at=%s\n' \
  aaeaa4c9a19732339845d6124fbbdfe054dba71f1d3e96a36d20b03e711b61b6 \
  2651789b4a3816e9c80f2deb4f2add2ff3249d1e2efa1978bba128457dfa7565 \
  f8fbbc7b995ffd1742fa3e75bd665dffddfecb0cb7229fd44cf6bd3725adfb96 \
  90e14986d8cf5f97525be340b2f47d85fb9715286a58314720921de7ba126f82 \
  "$ORIGINAL_COMMIT" "$FIRST_REPAIR_COMMIT" \
  7112eb9ff96ff3ae5997fb4be01b55d2209cc65873616f20c86a116868c34b90 \
  cdc7735b18d4b0acb622d4c4b76c16c90f632fe8a723f4bbbed563f134876ae0 \
  3839fb3b2ec847d49ca1be477355c4be7a3c763b75a1d42b0fa4830e424be9cf \
  ce10532a9709e85e0c9073fbe1e0d66d83d117716bed82bba1ae6cb1dddba50d \
  "$repair_commit" "$repair_diff_sha256" \
  41db818be86dc46c930bbac83a9f5e5d90a9ce476de1aada310b3197e94394f2 \
  ddab648c0a223fc5afa8e5b06198991185c1a0e4bc9bce897766bcb76383076b \
  f4b7613b5c4e1ad4fabd2c5635e9bd721e68b33b66ca814e99b5cd193e0fdba3 \
  "$(date --iso-8601=seconds)" > "$auth_build"
printf 'first_dispatch_failure_reason=tillicum_pending_jobs_omit_tresperjob\n' >> "$auth_build"
printf 'second_repair_repo_commit=%s\nsecond_resume_authorization_sha256=%s\nsecond_resume_jobs_sha256=%s\nsecond_resume_resumed_sha256=%s\nsecond_resume_lock_owner_sha256=%s\nsecond_dispatch_prepare_job_id=228992\nsecond_dispatch_prepare_state=CANCELLED\nsecond_dispatch_prepare_elapsed_seconds=0\nsecond_dispatch_train_job_id=228993\nsecond_dispatch_train_state=CANCELLED\nsecond_dispatch_train_elapsed_seconds=0\nsecond_dispatch_evaluate_job_id=228994\nsecond_dispatch_evaluate_state=CANCELLED\nsecond_dispatch_evaluate_elapsed_seconds=0\nsecond_dispatch_failure_reason=reqtres_split_on_every_equals\n' \
  "$SECOND_REPAIR_COMMIT" \
  79bf2d9316312aad8fc004478892d19deba228f757c10abece670f7e9524498f \
  a5d619a61152765b39f68d6e67ef9cbd352b8a34925c24f9800683e5201f4673 \
  7860dc64f230845b5bc108316bb0fc0100b6937e88997b942e2749e3498be785 \
  99a16e11621d2fbb08e3c361e9431863465933d75314317fc0a7644fe342782d \
  >> "$auth_build"
printf 'addendum_sha256=%s\n' "$(sha256sum "$auth_build" | awk '{print $1}')" >> "$auth_build"
mv "$auth_build" "$RESUME_AUTH"
"$ENV_ROOT/bin/python" scripts/verify_general_code_apps_repaired_authorization.py \
  --stage prepare --time-limit 00:29:00 --control-only

# Hold the first job until every ID and control record is durable. Any
# mid-dispatch failure therefore remains cost-free and fail-closed.
dispatch_attempt=$RESUME_ROOT/dispatch_attempt.tsv
printf 'stage\tjob_id\tmax_minutes\tsubmitted_at\n' > "$dispatch_attempt"
prepare_job=$(sbatch --parsable --hold --export=NONE --no-requeue \
  --nodes=1 --ntasks=1 --gres=gpu:h200:1 \
  --time=00:29:00 "$PREP_SCRIPT")
prepare_job=${prepare_job%%;*}
[[ "$prepare_job" =~ ^[0-9]+$ ]]
printf 'prepare\t%s\t29\t%s\n' "$prepare_job" "$(date --iso-8601=seconds)" >> "$dispatch_attempt"
train_job=$(sbatch --parsable --export=NONE --no-requeue \
  --nodes=1 --ntasks=1 --gres=gpu:h200:1 --time=00:30:00 \
  --kill-on-invalid-dep=yes --dependency="afterok:$prepare_job" "$TRAIN_SCRIPT")
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
printf 'stage\tjob_id\tmax_minutes\tsubmitted_at\nprepare\t%s\t29\t%s\ntrain\t%s\t30\t%s\nevaluate\t%s\t60\t%s\n' \
  "$prepare_job" "$submitted_at" "$train_job" "$submitted_at" \
  "$evaluate_job" "$submitted_at" > "$jobs_build"
mv "$jobs_build" "$RESUME_JOBS"
addendum_sha256=$(awk -F= '$1=="addendum_sha256" {print $2}' "$RESUME_AUTH")
jobs_sha256=$(sha256sum "$RESUME_JOBS" | awk '{print $1}')
resumed_build=$RESUME_ROOT/.resumed-$$
printf 'repair_repo_commit=%s\naddendum_sha256=%s\njobs_sha256=%s\nprepare_job_id=%s\ntrain_job_id=%s\nevaluate_job_id=%s\nsubmitted_at=%s\n' \
  "$repair_commit" "$addendum_sha256" "$jobs_sha256" "$prepare_job" "$train_job" \
  "$evaluate_job" "$submitted_at" > "$resumed_build"
printf 'dispatch_sha256=%s\n' "$(sha256sum "$resumed_build" | awk '{print $1}')" >> "$resumed_build"
mv "$resumed_build" "$RESUMED"

test "$(scontrol show job "$prepare_job" -o | tr ' ' '\n' | awk -F= '$1=="TimeLimit" {print $2; exit}')" = 00:29:00
test "$(scontrol show job "$train_job" -o | tr ' ' '\n' | awk -F= '$1=="TimeLimit" {print $2; exit}')" = 00:30:00
test "$(scontrol show job "$evaluate_job" -o | tr ' ' '\n' | awk -F= '$1=="TimeLimit" {print $2; exit}')" = 01:00:00
for job_id in "$prepare_job" "$train_job" "$evaluate_job"; do
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
scontrol release "$prepare_job"
echo "Submitted repaired-pilot resume: prepare=$prepare_job train=$train_job evaluate=$evaluate_job"
echo "Cumulative hard ceiling remains 120 H200-minutes / $1.80."
