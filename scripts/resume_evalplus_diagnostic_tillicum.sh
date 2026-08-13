#!/bin/bash
# Resume one failed diagnostic inside its original 59-minute/$0.90 ceiling.
# Usage: scripts/resume_evalplus_diagnostic_tillicum.sh resume --prior-job 226822 --resume-minutes 57

set -euo pipefail
umask 077

if [[ "$#" -ne 5 || "$1" != resume || "$2" != --prior-job || \
      ! "$3" =~ ^[0-9]+$ || "$4" != --resume-minutes || ! "$5" =~ ^[0-9]+$ ]]; then
  echo "Usage: $0 resume --prior-job JOB_ID --resume-minutes MINUTES" >&2
  exit 2
fi

prior_job=$3
requested_minutes=$5
TILLICUM_ROOT=/gpfs/projects/stf/claizhan/subliminal-mitigate
REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/general_code_evalplus_base_vs_pilot_v1
CONTROL_ROOT=$OUTPUT_ROOT/control
AUTH_FILE=$CONTROL_ROOT/AUTHORIZED_MAX_COST_USD_0.90
RESUME_AUTH_FILE=$CONTROL_ROOT/AUTHORIZED_RESUME_WITHIN_ORIGINAL_CAP
JOBS_FILE=$CONTROL_ROOT/jobs.tsv
SBATCH_SCRIPT=scripts/sbatch_general_code_evalplus_diagnostic_tillicum_h200.sbatch

cd "$REPO_ROOT"
test -z "$(git status --porcelain)" || {
  echo "Refusing to resume from a dirty Tillicum checkout." >&2
  exit 3
}
test -s "$AUTH_FILE"
test -s "$JOBS_FILE"
test ! -s "$OUTPUT_ROOT/DIAGNOSTIC_COMPLETE"
test "$(awk -F= '$1=="max_h200_hours" {print $2}' "$AUTH_FILE")" = 1
test "$(awk -F= '$1=="ack_max_cost_usd" {print $2}' "$AUTH_FILE")" = 0.90
original_auth_sha256=$(sha256sum "$AUTH_FILE" | awk '{print $1}')
test "$original_auth_sha256" = c2a1be0618948e29258aaee7a502bec9b3e4cf75641c119a46f5933a67723f89
test "$(sha256sum "$JOBS_FILE" | awk '{print $1}')" = \
  a00c11fd3b1612e6febef02fe7441cb3199043339ebf0b14b89a6ccaf5c02853
test "$(sha256sum "$CONTROL_ROOT/SUBMITTED" | awk '{print $1}')" = \
  6af043cfe97ff2cfd22f68dd2dc4c7d8f09fa9a91675ddf7d0ec62b4df987431

recorded_prior=$(awk -F '\t' '$1=="evalplus_diagnostic" {print $2}' "$JOBS_FILE")
test "$recorded_prior" = "$prior_job"
test "$(wc -l < "$JOBS_FILE" | tr -d ' ')" = 2
prior_state=$(sacct -X -j "$prior_job" --starttime 2026-08-01 \
  --format=JobIDRaw,State -n -P | awk -F'|' -v id="$prior_job" '$1==id {print $2; exit}')
prior_elapsed=$(sacct -X -j "$prior_job" --starttime 2026-08-01 \
  --format=JobIDRaw,ElapsedRaw -n -P | awk -F'|' -v id="$prior_job" '$1==id {print $2; exit}')
test "$prior_state" = FAILED
[[ "$prior_elapsed" =~ ^[0-9]+$ ]]
prior_rounded_minutes=$(( (prior_elapsed + 59) / 60 ))
remaining_minutes=$(( 59 - prior_rounded_minutes ))
if (( remaining_minutes <= 0 || requested_minutes != remaining_minutes )); then
  echo "Requested resume does not equal safe remaining cap: $remaining_minutes minutes" >&2
  exit 3
fi

lock=$CONTROL_ROOT/RESUME_LOCK
if ! mkdir "$lock" 2>/dev/null; then
  echo "Resume lock already exists; refusing duplicate allocation." >&2
  exit 3
fi
printf 'pid=%s\ncreated_at=%s\n' "$$" "$(date --iso-8601=seconds)" > "$lock/owner"
if [[ -e "$CONTROL_ROOT/RESUMED" || -e "$RESUME_AUTH_FILE" ]]; then
  echo "Resume state already exists; refusing duplicate allocation." >&2
  exit 3
fi

original_commit=$(awk -F= '$1=="repo_commit" {print $2}' "$AUTH_FILE")
repair_commit=$(git rev-parse HEAD)
test "$original_commit" = bd4ee4583b8eb6bc15e0ab8ad81a7e98ed69e56f
test "$(git rev-parse HEAD^)" = "$original_commit"

echo "=== Resume admission preflight (does not submit) ==="
sbatch --test-only --export=NONE --time="00:${requested_minutes}:00" "$SBATCH_SCRIPT"

auth_build=$CONTROL_ROOT/.resume-authorization-$$
printf 'within_original_authorization=true\noriginal_auth_sha256=%s\noriginal_repo_commit=%s\nrepair_repo_commit=%s\noriginal_failed_job_id=%s\noriginal_state=%s\noriginal_elapsed_seconds=%s\nprior_rounded_minutes=%s\nresume_max_minutes=%s\nresume_max_seconds=3420\ncumulative_max_seconds=3532\ncumulative_max_minutes=59\ncumulative_max_h200_hours=0.983333\ncumulative_max_cost_usd=0.90\nno_requeue=true\nreason=%s\nrecorded_at=%s\n' \
  "$original_auth_sha256" \
  "$original_commit" "$repair_commit" "$prior_job" "$prior_state" \
  "$prior_elapsed" "$prior_rounded_minutes" "$requested_minutes" \
  oracle_cache_moved_to_node_local_result_then_deleted_before_workers \
  "$(date --iso-8601=seconds)" > "$auth_build"
printf 'addendum_sha256=%s\n' "$(sha256sum "$auth_build" | awk '{print $1}')" >> "$auth_build"
mv "$auth_build" "$RESUME_AUTH_FILE"

job_id=$(sbatch --parsable --export=NONE --time="00:${requested_minutes}:00" "$SBATCH_SCRIPT")
job_id=${job_id%%;*}
jobs_build=$CONTROL_ROOT/.jobs-resume-$$
cp "$JOBS_FILE" "$jobs_build"
printf 'evalplus_diagnostic_resume\t%s\t%s\n' \
  "$job_id" "$(date --iso-8601=seconds)" >> "$jobs_build"
mv "$jobs_build" "$JOBS_FILE"
resumed_build=$CONTROL_ROOT/.resumed-$$
printf 'prior_job_id=%s\njob_id=%s\nresume_max_minutes=%s\nsubmitted_at=%s\n' \
  "$prior_job" "$job_id" "$requested_minutes" "$(date --iso-8601=seconds)" \
  > "$resumed_build"
mv "$resumed_build" "$CONTROL_ROOT/RESUMED"

echo "Submitted resume job $job_id for at most ${requested_minutes} minutes."
echo 'Cumulative conservative cap: 59 H200-minutes, still below $0.90.'
