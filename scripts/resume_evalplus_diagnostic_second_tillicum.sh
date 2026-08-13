#!/bin/bash
# Second and final cost-neutral resume after two short, audited failures.
# Usage: scripts/resume_evalplus_diagnostic_second_tillicum.sh resume --prior-job 226887 --resume-minutes 54

set -euo pipefail
umask 077

if [[ "$#" -ne 5 || "$1" != resume || "$2" != --prior-job || "$3" != 226887 || \
      "$4" != --resume-minutes || "$5" != 54 ]]; then
  echo "Usage: $0 resume --prior-job 226887 --resume-minutes 54" >&2
  exit 2
fi

TILLICUM_ROOT=/gpfs/projects/stf/claizhan/subliminal-mitigate
REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/general_code_evalplus_base_vs_pilot_v1
CONTROL_ROOT=$OUTPUT_ROOT/control
AUTH_FILE=$CONTROL_ROOT/AUTHORIZED_MAX_COST_USD_0.90
FIRST_RESUME_AUTH=$CONTROL_ROOT/AUTHORIZED_RESUME_WITHIN_ORIGINAL_CAP
SECOND_RESUME_AUTH=$CONTROL_ROOT/AUTHORIZED_SECOND_RESUME_WITHIN_ORIGINAL_CAP
JOBS_FILE=$CONTROL_ROOT/jobs.tsv
SBATCH_SCRIPT=scripts/sbatch_general_code_evalplus_diagnostic_tillicum_h200.sbatch

cd "$REPO_ROOT"
test -z "$(git status --porcelain)"
test "$(sha256sum "$AUTH_FILE" | awk '{print $1}')" = \
  c2a1be0618948e29258aaee7a502bec9b3e4cf75641c119a46f5933a67723f89
test "$(sha256sum "$FIRST_RESUME_AUTH" | awk '{print $1}')" = \
  79f95d1b64e3703573435fad421bff2ac63777a5e3e89da12b46c967fdafc9e1
test "$(sha256sum "$JOBS_FILE" | awk '{print $1}')" = \
  10f919c1d21df3744b82be715a9dd2a2c83d6adc79a68fce0c7d98a2cf6475c0
test "$(sha256sum "$CONTROL_ROOT/RESUMED" | awk '{print $1}')" = \
  eae47341d5b0f50a0b7ebce025d544c1f5807692a749ec6fec9e606cfa1aad05
test "$(sha256sum "$TILLICUM_ROOT/outputs/logs/general_code_evalplus_diagnostic_226887.out" | awk '{print $1}')" = \
  5a5b7abe0a2998f2d25052bcd1ee7f77e5eb7cce366096e7aef5185bee70690e
test "$(sha256sum "$TILLICUM_ROOT/outputs/logs/general_code_evalplus_diagnostic_226887.err" | awk '{print $1}')" = \
  01f1d89ff702d13d08040d0dbe31a2e985db37bfd332b1035309603e9645e105
test ! -e "$CONTROL_ROOT/RESUMED_2"
test ! -e "$SECOND_RESUME_AUTH"
test ! -s "$OUTPUT_ROOT/DIAGNOSTIC_COMPLETE"
test -s "$OUTPUT_ROOT/generations/pi_base.json"
test -s "$OUTPUT_ROOT/generations/pi_good_0.json"
test -s "$OUTPUT_ROOT/evaluations/humaneval/pi_base.json"
test -s "$OUTPUT_ROOT/evaluations/humaneval/pi_good_0.json"
test ! -s "$OUTPUT_ROOT/evaluations/mbpp/pi_base.json"
test ! -s "$OUTPUT_ROOT/evaluations/mbpp/pi_good_0.json"

accounting() {
  local job_id=$1
  local field=$2
  sacct -X -j "$job_id" --starttime 2026-08-01 \
    --format=JobIDRaw,"$field" -n -P | \
    awk -F'|' -v id="$job_id" '$1==id {print $2; exit}'
}
test "$(accounting 226822 State)" = FAILED
test "$(accounting 226822 ElapsedRaw)" = 112
test "$(accounting 226822 Timelimit)" = 00:59:00
test "$(accounting 226887 State)" = FAILED
test "$(accounting 226887 ElapsedRaw)" = 126
test "$(accounting 226887 Timelimit)" = 00:57:00

lock=$CONTROL_ROOT/SECOND_RESUME_LOCK
if ! mkdir "$lock" 2>/dev/null; then
  echo "Second-resume lock already exists; refusing duplicate allocation." >&2
  exit 3
fi
printf 'pid=%s\ncreated_at=%s\n' "$$" "$(date --iso-8601=seconds)" > "$lock/owner"

original_commit=bd4ee4583b8eb6bc15e0ab8ad81a7e98ed69e56f
first_repair_commit=ea7612c772be59fc85ef47f6b9cdd009f757ca8e
repair_commit=$(git rev-parse HEAD)
test "$(git rev-parse HEAD^)" = "$first_repair_commit"

echo "=== Second-resume admission preflight (does not submit) ==="
sbatch --test-only --export=NONE --time=00:54:00 "$SBATCH_SCRIPT"

auth_build=$CONTROL_ROOT/.second-resume-authorization-$$
printf 'within_original_authorization=true\noriginal_auth_sha256=%s\nfirst_resume_auth_sha256=%s\noriginal_repo_commit=%s\nfirst_repair_repo_commit=%s\nrepair_repo_commit=%s\nfirst_failed_job_id=226822\nfirst_state=FAILED\nfirst_elapsed_seconds=112\nsecond_failed_job_id=226887\nsecond_state=FAILED\nsecond_elapsed_seconds=126\nsecond_stdout_sha256=%s\nsecond_stderr_sha256=%s\nfirst_rounded_minutes=2\nsecond_rounded_minutes=3\nresume_max_minutes=54\nresume_max_seconds=3240\ncumulative_max_seconds=3478\ncumulative_conservative_minutes=59\ncumulative_max_h200_hours=0.983333\ncumulative_max_cost_usd=0.90\nno_requeue=true\nreason=%s\nrecorded_at=%s\n' \
  c2a1be0618948e29258aaee7a502bec9b3e4cf75641c119a46f5933a67723f89 \
  79f95d1b64e3703573435fad421bff2ac63777a5e3e89da12b46c967fdafc9e1 \
  "$original_commit" "$first_repair_commit" "$repair_commit" \
  5a5b7abe0a2998f2d25052bcd1ee7f77e5eb7cce366096e7aef5185bee70690e \
  01f1d89ff702d13d08040d0dbe31a2e985db37bfd332b1035309603e9645e105 \
  evalplus_oracle_cache_write_suppressed_while_preserving_official_computation \
  "$(date --iso-8601=seconds)" > "$auth_build"
printf 'addendum_sha256=%s\n' "$(sha256sum "$auth_build" | awk '{print $1}')" >> "$auth_build"
mv "$auth_build" "$SECOND_RESUME_AUTH"

job_id=$(sbatch --parsable --export=NONE --time=00:54:00 "$SBATCH_SCRIPT")
job_id=${job_id%%;*}
jobs_build=$CONTROL_ROOT/.jobs-second-resume-$$
cp "$JOBS_FILE" "$jobs_build"
printf 'evalplus_diagnostic_resume_2\t%s\t%s\n' \
  "$job_id" "$(date --iso-8601=seconds)" >> "$jobs_build"
mv "$jobs_build" "$JOBS_FILE"
resumed_build=$CONTROL_ROOT/.resumed-2-$$
printf 'prior_job_id=226887\njob_id=%s\nresume_max_minutes=54\nsubmitted_at=%s\n' \
  "$job_id" "$(date --iso-8601=seconds)" > "$resumed_build"
mv "$resumed_build" "$CONTROL_ROOT/RESUMED_2"

echo "Submitted final resume job $job_id for at most 54 minutes."
echo 'Cumulative conservative cap remains 59 H200-minutes / below $0.90.'
