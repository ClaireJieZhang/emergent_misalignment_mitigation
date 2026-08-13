#!/bin/bash
# Final resume after quarantining a derived Python bytecode cache.
# Usage: scripts/resume_evalplus_diagnostic_third_tillicum.sh resume --prior-job 226890 --resume-minutes 53

set -euo pipefail
umask 077

if [[ "$#" -ne 5 || "$1" != resume || "$2" != --prior-job || "$3" != 226890 || \
      "$4" != --resume-minutes || "$5" != 53 ]]; then
  echo "Usage: $0 resume --prior-job 226890 --resume-minutes 53" >&2
  exit 2
fi

TILLICUM_ROOT=/gpfs/projects/stf/claizhan/subliminal-mitigate
REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/general_code_evalplus_base_vs_pilot_v1
CONTROL_ROOT=$OUTPUT_ROOT/control
AUTH_FILE=$CONTROL_ROOT/AUTHORIZED_MAX_COST_USD_0.90
FIRST_AUTH=$CONTROL_ROOT/AUTHORIZED_RESUME_WITHIN_ORIGINAL_CAP
SECOND_AUTH=$CONTROL_ROOT/AUTHORIZED_SECOND_RESUME_WITHIN_ORIGINAL_CAP
THIRD_AUTH=$CONTROL_ROOT/AUTHORIZED_THIRD_RESUME_WITHIN_ORIGINAL_CAP
JOBS_FILE=$CONTROL_ROOT/jobs.tsv
SBATCH_SCRIPT=scripts/sbatch_general_code_evalplus_diagnostic_tillicum_h200.sbatch

cd "$REPO_ROOT"
test -z "$(git status --porcelain)"
test "$(sha256sum "$AUTH_FILE" | awk '{print $1}')" = \
  c2a1be0618948e29258aaee7a502bec9b3e4cf75641c119a46f5933a67723f89
test "$(sha256sum "$FIRST_AUTH" | awk '{print $1}')" = \
  79f95d1b64e3703573435fad421bff2ac63777a5e3e89da12b46c967fdafc9e1
test "$(sha256sum "$SECOND_AUTH" | awk '{print $1}')" = \
  b68e3f6afe115f73650ee03535bd334053559a70e475fcdfd46c3a220e667c10
test "$(sha256sum "$JOBS_FILE" | awk '{print $1}')" = \
  a3ef021794bcaefd2eb5fa22957abd2cdda9f03b92c604492274337057df0c9b
test "$(sha256sum "$CONTROL_ROOT/RESUMED_2" | awk '{print $1}')" = \
  5d00e18a46a2b11fbb34d71dc8f08bdfa302d1cb72f9fd1145832ca753b98073
test "$(sha256sum "$TILLICUM_ROOT/outputs/logs/general_code_evalplus_diagnostic_226890.out" | awk '{print $1}')" = \
  b4bfa9b172d9cc97139891662a1d535194531ed25734e7c13ca8589d9af3105c
test "$(sha256sum "$TILLICUM_ROOT/outputs/logs/general_code_evalplus_diagnostic_226890.err" | awk '{print $1}')" = \
  864c53dc503e28b5465dc07f2879803b2c602f10146bfb86b6f23de274fefe7d
test "$(sha256sum "$CONTROL_ROOT/quarantine/job226887_evalplus_site_pycache.sha256" | awk '{print $1}')" = \
  c4a2f936d8cf0614e57d31deb2019b3d47f4a0ef8d34a2da38b9c99b0b788866
test "$(sha256sum "$CONTROL_ROOT/quarantine/job226887_evalplus_site_pycache/appdirs.cpython-39.pyc" | awk '{print $1}')" = \
  6a13905616aee8ce20c185d876047986cad084ae28d7e4945c31d8c0bd946104
test ! -e "$THIRD_AUTH"
test ! -e "$CONTROL_ROOT/RESUMED_3"
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
for record in '226822 112 00:59:00' '226887 126 00:57:00' '226890 6 00:54:00'; do
  read -r job seconds limit <<< "$record"
  test "$(accounting "$job" State)" = FAILED
  test "$(accounting "$job" ElapsedRaw)" = "$seconds"
  test "$(accounting "$job" Timelimit)" = "$limit"
done

/usr/bin/python3 scripts/audit_evalplus_assets.py \
  --asset_root "$OUTPUT_ROOT/assets" \
  --source_root "$TILLICUM_ROOT/outputs/general_code_magicoder_lcb_q3_m4"

lock=$CONTROL_ROOT/THIRD_RESUME_LOCK
if ! mkdir "$lock" 2>/dev/null; then
  echo "Third-resume lock already exists; refusing duplicate allocation." >&2
  exit 3
fi
printf 'pid=%s\ncreated_at=%s\n' "$$" "$(date --iso-8601=seconds)" > "$lock/owner"

repair_commit=$(git rev-parse HEAD)
test "$(git rev-parse HEAD^)" = 6e86ac571a25934bbee701a8ca9e181f3a84e9ff

echo "=== Third-resume admission preflight (does not submit) ==="
sbatch --test-only --export=NONE --time=00:53:00 "$SBATCH_SCRIPT"

auth_build=$CONTROL_ROOT/.third-resume-authorization-$$
printf 'within_original_authorization=true\noriginal_auth_sha256=%s\nfirst_resume_auth_sha256=%s\nsecond_resume_auth_sha256=%s\nrepair_repo_commit=%s\nfirst_failed_job_id=226822\nfirst_state=FAILED\nfirst_elapsed_seconds=112\nsecond_failed_job_id=226887\nsecond_state=FAILED\nsecond_elapsed_seconds=126\nthird_failed_job_id=226890\nthird_state=FAILED\nthird_elapsed_seconds=6\nquarantined_pyc_sha256=%s\nfirst_rounded_minutes=2\nsecond_rounded_minutes=3\nthird_rounded_minutes=1\nresume_max_minutes=53\nresume_max_seconds=3180\ncumulative_max_seconds=3424\ncumulative_conservative_minutes=59\ncumulative_max_h200_hours=0.983333\ncumulative_max_cost_usd=0.90\nno_requeue=true\nreason=%s\nrecorded_at=%s\n' \
  c2a1be0618948e29258aaee7a502bec9b3e4cf75641c119a46f5933a67723f89 \
  79f95d1b64e3703573435fad421bff2ac63777a5e3e89da12b46c967fdafc9e1 \
  b68e3f6afe115f73650ee03535bd334053559a70e475fcdfd46c3a220e667c10 \
  "$repair_commit" 6a13905616aee8ce20c185d876047986cad084ae28d7e4945c31d8c0bd946104 \
  quarantined_derived_pyc_and_disabled_container_bytecode_writes \
  "$(date --iso-8601=seconds)" > "$auth_build"
printf 'addendum_sha256=%s\n' "$(sha256sum "$auth_build" | awk '{print $1}')" >> "$auth_build"
mv "$auth_build" "$THIRD_AUTH"

job_id=$(sbatch --parsable --export=NONE --time=00:53:00 "$SBATCH_SCRIPT")
job_id=${job_id%%;*}
jobs_build=$CONTROL_ROOT/.jobs-third-resume-$$
cp "$JOBS_FILE" "$jobs_build"
printf 'evalplus_diagnostic_resume_3\t%s\t%s\n' \
  "$job_id" "$(date --iso-8601=seconds)" >> "$jobs_build"
mv "$jobs_build" "$JOBS_FILE"
resumed_build=$CONTROL_ROOT/.resumed-3-$$
printf 'prior_job_id=226890\njob_id=%s\nresume_max_minutes=53\nsubmitted_at=%s\n' \
  "$job_id" "$(date --iso-8601=seconds)" > "$resumed_build"
mv "$resumed_build" "$CONTROL_ROOT/RESUMED_3"

echo "Submitted final resume job $job_id for at most 53 minutes."
echo 'Cumulative conservative cap remains 59 H200-minutes / below $0.90.'
