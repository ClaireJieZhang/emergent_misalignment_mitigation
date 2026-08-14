#!/bin/bash
# Submit the exact three-job repaired APPS pilot DAG.

set -euo pipefail
umask 077

if [[ "$#" -ne 3 || "$1" != pilot || "$2" != --ack-max-cost-usd || "$3" != 1.80 ]]; then
  cat >&2 <<'EOF'
Usage:
  scripts/submit_general_code_apps_repaired_pilot_tillicum.sh pilot --ack-max-cost-usd 1.80

This exact acknowledgement authorizes at most 120 H200-minutes at $0.90/hour:
  APPS download/filter/sandbox verification: 30 minutes
  one completion-only pilot trajectory:       30 minutes
  APPS selection + external evaluation:       60 minutes

Maximum: 2 H200-hours = $1.80. Every job is --no-requeue. This workflow
never trains additional adapters, runs quorum, or submits continuation jobs.
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
AUTH_FILE=$CONTROL_ROOT/AUTHORIZED_MAX_COST_USD_1.80
JOBS_FILE=$CONTROL_ROOT/jobs.tsv

cd "$REPO_ROOT"
mkdir -p "$CONTROL_ROOT" "$TILLICUM_ROOT/outputs/logs"
test -f "$ENV_ROOT/.ready" || {
  echo "Missing ready Tillicum environment: $ENV_ROOT" >&2
  exit 2
}
test -z "$(git status --porcelain)" || {
  echo "Refusing to submit from a dirty Tillicum checkout" >&2
  git status --short >&2
  exit 2
}
for path in \
  configs/training_qwen25_7b_apps_repaired_pilot.yaml \
  scripts/prepare_repaired_code_pilot_data.py \
  scripts/audit_repaired_code_pilot_model.py \
  scripts/select_repaired_code_pilot_checkpoint.py \
  scripts/summarize_repaired_code_pilot.py \
  scripts/run_lcb_one_tillicum.sh \
  scripts/run_evalplus_one_tillicum.sh \
  scripts/verify_general_code_apps_repaired_authorization.py \
  scripts/sbatch_general_code_apps_repaired_prepare_tillicum_h200.sbatch \
  scripts/sbatch_general_code_apps_repaired_train_tillicum_h200.sbatch \
  scripts/sbatch_general_code_apps_repaired_evaluate_tillicum_h200.sbatch; do
  test -s "$path" || { echo "Missing workflow file: $path" >&2; exit 2; }
done

# Persistent atomic lock: two concurrent acknowledgements must never both
# submit the capped DAG. It is intentionally retained even if a later
# preflight fails; a human must audit state before any retry.
SUBMISSION_LOCK=$CONTROL_ROOT/SUBMISSION_LOCK
if ! mkdir "$SUBMISSION_LOCK" 2>/dev/null; then
  echo "A repaired-pilot submission attempt already owns $SUBMISSION_LOCK" >&2
  echo "Inspect state before considering any retry." >&2
  exit 3
fi
printf 'created_at=%s\nrepo_commit=%s\n' \
  "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)" > "$SUBMISSION_LOCK/owner"

if [[ -e "$CONTROL_ROOT/SUBMITTED" || -e "$AUTH_FILE" || -s "$JOBS_FILE" ]]; then
  echo "This output root already has submission state; refusing duplicate GPU jobs." >&2
  echo "Inspect with scripts/status_general_code_apps_repaired_pilot_tillicum.sh" >&2
  exit 3
fi

repo_commit=$(git rev-parse HEAD)
auth_build=$CONTROL_ROOT/.authorization-$$
printf 'ack_max_cost_usd=1.80\nmax_h200_minutes=120\nh200_usd_per_hour=0.90\nprepare_minutes=30\ntrain_minutes=30\nevaluate_minutes=60\nno_requeue=true\nautomatic_continuation=false\nauthorized_at=%s\nrepo_commit=%s\n' \
  "$(date --iso-8601=seconds)" "$repo_commit" > "$auth_build"
mv "$auth_build" "$AUTH_FILE"

jobs_build=$CONTROL_ROOT/.jobs-$$
printf 'stage\tjob_id\tmax_minutes\tsubmitted_at\n' > "$jobs_build"
mv "$jobs_build" "$JOBS_FILE"
append_job() {
  local stage=$1 job_id=$2 minutes=$3
  printf '%s\t%s\t%s\t%s\n' "$stage" "$job_id" "$minutes" \
    "$(date --iso-8601=seconds)" >> "$JOBS_FILE"
}

prepare_job=$(sbatch --parsable --export=NONE \
  scripts/sbatch_general_code_apps_repaired_prepare_tillicum_h200.sbatch)
prepare_job=${prepare_job%%;*}
[[ "$prepare_job" =~ ^[0-9]+$ ]]
append_job prepare "$prepare_job" 30

train_job=$(sbatch --parsable --export=NONE --kill-on-invalid-dep=yes \
  --dependency="afterok:$prepare_job" \
  scripts/sbatch_general_code_apps_repaired_train_tillicum_h200.sbatch)
train_job=${train_job%%;*}
[[ "$train_job" =~ ^[0-9]+$ ]]
append_job train "$train_job" 30

evaluate_job=$(sbatch --parsable --export=NONE --kill-on-invalid-dep=yes \
  --dependency="afterok:$train_job" \
  scripts/sbatch_general_code_apps_repaired_evaluate_tillicum_h200.sbatch)
evaluate_job=${evaluate_job%%;*}
[[ "$evaluate_job" =~ ^[0-9]+$ ]]
append_job evaluate "$evaluate_job" 60

submitted_build=$CONTROL_ROOT/.submitted-$$
printf 'submitted_at=%s\nrepo_commit=%s\nprepare_job=%s\ntrain_job=%s\nevaluate_job=%s\n' \
  "$(date --iso-8601=seconds)" "$repo_commit" "$prepare_job" "$train_job" \
  "$evaluate_job" > "$submitted_build"
mv "$submitted_build" "$CONTROL_ROOT/SUBMITTED"

echo "Submitted repaired pilot: prepare=$prepare_job train=$train_job evaluate=$evaluate_job"
echo "Hard ceiling: 2 H200-hours = $1.80. No continuation jobs exist."
echo "Output root: $OUTPUT_ROOT"
