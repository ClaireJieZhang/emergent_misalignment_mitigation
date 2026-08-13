#!/bin/bash
# Cost-gated entry point for the unattended general-coding experiment.
# The only accepted invocation is:
#   scripts/submit_general_code_magicoder_lcb_tillicum.sh overnight --ack-max-cost-usd 14.40

set -euo pipefail
umask 077

if [[ "$#" -ne 3 || "$1" != overnight || "$2" != --ack-max-cost-usd || "$3" != 14.40 ]]; then
  cat >&2 <<'EOF'
Usage:
  scripts/submit_general_code_magicoder_lcb_tillicum.sh overnight --ack-max-cost-usd 14.40

This exact acknowledgement authorizes at most 16 H200-hours at $0.90/hour:
  pre-gate preparation/train/gate: 3 H200-hours ($2.70 maximum)
  post-GO continuation:           13 H200-hours ($11.70 maximum)
  whole workflow:                 16 H200-hours ($14.40 maximum)

Tillicum currently requires a GPU request even for preparation and sandbox
scoring. NO_GO submits no post-gate jobs. Every job is --no-requeue.
EOF
  exit 2
fi

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY

TILLICUM_ROOT=/gpfs/projects/stf/claizhan/subliminal-mitigate
REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate
ENV_ROOT=$TILLICUM_ROOT/envs/subliminal-mitigate-py311
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/general_code_magicoder_lcb_q3_m4
CONTROL_ROOT=$OUTPUT_ROOT/control
BAD_MODEL=$TILLICUM_ROOT/staged/bad_medical_pi_A
BAD_CONFIG_SHA256=87b2798d8e7deabc5d13907f4e729bd54b5fb9c08401e5122d8988f7502bd643
BAD_WEIGHTS_SHA256=cdcf3125f2538009e06ecce4c2ab1e0cdc3e72317e540c0e807e169fe8820214
JOBS_FILE=$CONTROL_ROOT/jobs.tsv
AUTH_FILE=$CONTROL_ROOT/AUTHORIZED_MAX_COST_USD_14.40

cd "$REPO_ROOT"
mkdir -p "$CONTROL_ROOT" "$TILLICUM_ROOT/outputs/logs"
test -f "$ENV_ROOT/.ready" || {
  echo "Missing ready Tillicum environment: $ENV_ROOT" >&2
  exit 2
}
test -s "$BAD_MODEL/adapter_config.json" || {
  echo "Missing staged bad-medical adapter: $BAD_MODEL" >&2
  exit 2
}
echo "$BAD_CONFIG_SHA256  $BAD_MODEL/adapter_config.json" | sha256sum -c -
echo "$BAD_WEIGHTS_SHA256  $BAD_MODEL/adapter_model.safetensors" | sha256sum -c -
for script in \
  scripts/sbatch_general_code_prepare_tillicum.sbatch \
  scripts/sbatch_general_code_train_tillicum_h200.sbatch \
  scripts/sbatch_general_code_gate_tillicum_h200.sbatch \
  scripts/sbatch_general_code_direct_final_tillicum_h200.sbatch \
  scripts/sbatch_general_code_quorum_tillicum_h200.sbatch \
  scripts/sbatch_general_code_final_evaluation_tillicum.sbatch \
  scripts/dispatch_general_code_magicoder_lcb_tillicum.sh; do
  test -s "$script" || {
    echo "Missing workflow script: $script" >&2
    exit 2
  }
done

if [[ -e "$CONTROL_ROOT/SUBMITTED" || -e "$AUTH_FILE" || -s "$JOBS_FILE" ]]; then
  echo "This output root already has submission state. Refusing duplicate GPU jobs." >&2
  echo "Inspect it with scripts/status_general_code_magicoder_lcb_tillicum.sh" >&2
  exit 3
fi

auth_build=$CONTROL_ROOT/.authorization-$$
printf 'ack_max_cost_usd=14.40\nmax_h200_hours=16\nh200_usd_per_hour=0.90\nauthorized_at=%s\nrepo_commit=%s\n' \
  "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)" > "$auth_build"
mv "$auth_build" "$AUTH_FILE"

jobs_build=$CONTROL_ROOT/.jobs-$$
printf 'stage\tjob_id\tsubmitted_at\n' > "$jobs_build"
mv "$jobs_build" "$JOBS_FILE"
append_job() {
  local stage=$1
  local job_id=$2
  printf '%s\t%s\t%s\n' "$stage" "$job_id" "$(date --iso-8601=seconds)" >> "$JOBS_FILE"
}

echo "Submitting the pre-gate DAG. No continuation job exists until the pilot writes GO."
prep_job=$(sbatch --parsable --export=NONE \
  scripts/sbatch_general_code_prepare_tillicum.sbatch)
prep_job=${prep_job%%;*}
append_job prepare_h200 "$prep_job"

pilot_train_job=$(sbatch --parsable --export=NONE --dependency="afterok:$prep_job" \
  scripts/sbatch_general_code_train_tillicum_h200.sbatch 0)
pilot_train_job=${pilot_train_job%%;*}
append_job train_pi_good_0 "$pilot_train_job"

gate_job=$(sbatch --parsable --export=NONE --dependency="afterok:$pilot_train_job" \
  scripts/sbatch_general_code_gate_tillicum_h200.sbatch)
gate_job=${gate_job%%;*}
append_job pilot_gate "$gate_job"

submitted_build=$CONTROL_ROOT/.submitted-$$
printf 'submitted_at=%s\nprepare_job=%s\npilot_train_job=%s\ngate_job=%s\n' \
  "$(date --iso-8601=seconds)" "$prep_job" "$pilot_train_job" "$gate_job" \
  > "$submitted_build"
mv "$submitted_build" "$CONTROL_ROOT/SUBMITTED"

echo "Submitted: prepare=$prep_job pilot_train=$pilot_train_job gate=$gate_job"
echo "Output root: $OUTPUT_ROOT"
echo 'NO_GO maximum: 3 H200-hours = $2.70.'
echo 'GO maximum: 16 H200-hours = $14.40.'
echo "The gate job will submit the remaining DAG itself only after an audited GO."
