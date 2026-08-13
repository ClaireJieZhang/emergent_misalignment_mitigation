#!/bin/bash
# Exact cost acknowledgement for the isolated EvalPlus diagnostic.
# Usage: scripts/submit_evalplus_diagnostic_tillicum.sh diagnostic --ack-max-cost-usd 0.90

set -euo pipefail
umask 077

if [[ "$#" -ne 3 || "$1" != diagnostic || "$2" != --ack-max-cost-usd || "$3" != 0.90 ]]; then
  cat >&2 <<'EOF'
Usage:
  scripts/submit_evalplus_diagnostic_tillicum.sh diagnostic --ack-max-cost-usd 0.90

This exact acknowledgement authorizes one non-requeued H200 job with a
59-minute time limit. Hard maximum: 1 H200-hour = $0.90. It evaluates only the
existing base and pi_good_0; it does not train, run quorum, or touch the sealed
LiveCodeBench final set.
EOF
  exit 2
fi

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY

TILLICUM_ROOT=/gpfs/projects/stf/claizhan/subliminal-mitigate
REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate
SOURCE_ROOT=$TILLICUM_ROOT/outputs/general_code_magicoder_lcb_q3_m4
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/general_code_evalplus_base_vs_pilot_v1
CONTROL_ROOT=$OUTPUT_ROOT/control
ASSET_ROOT=$OUTPUT_ROOT/assets
DATA_ROOT=$OUTPUT_ROOT/data
MODEL_ROOT=$SOURCE_ROOT/models
AUTH_FILE=$CONTROL_ROOT/AUTHORIZED_MAX_COST_USD_0.90
JOBS_FILE=$CONTROL_ROOT/jobs.tsv
SBATCH_SCRIPT=scripts/sbatch_general_code_evalplus_diagnostic_tillicum_h200.sbatch

cd "$REPO_ROOT"
mkdir -p "$CONTROL_ROOT" "$TILLICUM_ROOT/outputs/logs"
test -z "$(git status --porcelain)" || {
  echo "Refusing to authorize a dirty Tillicum checkout." >&2
  git status --short >&2
  exit 3
}
test -s "$ASSET_ROOT/ASSETS_READY"
test -s "$MODEL_ROOT/pi_good_0/GENERAL_CODE_TRAIN_COMPLETE"
echo 'cc64b027f01370c95a0aeedc07afe75470dd70e2d4030733c99fae4926a486d0  '"$MODEL_ROOT/pi_good_0/adapter_config.json" | sha256sum -c -
echo 'ab12b48ab9f9af500f8d234a8fc434727722241b219dd5a38e09cd2ebfcbea60  '"$MODEL_ROOT/pi_good_0/adapter_model.safetensors" | sha256sum -c -
python scripts/prepare_evalplus_diagnostic.py --output_root "$DATA_ROOT" --audit-only
python scripts/audit_evalplus_assets.py \
  --asset_root "$ASSET_ROOT" \
  --source_root "$SOURCE_ROOT"
test -s "$SBATCH_SCRIPT"

SUBMIT_LOCK=$CONTROL_ROOT/SUBMIT_LOCK
if ! mkdir "$SUBMIT_LOCK" 2>/dev/null; then
  echo "A diagnostic submission lock already exists; refusing duplicate GPU work." >&2
  exit 3
fi
printf 'pid=%s\ncreated_at=%s\n' "$$" "$(date --iso-8601=seconds)" > "$SUBMIT_LOCK/owner"

if [[ -e "$CONTROL_ROOT/SUBMITTED" || -e "$AUTH_FILE" || -s "$JOBS_FILE" ]]; then
  echo "Diagnostic output root already has submission state; refusing duplicate GPU work." >&2
  echo "Inspect with scripts/status_evalplus_diagnostic_tillicum.sh" >&2
  exit 3
fi

echo "=== Slurm admission preflight (does not submit) ==="
sbatch --test-only --export=NONE "$SBATCH_SCRIPT"

auth_build=$CONTROL_ROOT/.authorization-$$
printf 'ack_max_cost_usd=0.90\nmax_h200_hours=1\nh200_usd_per_hour=0.90\njob_time_limit=00:59:00\nno_requeue=true\nauthorized_at=%s\nrepo_commit=%s\nsource_output_root=%s\npilot_adapter_config_sha256=%s\npilot_adapter_weights_sha256=%s\nevalplus_commit=%s\nhumaneval_sha256=%s\nmbpp_sha256=%s\nasset_manifest_sha256=%s\n' \
  "$(date --iso-8601=seconds)" \
  "$(git rev-parse HEAD)" \
  "$SOURCE_ROOT" \
  cc64b027f01370c95a0aeedc07afe75470dd70e2d4030733c99fae4926a486d0 \
  ab12b48ab9f9af500f8d234a8fc434727722241b219dd5a38e09cd2ebfcbea60 \
  e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2 \
  272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101 \
  af43697e8791c4c149bdfd6b489d8b5412507551ac20e28a439f650b8225db63 \
  "$(sha256sum "$ASSET_ROOT/asset_manifest.json" | awk '{print $1}')" \
  > "$auth_build"
mv "$auth_build" "$AUTH_FILE"

job_id=$(sbatch --parsable --export=NONE "$SBATCH_SCRIPT")
job_id=${job_id%%;*}
jobs_build=$CONTROL_ROOT/.jobs-$$
printf 'stage\tjob_id\tsubmitted_at\nevalplus_diagnostic\t%s\t%s\n' \
  "$job_id" "$(date --iso-8601=seconds)" > "$jobs_build"
mv "$jobs_build" "$JOBS_FILE"
submitted_build=$CONTROL_ROOT/.submitted-$$
printf 'job_id=%s\nsubmitted_at=%s\n' "$job_id" "$(date --iso-8601=seconds)" > "$submitted_build"
mv "$submitted_build" "$CONTROL_ROOT/SUBMITTED"

echo "Submitted EvalPlus diagnostic job $job_id"
echo "Hard maximum: 1 H200-hour = $0.90; no automatic requeue or continuation."
