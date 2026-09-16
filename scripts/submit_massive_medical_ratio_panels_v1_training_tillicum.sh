#!/bin/bash
# Exact held-first release of the two separately authorized training jobs.

set -euo pipefail
umask 077
ulimit -c 0

[[ $# -eq 2 && $1 == --ack-max-cost-usd && $2 == 0.900000 ]] || {
  echo 'Usage: scripts/submit_massive_medical_ratio_panels_v1_training_tillicum.sh --ack-max-cost-usd 0.900000' >&2
  exit 2
}

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-ratio-panels-v1
output=$root/outputs/massive_medical_ratio_panels_v1
control=$output/control/training
logs=$root/outputs/logs
python=$root/envs/subliminal-mitigate-py311/bin/python
manager=$repo/scripts/manage_massive_medical_ratio_panels_v1.py
batch=$repo/scripts/sbatch_massive_medical_ratio_panels_v1_train_tillicum_h200.sbatch

cd "$repo"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
"$python" "$manager" audit-stage >/dev/null
"$python" "$manager" authorize --ack-max-cost-usd "$2"
"$python" "$manager" record-submission-lock

# Slurm treats inherited SBATCH_* variables as implicit command-line options.
# Remove every such option before either exact held submission so a caller
# cannot silently turn either job into an array or alter its resource shape.
while IFS='=' read -r variable _; do
  case "$variable" in
    SBATCH_*) unset "$variable" ;;
  esac
done < <(env)

temporary=$(mktemp -d "$control/.held-submit.XXXXXX")
a2_job_id=
a3_job_id=
released=0

cleanup() {
  status=$?
  trap - EXIT
  if (( status != 0 && released == 0 )); then
    for job_id in "$a2_job_id" "$a3_job_id"; do
      if [[ -n $job_id ]]; then
        scontrol hold "$job_id" >/dev/null 2>&1 || true
        scancel "$job_id" >/dev/null 2>&1 || true
      fi
    done
    "$python" "$manager" record-submission-failure \
      --exit-code "$status" --a2-job-id "$a2_job_id" --a3-job-id "$a3_job_id" \
      >/dev/null 2>&1 || true
  fi
  rm -rf -- "$temporary"
  exit "$status"
}
trap cleanup EXIT

a2_job_id=$(sbatch --parsable --hold --export=NONE \
  --job-name=mmu_ratio_A2 \
  --output="$logs/massive_medical_ratio_panels_v1_A2_%j.out" \
  --error="$logs/massive_medical_ratio_panels_v1_A2_%j.err" \
  "$batch")
[[ $a2_job_id =~ ^[0-9]+$ ]]

a3_job_id=$(sbatch --parsable --hold --export=NONE \
  --job-name=mmu_ratio_A3 \
  --output="$logs/massive_medical_ratio_panels_v1_A3_%j.out" \
  --error="$logs/massive_medical_ratio_panels_v1_A3_%j.err" \
  "$batch")
[[ $a3_job_id =~ ^[0-9]+$ && $a3_job_id != "$a2_job_id" ]]

scontrol show job "$a2_job_id" -o > "$temporary/A2.held.scontrol"
scontrol show job "$a3_job_id" -o > "$temporary/A3.held.scontrol"
scontrol write batch_script "$a2_job_id" "$temporary/A2.spooled.sbatch" >/dev/null
scontrol write batch_script "$a3_job_id" "$temporary/A3.spooled.sbatch" >/dev/null

"$python" "$manager" record-jobs \
  --a2-job-id "$a2_job_id" \
  --a2-record-file "$temporary/A2.held.scontrol" \
  --a2-spooled-file "$temporary/A2.spooled.sbatch" \
  --a3-job-id "$a3_job_id" \
  --a3-record-file "$temporary/A3.held.scontrol" \
  --a3-spooled-file "$temporary/A3.spooled.sbatch"

"$python" "$manager" authorize-release

# Both jobs are byte- and resource-audited while held before either can run.
scontrol release "$a3_job_id"
scontrol release "$a2_job_id"
scontrol show job "$a2_job_id" -o > "$temporary/A2.released.scontrol"
scontrol show job "$a3_job_id" -o > "$temporary/A3.released.scontrol"

"$python" "$manager" record-release \
  --a2-record-file "$temporary/A2.released.scontrol" \
  --a3-record-file "$temporary/A3.released.scontrol"

released=1
trap - EXIT
rm -rf -- "$temporary"
echo "RELEASED_EXACT_TWO_TRAINING_JOBS: A2=$a2_job_id A3=$a3_job_id"
echo 'Maximum new exposure: 60 H200-minutes / $0.900000. No retry or downstream authority.'
