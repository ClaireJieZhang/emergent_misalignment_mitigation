#!/bin/bash
# Audit all four exact jobs held before a separately sealed release.

set -euo pipefail
umask 077
ulimit -c 0

[[ $# -eq 2 && $1 == --ack-max-cost-usd && $2 == 4.800000 ]] || {
  echo 'Usage: scripts/submit_massive_medical_ratio_panels_v1_evaluation_tillicum.sh --ack-max-cost-usd 4.800000' >&2
  exit 2
}

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY
unset MMU_RATIO_EVAL_TILLICUM_ROOT
while IFS= read -r variable; do
  case "$variable" in SBATCH_*) unset "$variable" ;; esac
done < <(compgen -v)

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-ratio-panel-evaluation-v1
output=$root/outputs/massive_medical_ratio_panels_v1_evaluation
control=$output/control
logs=$root/outputs/logs
python=$root/envs/subliminal-mitigate-py311/bin/python
manager=$repo/scripts/manage_massive_medical_ratio_panels_v1_evaluation.py
batch=$repo/scripts/sbatch_massive_medical_ratio_panels_v1_evaluation_tillicum_h200.sbatch
export PYTHONDONTWRITEBYTECODE=1

cd "$repo"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
"$python" -B "$manager" audit-stage >/dev/null
"$python" -B "$manager" authorize --ack-max-cost-usd "$2"
"$python" -B "$manager" record-submission-lock
test -d "$control"
test ! -L "$control"
test -d "$logs"

# Keep this narrowly scoped directory permanently as recoverable evidence,
# including after partial submission. Never delete it or permanent locks.
temporary=$(mktemp -d "$control/.held-submit.XXXXXX")
[[ $temporary == "$control"/.held-submit.* && -d $temporary && ! -L $temporary ]]
job_ids=()
release_authorized=0
stages=(
  two_bad_two_benign_benefit
  three_bad_one_benign_benefit
  two_bad_two_benign_medical
  three_bad_one_benign_medical
)

record_failure() {
  status=$?
  trap - EXIT
  if (( status != 0 )); then
    # Before release authority, only the jobs created by this wrapper may be
    # cancelled. After authority exists, preserve evidence without intervention.
    if (( release_authorized == 0 )) && [[ ! -e $control/RELEASE_AUTHORIZED.json ]]; then
      for job_id in "${job_ids[@]}"; do
        [[ $job_id =~ ^[0-9]+$ ]] || continue
        scontrol hold "$job_id" >/dev/null 2>&1 || true
        scancel "$job_id" >/dev/null 2>&1 || true
      done
    fi
    "$python" -B "$manager" record-submission-failure \
      --records-dir "$temporary" --exit-code "$status" || true
    echo "SUBMISSION_STOPPED: preserved evidence at $temporary; no automatic rerun." >&2
  fi
  exit "$status"
}
trap record_failure EXIT

for stage in "${stages[@]}"; do
  case "$stage" in
    two_bad_two_benign_benefit) name=mmu_ratio_eval_22_benefit; limit=01:05:00 ;;
    three_bad_one_benign_benefit) name=mmu_ratio_eval_31_benefit; limit=01:05:00 ;;
    two_bad_two_benign_medical) name=mmu_ratio_eval_22_medical; limit=01:35:00 ;;
    three_bad_one_benign_medical) name=mmu_ratio_eval_31_medical; limit=01:35:00 ;;
    *) exit 2 ;;
  esac
  sbatch --parsable --hold --export=NONE --time="$limit" \
    --job-name="$name" --chdir="$repo" \
    --output="$logs/massive_medical_ratio_panels_v1_evaluation_${stage}_%j.out" \
    --error="$logs/massive_medical_ratio_panels_v1_evaluation_${stage}_%j.err" \
    "$batch" > "$temporary/$stage.submitted.stdout"
  IFS= read -r job_id < "$temporary/$stage.submitted.stdout"
  [[ $job_id =~ ^[0-9]+$ ]]
  for existing_id in "${job_ids[@]}"; do
    [[ $job_id != "$existing_id" ]]
  done
  job_ids+=("$job_id")
  printf '%s\n' "$job_id" > "$temporary/$stage.job_id"
  scontrol show job "$job_id" -o > "$temporary/$stage.held.scontrol"
  scontrol write batch_script "$job_id" "$temporary/$stage.spooled.sbatch" >/dev/null
done

"$python" -B "$manager" record-jobs --records-dir "$temporary"
"$python" -B "$manager" authorize-release
release_authorized=1

# No release occurs until every held record and spooled script is audited.
for job_id in "${job_ids[@]}"; do
  scontrol release "$job_id"
done
for index in "${!stages[@]}"; do
  scontrol show job "${job_ids[$index]}" -o > "$temporary/${stages[$index]}.released.scontrol"
done
"$python" -B "$manager" record-release --records-dir "$temporary"
trap - EXIT

printf 'RELEASED_EXACT_FOUR_EVALUATION_JOBS:'
for index in "${!stages[@]}"; do
  printf ' %s=%s' "${stages[$index]}" "${job_ids[$index]}"
done
printf '\n'
echo "Submission evidence retained at $temporary."
echo 'Maximum new exposure: 320 H200-minutes / $4.800000; no retry, resume, or API authority.'
