#!/bin/bash
# Exactly one authorized panel/endpoint stage; stage comes only from JOBS.json.

#SBATCH --account=stf
#SBATCH --qos=normal
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=01:05:00
#SBATCH --no-requeue

set -euo pipefail
umask 077
ulimit -c 0

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY
unset MMU_RATIO_EVAL_TILLICUM_ROOT
while IFS= read -r variable; do
  case "$variable" in SBATCH_*) unset "$variable" ;; esac
done < <(compgen -v)

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-ratio-panel-evaluation-v1
env_root=$root/envs/subliminal-mitigate-py311
output=$root/outputs/massive_medical_ratio_panels_v1_evaluation
control=$output/control
manager=$repo/scripts/manage_massive_medical_ratio_panels_v1_evaluation.py
sampler=$repo/scripts/sample_massive_medical_ratio_panels_v1_evaluation.py
python=$env_root/bin/python

test -z "${SLURM_ARRAY_JOB_ID:-}"
test -z "${SLURM_ARRAY_TASK_ID:-}"
test -z "${SLURM_HET_GROUP:-}"
test -z "${SLURM_HET_JOB_ID:-}"
test "${SLURM_RESTART_COUNT:-0}" = 0
[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]]
test -s "$control/JOBS.json"
test -s "$control/AUTHORIZATION.json"
test -s "$control/RELEASE_AUTHORIZED.json"

stage=$("$python" -B -c 'import json,sys; d=json.load(open(sys.argv[1])); stages=("two_bad_two_benign_benefit","three_bad_one_benign_benefit","two_bad_two_benign_medical","three_bad_one_benign_medical"); matches=[s for s in stages if str(d["jobs"][s]["job_id"])==sys.argv[2]]; assert len(matches)==1; print(matches[0])' "$control/JOBS.json" "$SLURM_JOB_ID")
case "$stage" in
  two_bad_two_benign_benefit|three_bad_one_benign_benefit|two_bad_two_benign_medical|three_bad_one_benign_medical) ;;
  *) echo 'Job is not an exact authorized evaluation stage.' >&2; exit 2 ;;
esac

record_failure() {
  status=$?
  trap - EXIT
  if (( status != 0 )); then
    "$python" -B "$manager" record-failure \
      --stage "$stage" --job-id "$SLURM_JOB_ID" --exit-code "$status" || true
  fi
  exit "$status"
}
# A dispatch may race the submit-side post-release seal, but must not decode.
for ((release_wait=0; release_wait<60; release_wait++)); do
  [[ -s $control/RELEASED.json ]] && break
  sleep 1
done
test -s "$control/RELEASED.json"
cd "$repo"
"$python" -B "$manager" verify-runtime --stage "$stage" --job-id "$SLURM_JOB_ID"
# Only the process that successfully claims the permanent runtime entry may
# write a stage-failure receipt. A rejected duplicate must not poison its owner.
trap record_failure EXIT

# Only after release, scheduler, and provenance reconciliation may the model
# runtime be initialized. No performance-conditioned release gate is used.
module load gcc/13.4.0 cuda/12.9.1
module load conda/Miniforge3-25.3.1-3
conda activate "$env_root"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export XDG_CACHE_HOME=$root/cache XDG_CONFIG_HOME=$root/config
export PIP_CACHE_DIR=$root/cache/pip TMPDIR=$root/tmp
export HF_HOME=$root/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$root/cache/huggingface/hub
export TRANSFORMERS_CACHE=$root/cache/huggingface
export VLLM_CACHE_ROOT=$root/cache/vllm TRITON_CACHE_DIR=$root/cache/triton
export UNSLOTH_COMPILE_LOCATION=$root/cache/unsloth_compiled_cache
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-ratio-eval-pyc-$SLURM_JOB_ID
mkdir -p "$XDG_CONFIG_HOME" "$PIP_CACHE_DIR" "$TMPDIR" \
  "$HUGGINGFACE_HUB_CACHE" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" \
  "$UNSLOTH_COMPILE_LOCATION"

"$python" -B "$sampler" --stage "$stage" --job-id "$SLURM_JOB_ID" --device cuda:0
"$python" -B "$sampler" --stage "$stage" --job-id "$SLURM_JOB_ID" --audit-only
"$python" -B "$manager" complete-stage --stage "$stage" --job-id "$SLURM_JOB_ID"
trap - EXIT
echo "EVALUATION_STAGE_SEALED: $stage; medical judging remains separately authorized."
