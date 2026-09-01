#!/bin/bash
#SBATCH --job-name=mmu_base_kalai_v1
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/massive_medical_composition_baselines_v1_kalai_smoke_%j.out
#SBATCH --error=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/massive_medical_composition_baselines_v1_kalai_smoke_%j.err

set -euo pipefail
umask 077
ulimit -c 0

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-composition-baselines-v1-stage-recovery-v3
output=$root/outputs/massive_medical_composition_baselines_v1
control=$output/control
source_protocol=$root/outputs/massive_medical_union_composition_exploratory_sequential_confirmation_v1_submit_recovery_v3/protocol/manifest.json
whole_root=$output/generation/whole_output
sampler=$repo/scripts/sample_massive_medical_whole_output_consensus_v1.py
auth_script=$repo/scripts/authorize_massive_medical_composition_baselines_v1.py
finalizer=$repo/scripts/finalize_massive_medical_composition_baseline_gpu_stage_v1.py
started=$SECONDS
completed=false

on_exit() {
  code=$?
  if [[ $completed != true && ! -e $control/WHOLE_OUTPUT_SMOKE_RESULT.json && ! -e $control/WHOLE_OUTPUT_SMOKE_STOPPED ]]; then
    temporary=$control/WHOLE_OUTPUT_SMOKE_STOPPED.tmp.$SLURM_JOB_ID
    printf 'stage=whole_output_smoke\njob_id=%s\nexit_code=%s\nrestart_or_resume_authorized=false\n' "$SLURM_JOB_ID" "$code" > "$temporary"
    chmod 0400 "$temporary"
    mv "$temporary" "$control/WHOLE_OUTPUT_SMOKE_STOPPED"
  fi
}
trap on_exit EXIT

[[ -z ${OPENAI_API_KEY:-} ]] || { echo 'OPENAI_API_KEY must not reach a GPU job.' >&2; exit 10; }
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY
unset TRANSFORMERS_CACHE

mkdir "$control/WHOLE_OUTPUT_SMOKE_INVOCATION_LOCK"
test ! -e "$whole_root"
test ! -e "$control/WHOLE_OUTPUT_SMOKE_RESULT.json"
test ! -e "$control/WHOLE_OUTPUT_SMOKE_STOPPED"

module load gcc/13.4.0 cuda/12.9.1
module load conda/Miniforge3-25.3.1-3
conda activate "$root/envs/subliminal-mitigate-py311"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-baseline-kalai-smoke-pyc-$SLURM_JOB_ID
export DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export XDG_CACHE_HOME=$root/cache XDG_CONFIG_HOME=$root/config TMPDIR=$root/tmp
export HF_HOME=$root/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRITON_CACHE_DIR=$root/cache/triton
mkdir -p "$XDG_CONFIG_HOME" "$TMPDIR" "$HUGGINGFACE_HUB_CACHE" "$TRITON_CACHE_DIR"

cd "$repo"
python "$auth_script" verify --stage whole_output_smoke --output-root "$output" --repo-root "$repo"
for phase in benefit medical; do
  python "$sampler" \
    --source-protocol-manifest "$source_protocol" \
    --output-root "$whole_root" \
    --phase "$phase" \
    --stage smoke \
    --device cuda:0
  python "$sampler" \
    --source-protocol-manifest "$source_protocol" \
    --output-root "$whole_root" \
    --phase "$phase" \
    --stage smoke \
    --device cuda:0 \
    --audit-only
done
python "$finalizer" \
  --stage whole_output_smoke \
  --output-root "$output" \
  --repo-root "$repo" \
  --elapsed-seconds "$((SECONDS - started))" \
  --slurm-job-id "$SLURM_JOB_ID"
completed=true
trap - EXIT
echo MASSIVE_MEDICAL_COMPOSITION_BASELINES_V1_WHOLE_OUTPUT_SMOKE_COMPLETE
