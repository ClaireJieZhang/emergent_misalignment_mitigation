#!/bin/bash
#SBATCH --job-name=mmu_kalai_s3_gate
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
#SBATCH --output=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/massive_medical_kalai_s3_r20_v2_submit_recovery_v3_gate_%j.out
#SBATCH --error=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/massive_medical_kalai_s3_r20_v2_submit_recovery_v3_gate_%j.err

set -euo pipefail
umask 077
ulimit -c 0

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-kalai-s3-r20-v2-submit-recovery-v3
output=$root/outputs/massive_medical_kalai_s3_r20_v2_submit_recovery_v3
source_protocol=$root/outputs/massive_medical_union_composition_exploratory_sequential_confirmation_v1_submit_recovery_v3/protocol/manifest.json
generation_root=$output/generation
control=$output/control
sampler=$repo/scripts/sample_massive_medical_whole_output_consensus_s3_v2.py
authorizer=$repo/scripts/authorize_massive_medical_kalai_s3_v2.py
evaluator=$repo/scripts/evaluate_massive_medical_kalai_s3_gate_v2.py
started=$SECONDS
completed=false

on_exit() {
  code=$?
  if [[ $completed != true && ! -e $control/GATE_RESULT.json && ! -e $control/GATE_STOPPED ]]; then
    temporary=$control/GATE_STOPPED.tmp.$SLURM_JOB_ID
    printf 'stage=gate\njob_id=%s\nexit_code=%s\nrestart_or_resume_authorized=false\ncompletion_authorized=false\n' "$SLURM_JOB_ID" "$code" > "$temporary"
    chmod 0400 "$temporary"
    mv "$temporary" "$control/GATE_STOPPED"
  fi
}

[[ -z ${OPENAI_API_KEY:-} ]] || { echo 'OPENAI_API_KEY must not reach a GPU job.' >&2; exit 10; }
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY
unset TRANSFORMERS_CACHE

test -d "$control/GATE_SUBMISSION_LOCK"
grep -Fx 'stage=gate' "$control/GATE_SUBMITTED" >/dev/null
grep -Fx "job_id=$SLURM_JOB_ID" "$control/GATE_SUBMITTED" >/dev/null
grep -Fx 'held_first=true' "$control/GATE_SUBMITTED" >/dev/null
grep -Fx 'held_audit_passed=true' "$control/GATE_SUBMITTED" >/dev/null
grep -Fx "job_id=$SLURM_JOB_ID" "$control/GATE_RELEASE_AUTHORIZED" >/dev/null
grep -Fx 'release_authorized=true' "$control/GATE_RELEASE_AUTHORIZED" >/dev/null
grep -Fx "job_id=$SLURM_JOB_ID" "$control/GATE_RELEASED" >/dev/null
grep -Fx 'released=true' "$control/GATE_RELEASED" >/dev/null
mkdir "$control/GATE_INVOCATION_LOCK"
trap on_exit EXIT
test ! -e "$generation_root"
test ! -e "$control/GATE_RESULT.json"
test ! -e "$control/GATE_STOPPED"
test ! -e "$control/COMPLETION_AUTHORIZATION.json"

module load gcc/13.4.0 cuda/12.9.1
module load conda/Miniforge3-25.3.1-3
conda activate "$root/envs/subliminal-mitigate-py311"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-kalai-s3-gate-pyc-$SLURM_JOB_ID
export DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export XDG_CACHE_HOME=$root/cache XDG_CONFIG_HOME=$root/config TMPDIR=$root/tmp
export HF_HOME=$root/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRITON_CACHE_DIR=$root/cache/triton
mkdir -p "$XDG_CONFIG_HOME" "$TMPDIR" "$HUGGINGFACE_HUB_CACHE" "$TRITON_CACHE_DIR"

cd "$repo"
python "$authorizer" verify \
  --stage gate \
  --output-root "$output" \
  --repo-root "$repo"
for phase in benefit medical; do
  python "$sampler" \
    --source-protocol-manifest "$source_protocol" \
    --output-root "$generation_root" \
    --phase "$phase" \
    --stage gate \
    --gpu-authorization "$control/GATE_AUTHORIZATION.json" \
    --device cuda:0
  python "$sampler" \
    --source-protocol-manifest "$source_protocol" \
    --output-root "$generation_root" \
    --phase "$phase" \
    --stage gate \
    --device cuda:0 \
    --audit-only
done
python "$evaluator" \
  --workflow-root "$output" \
  --source-protocol-manifest "$source_protocol" \
  --elapsed-seconds "$((SECONDS - started))" \
  --slurm-job-id "$SLURM_JOB_ID" \
  --authorization "$control/GATE_AUTHORIZATION.json"
completed=true
trap - EXIT
echo MASSIVE_MEDICAL_KALAI_S3_R20_V2_GATE_COMPLETE_NO_AUTOMATIC_CONTINUATION
