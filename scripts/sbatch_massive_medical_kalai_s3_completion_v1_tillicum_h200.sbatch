#!/bin/bash
#SBATCH --job-name=mmu_kalai_s3_complete
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:34:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/massive_medical_kalai_s3_r20_v2_completion_v1_%j.out
#SBATCH --error=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/massive_medical_kalai_s3_r20_v2_completion_v1_%j.err

set -euo pipefail
umask 077
ulimit -c 0

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
controller_repo=$root/projects/subliminal-mitigate-mmu-kalai-s3-r20-v2-completion-controller-v1
controller_output=$root/outputs/massive_medical_kalai_s3_r20_v2_completion_controller_v1
gate_repo=$root/projects/subliminal-mitigate-mmu-kalai-s3-r20-v2-submit-recovery-v3
gate_output=$root/outputs/massive_medical_kalai_s3_r20_v2_submit_recovery_v3
source_protocol=$root/outputs/massive_medical_union_composition_exploratory_sequential_confirmation_v1_submit_recovery_v3/protocol/manifest.json
control=$gate_output/control
generation_root=$gate_output/generation
authorizer=$controller_repo/scripts/authorize_massive_medical_kalai_s3_completion_v1.py
runner=$controller_repo/scripts/sample_massive_medical_kalai_s3_completion_combined_v1.py
sampler=$controller_repo/scripts/sample_massive_medical_whole_output_consensus_s3_v2.py
assembler=$controller_repo/scripts/assemble_massive_medical_kalai_s3_v2.py
evaluator=$controller_repo/scripts/evaluate_massive_medical_kalai_s3_completion_v1.py
started=$SECONDS
completed=false

on_exit() {
  code=$?
  if [[ $completed != true && ! -e $control/COMPLETION_RESULT.json && ! -e $control/COMPLETION_STOPPED ]]; then
    temporary=$control/COMPLETION_STOPPED.tmp.$SLURM_JOB_ID
    printf 'controller_protocol_id=massive_medical_kalai_s3_r20_v2_completion_v1\nstage=completion\njob_id=%s\nexit_code=%s\nrestart_or_resume_authorized=false\nretry_authorized=false\nexternal_api_calls=0\n' \
      "$SLURM_JOB_ID" "$code" > "$temporary"
    chmod 0400 "$temporary"
    mv "$temporary" "$control/COMPLETION_STOPPED"
  fi
}
trap on_exit EXIT

[[ -z ${OPENAI_API_KEY:-} ]] || { echo 'OPENAI_API_KEY must not reach a GPU job.' >&2; exit 10; }
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY
unset TRANSFORMERS_CACHE

test -d "$control/COMPLETION_SUBMISSION_LOCK"
test -s "$control/COMPLETION_AUTHORIZATION.json"
test -s "$control/COMPLETION_SUBMISSION_ATTEMPT.tsv"
test -s "$control/COMPLETION_SUBMITTED"
test -s "$control/COMPLETION_RELEASE_AUTHORIZED"
for _release_wait in $(seq 1 60); do
  [[ -s $control/COMPLETION_RELEASED ]] && break
  sleep 1
done
test -s "$control/COMPLETION_RELEASED"
test ! -e "$control/COMPLETION_INVOCATION_LOCK"
test ! -e "$control/COMPLETION_RESULT.json"
test ! -e "$control/COMPLETION_STOPPED"
test ! -e "$generation_root/completion"
test ! -e "$gate_output/assembled"

module load gcc/13.4.0 cuda/12.9.1
module load conda/Miniforge3-25.3.1-3
conda activate "$root/envs/subliminal-mitigate-py311"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-kalai-s3-completion-v1-pyc-$SLURM_JOB_ID
export DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export XDG_CACHE_HOME=$root/cache XDG_CONFIG_HOME=$root/config TMPDIR=$root/tmp
export HF_HOME=$root/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRITON_CACHE_DIR=$root/cache/triton
mkdir -p "$XDG_CONFIG_HOME" "$TMPDIR" "$HUGGINGFACE_HUB_CACHE" "$TRITON_CACHE_DIR"

cd "$controller_repo"
python "$authorizer" verify \
  --controller-output "$controller_output" \
  --controller-repo "$controller_repo" \
  --gate-output "$gate_output" \
  --gate-repo "$gate_repo" \
  --expected-job-id "$SLURM_JOB_ID"

mkdir "$control/COMPLETION_INVOCATION_LOCK"
owner_tmp=$control/COMPLETION_INVOCATION_LOCK/owner.tmp.$SLURM_JOB_ID
printf 'controller_protocol_id=massive_medical_kalai_s3_r20_v2_completion_v1\nstage=completion\njob_id=%s\nrestart_or_resume_authorized=false\nretry_authorized=false\n' \
  "$SLURM_JOB_ID" > "$owner_tmp"
chmod 0400 "$owner_tmp"
mv "$owner_tmp" "$control/COMPLETION_INVOCATION_LOCK/owner"

python "$runner" \
  --source-protocol-manifest "$source_protocol" \
  --output-root "$generation_root" \
  --gate-result "$control/GATE_RESULT.json" \
  --gpu-authorization "$control/COMPLETION_AUTHORIZATION.json" \
  --device cuda:0
for phase in benefit medical; do
  python "$sampler" \
    --source-protocol-manifest "$source_protocol" \
    --output-root "$generation_root" \
    --phase "$phase" \
    --stage completion \
    --gate-result "$control/GATE_RESULT.json" \
    --device cuda:0 \
    --audit-only
done
python "$assembler" \
  --workflow-root "$gate_output" \
  --source-protocol-manifest "$source_protocol"
python "$evaluator" \
  --controller-output "$controller_output" \
  --controller-repo "$controller_repo" \
  --gate-output "$gate_output" \
  --gate-repo "$gate_repo" \
  --source-protocol-manifest "$source_protocol" \
  --authorization "$control/COMPLETION_AUTHORIZATION.json" \
  --elapsed-seconds "$((SECONDS - started))" \
  --slurm-job-id "$SLURM_JOB_ID"
completed=true
trap - EXIT
echo KALAI_S3_COMPLETION_V1_COMPLETE_NO_API_NO_AUTOMATIC_CONTINUATION
