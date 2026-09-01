#!/bin/bash
#SBATCH --job-name=mmu_kalai_s1_gate
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:35:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/massive_medical_kalai_s1_r20_trace_reuse_v1_gate_%j.out
#SBATCH --error=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/massive_medical_kalai_s1_r20_trace_reuse_v1_gate_%j.err

set -euo pipefail
umask 077
ulimit -c 0

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-kalai-s1-r20-trace-reuse-v1
output=$root/outputs/massive_medical_kalai_s1_r20_trace_reuse_v1
control=$output/control
controller=$repo/scripts/sample_massive_medical_kalai_s1_trace_reuse_v1.py
authorizer=$repo/scripts/authorize_massive_medical_kalai_s1_trace_reuse_v1.py
evaluator=$repo/scripts/evaluate_massive_medical_kalai_s1_trace_reuse_gate_v1.py
started=$SECONDS
completed=false

on_exit() {
  code=$?
  if [[ $completed != true && ! -e $control/TECHNICAL_GATE_RESULT.json && ! -e $control/TECHNICAL_GATE_STOPPED ]]; then
    temporary=$control/TECHNICAL_GATE_STOPPED.tmp.$SLURM_JOB_ID
    printf 'stage=technical_gate\njob_id=%s\nexit_code=%s\nrestart_or_resume_authorized=false\ncompletion_authorized=false\n' \
      "$SLURM_JOB_ID" "$code" > "$temporary"
    chmod 0400 "$temporary"
    mv "$temporary" "$control/TECHNICAL_GATE_STOPPED"
  fi
}

[[ -z ${OPENAI_API_KEY:-} ]] || { echo 'OPENAI_API_KEY must not reach a GPU job.' >&2; exit 10; }
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY
unset TRANSFORMERS_CACHE

test -d "$control/TECHNICAL_GATE_SUBMISSION_LOCK"
grep -Fx 'stage=technical_gate' "$control/TECHNICAL_GATE_SUBMITTED" >/dev/null
grep -Fx "job_id=$SLURM_JOB_ID" "$control/TECHNICAL_GATE_SUBMITTED" >/dev/null
grep -Fx 'held_first=true' "$control/TECHNICAL_GATE_SUBMITTED" >/dev/null
grep -Fx 'held_audit_passed=true' "$control/TECHNICAL_GATE_SUBMITTED" >/dev/null
grep -Fx "job_id=$SLURM_JOB_ID" "$control/TECHNICAL_GATE_RELEASE_AUTHORIZED" >/dev/null
grep -Fx 'release_authorized=true' "$control/TECHNICAL_GATE_RELEASE_AUTHORIZED" >/dev/null
mkdir "$control/TECHNICAL_GATE_INVOCATION_LOCK"
owner=$control/TECHNICAL_GATE_INVOCATION_LOCK/owner
owner_tmp=$owner.tmp.$SLURM_JOB_ID
printf 'stage=technical_gate\njob_id=%s\nrestart_or_resume_authorized=false\nretry_authorized=false\ncompletion_authorized=false\n' \
  "$SLURM_JOB_ID" > "$owner_tmp"
chmod 0400 "$owner_tmp"
mv "$owner_tmp" "$owner"
trap on_exit EXIT

# Slurm may dispatch immediately after ``scontrol release`` and before the
# submitter has atomically published its post-release receipt.  Install the
# terminal-failure trap first, then wait briefly for that receipt rather than
# racing it.
for _release_wait in $(seq 1 60); do
  [[ -s $control/TECHNICAL_GATE_RELEASED ]] && break
  sleep 1
done
test -s "$control/TECHNICAL_GATE_RELEASED"
grep -Fx "job_id=$SLURM_JOB_ID" "$control/TECHNICAL_GATE_RELEASED" >/dev/null
grep -Fx 'released=true' "$control/TECHNICAL_GATE_RELEASED" >/dev/null

test ! -e "$output/generation"
test ! -e "$output/assembled"
test ! -e "$control/TECHNICAL_GATE_RESULT.json"
test ! -e "$control/TECHNICAL_GATE_STOPPED"
test ! -e "$control/COMPLETION_AUTHORIZATION.json"

module load gcc/13.4.0 cuda/12.9.1
module load conda/Miniforge3-25.3.1-3
conda activate "$root/envs/subliminal-mitigate-py311"
export PYTHONPATH=$repo
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-kalai-s1-gate-pyc-$SLURM_JOB_ID
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
  --output-root "$output" \
  --repo-root "$repo"
python "$controller" \
  --replay-plan "$control/REPLAY_PLAN.json" \
  --output-root "$output" \
  --repo-root "$repo" \
  --stage technical_gate \
  --gpu-authorization "$control/TECHNICAL_GATE_AUTHORIZATION.json" \
  --device cuda:0
python "$controller" \
  --replay-plan "$control/REPLAY_PLAN.json" \
  --output-root "$output" \
  --stage technical_gate \
  --audit-only
python "$evaluator" \
  --output-root "$output" \
  --repo-root "$repo" \
  --replay-plan "$control/REPLAY_PLAN.json" \
  --authorization "$control/TECHNICAL_GATE_AUTHORIZATION.json" \
  --elapsed-seconds "$((SECONDS - started))" \
  --slurm-job-id "$SLURM_JOB_ID"
completed=true
trap - EXIT
echo MASSIVE_MEDICAL_KALAI_S1_TRACE_REUSE_TECHNICAL_GATE_COMPLETE_NO_AUTOMATIC_CONTINUATION
