#!/bin/bash
# Two MASSIVE plus one-per-medical-prompt smoke for the A/B1 diagnostic.

#SBATCH --job-name=mmu_k2s1_smoke
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/massive_medical_kalai_k2_s1_smoke_v1_%j.out
#SBATCH --error=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/massive_medical_kalai_k2_s1_smoke_v1_%j.err

set -euo pipefail
umask 077
ulimit -c 0

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-panel-diagnostics-v1
output=$root/outputs/massive_medical_kalai_k2_s1_r20_v1
control=$output/control
source_protocol=$root/outputs/massive_medical_union_composition_exploratory_sequential_confirmation_v1_submit_recovery_v3/protocol/manifest.json
sampler=$repo/scripts/sample_massive_medical_whole_output_consensus_k2_s1_v1.py
completed=false

on_exit() {
  code=$?
  if [[ $completed != true && ! -e $control/SMOKE_COMPLETE && ! -e $control/SMOKE_STOPPED ]]; then
    temporary=$control/SMOKE_STOPPED.tmp.$SLURM_JOB_ID
    printf 'protocol_id=massive_medical_kalai_k2_s1_r20_v1\nstage=smoke\njob_id=%s\nexit_code=%s\nrestart_resume_retry_authorized=false\nfull_run_authorized=false\nexternal_api_calls_authorized=0\n' \
      "$SLURM_JOB_ID" "$code" > "$temporary"
    chmod 0400 "$temporary"
    mv "$temporary" "$control/SMOKE_STOPPED"
  fi
}

[[ -z ${OPENAI_API_KEY:-} ]] || { echo 'OPENAI_API_KEY must not reach this generation job.' >&2; exit 10; }
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY
unset TRANSFORMERS_CACHE

test -s "$source_protocol"
test -x "$sampler"
test -z "$(git -C "$repo" status --porcelain)"
test -d "$control"
test -s "$control/CPU_STAGE"
grep -Fx 'protocol_id=massive_medical_kalai_k2_s1_r20_v1' "$control/CPU_STAGE" >/dev/null
grep -Fx 'status=CPU_STAGED_NO_GPU_OR_API_AUTHORITY' "$control/CPU_STAGE" >/dev/null
grep -Fx "repository_commit=$(git -C "$repo" rev-parse HEAD)" "$control/CPU_STAGE" >/dev/null
grep -Fx 'smoke_benefit_requests=2' "$control/CPU_STAGE" >/dev/null
grep -Fx 'smoke_medical_requests=16' "$control/CPU_STAGE" >/dev/null
grep -Fx 'smoke_maximum_candidate_attempts=360' "$control/CPU_STAGE" >/dev/null
grep -Fx 'smoke_authorized=false' "$control/CPU_STAGE" >/dev/null
grep -Fx 'full_run_authorized=false' "$control/CPU_STAGE" >/dev/null
grep -Fx 'external_api_calls_authorized=0' "$control/CPU_STAGE" >/dev/null

test -s "$control/SMOKE_AUTHORIZATION"
grep -Fx 'protocol_id=massive_medical_kalai_k2_s1_r20_v1' "$control/SMOKE_AUTHORIZATION" >/dev/null
grep -Fx 'stage=smoke' "$control/SMOKE_AUTHORIZATION" >/dev/null
grep -Fx "repository_commit=$(git -C "$repo" rev-parse HEAD)" "$control/SMOKE_AUTHORIZATION" >/dev/null
grep -Fx 'maximum_h200_minutes=30' "$control/SMOKE_AUTHORIZATION" >/dev/null
grep -Fx 'maximum_cost_usd=0.45' "$control/SMOKE_AUTHORIZATION" >/dev/null
grep -Fx 'maximum_gpu_jobs=1' "$control/SMOKE_AUTHORIZATION" >/dev/null
grep -Fx 'maximum_candidate_attempts=360' "$control/SMOKE_AUTHORIZATION" >/dev/null
grep -Fx 'external_api_calls_authorized=0' "$control/SMOKE_AUTHORIZATION" >/dev/null
grep -Fx 'full_run_authorized=false' "$control/SMOKE_AUTHORIZATION" >/dev/null
grep -Fx 'restart_resume_retry_replacement_requeue_authorized=false' "$control/SMOKE_AUTHORIZATION" >/dev/null

test -s "$control/SMOKE_SUBMITTED"
grep -Fx 'protocol_id=massive_medical_kalai_k2_s1_r20_v1' "$control/SMOKE_SUBMITTED" >/dev/null
grep -Fx 'stage=smoke' "$control/SMOKE_SUBMITTED" >/dev/null
grep -Fx "job_id=$SLURM_JOB_ID" "$control/SMOKE_SUBMITTED" >/dev/null
grep -Fx 'held_first=true' "$control/SMOKE_SUBMITTED" >/dev/null
grep -Fx 'held_audit_passed=true' "$control/SMOKE_SUBMITTED" >/dev/null
grep -Fx "repository_commit=$(git -C "$repo" rev-parse HEAD)" "$control/SMOKE_SUBMITTED" >/dev/null
grep -Fx 'full_run_authorized=false' "$control/SMOKE_SUBMITTED" >/dev/null
grep -Fx 'restart_resume_retry_authorized=false' "$control/SMOKE_SUBMITTED" >/dev/null

test -s "$control/SMOKE_RELEASE_AUTHORIZED"
grep -Fx 'protocol_id=massive_medical_kalai_k2_s1_r20_v1' "$control/SMOKE_RELEASE_AUTHORIZED" >/dev/null
grep -Fx 'stage=smoke' "$control/SMOKE_RELEASE_AUTHORIZED" >/dev/null
grep -Fx "job_id=$SLURM_JOB_ID" "$control/SMOKE_RELEASE_AUTHORIZED" >/dev/null
grep -Fx 'held_audit_passed=true' "$control/SMOKE_RELEASE_AUTHORIZED" >/dev/null
grep -Fx 'release_authorized=true' "$control/SMOKE_RELEASE_AUTHORIZED" >/dev/null
grep -Fx 'full_run_authorized=false' "$control/SMOKE_RELEASE_AUTHORIZED" >/dev/null
grep -Fx 'restart_resume_retry_authorized=false' "$control/SMOKE_RELEASE_AUTHORIZED" >/dev/null

mkdir "$control/SMOKE_INVOCATION_LOCK"
test ! -e "$control/SMOKE_COMPLETE"
test ! -e "$control/SMOKE_STOPPED"
test ! -e "$output/generation/smoke"
trap on_exit EXIT

# A released job can start just before the submit wrapper finishes this final
# receipt. Wait briefly, then bind it to this exact Slurm job before model load.
for _ in $(seq 1 60); do
  [[ -s $control/SMOKE_RELEASED ]] && break
  sleep 1
done
test -s "$control/SMOKE_RELEASED"
grep -Fx 'protocol_id=massive_medical_kalai_k2_s1_r20_v1' "$control/SMOKE_RELEASED" >/dev/null
grep -Fx 'stage=smoke' "$control/SMOKE_RELEASED" >/dev/null
grep -Fx "job_id=$SLURM_JOB_ID" "$control/SMOKE_RELEASED" >/dev/null
grep -Fx 'released=true' "$control/SMOKE_RELEASED" >/dev/null
grep -Fx 'full_run_authorized=false' "$control/SMOKE_RELEASED" >/dev/null
grep -Fx 'restart_resume_retry_authorized=false' "$control/SMOKE_RELEASED" >/dev/null

module load gcc/13.4.0 cuda/12.9.1
module load conda/Miniforge3-25.3.1-3
conda activate "$root/envs/subliminal-mitigate-py311"
export PYTHONPATH=$repo
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-k2s1-smoke-pyc-$SLURM_JOB_ID
export DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export XDG_CACHE_HOME=$root/cache XDG_CONFIG_HOME=$root/config TMPDIR=$root/tmp
export HF_HOME=$root/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRITON_CACHE_DIR=$root/cache/triton
mkdir -p "$XDG_CONFIG_HOME" "$TMPDIR" "$HUGGINGFACE_HUB_CACHE" "$TRITON_CACHE_DIR"

cd "$repo"
started=$SECONDS
for phase in benefit medical; do
  python "$sampler" \
    --source-protocol-manifest "$source_protocol" \
    --output-root "$output/generation" \
    --phase "$phase" \
    --stage smoke \
    --device cuda:0
  python "$sampler" \
    --source-protocol-manifest "$source_protocol" \
    --output-root "$output/generation" \
    --phase "$phase" \
    --stage smoke \
    --device cuda:0 \
    --audit-only
done

python -c 'import json,sys; from pathlib import Path
root=Path(sys.argv[1])
for phase in ("benefit","medical"):
 p=root/"smoke"/phase/"generation.json"
 d=json.load(open(p,encoding="utf-8")); s=d["summary"]
 print(phase, json.dumps({k:s[k] for k in ("requested_n","accepted_n","abstained_n","coverage","abstention_rate","total_attempts","total_candidate_sampled_tokens")},sort_keys=True))' \
  "$output/generation"

temporary=$control/SMOKE_COMPLETE.tmp.$SLURM_JOB_ID
printf 'protocol_id=massive_medical_kalai_k2_s1_r20_v1\nstage=smoke\njob_id=%s\nelapsed_seconds=%s\nbenefit_requested=2\nmedical_requested=16\nmaximum_candidate_attempts=360\nmaximum_h200_minutes=30\nmaximum_cost_usd=0.45\nfull_run_authorized=false\nexternal_api_calls=0\n' \
  "$SLURM_JOB_ID" "$((SECONDS - started))" > "$temporary"
chmod 0400 "$temporary"
mv "$temporary" "$control/SMOKE_COMPLETE"
completed=true
trap - EXIT
echo MASSIVE_MEDICAL_KALAI_K2_S1_R20_V1_SMOKE_COMPLETE_NO_AUTOMATIC_FULL_RUN
