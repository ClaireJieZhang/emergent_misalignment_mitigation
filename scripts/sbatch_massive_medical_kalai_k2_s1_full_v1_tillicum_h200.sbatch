#!/bin/bash
# Phase-isolated full-bank A/B1 Kalai k=2,s=1,R=20 diagnostic.
# Array index 0 is MASSIVE benefit; index 1 is official16x5 medical.

#SBATCH --job-name=mmu_k2s1_full
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --no-requeue
#SBATCH --output=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/massive_medical_kalai_k2_s1_full_v1_%A_%a.out
#SBATCH --error=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/massive_medical_kalai_k2_s1_full_v1_%A_%a.err

set -euo pipefail
umask 077
ulimit -c 0

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-panel-diagnostics-v1
output=$root/outputs/massive_medical_kalai_k2_s1_r20_v1
control=$output/control
source_protocol=$root/outputs/massive_medical_union_composition_exploratory_sequential_confirmation_v1_submit_recovery_v3/protocol/manifest.json
sampler=$repo/scripts/sample_massive_medical_whole_output_consensus_k2_s1_v1.py
case ${SLURM_ARRAY_TASK_ID:?full run must be submitted as an explicit array element} in
  0) phase=benefit; requested=360 ;;
  1) phase=medical; requested=80 ;;
  *) echo 'Unexpected array index.' >&2; exit 2 ;;
esac
phase_upper=${phase^^}
authorization=$control/FULL_${phase_upper}_AUTHORIZATION
submitted=$control/FULL_${phase_upper}_SUBMITTED
release_authorized=$control/FULL_${phase_upper}_RELEASE_AUTHORIZED
released_receipt=$control/FULL_${phase_upper}_RELEASED
completed=false

on_exit() {
  code=$?
  stopped=$control/FULL_${phase_upper}_STOPPED
  if [[ $completed != true && ! -e $control/FULL_${phase_upper}_COMPLETE && ! -e $stopped ]]; then
    temporary=$stopped.tmp.$SLURM_JOB_ID
    printf 'protocol_id=massive_medical_kalai_k2_s1_r20_v1\nstage=full\nphase=%s\njob_id=%s\nexit_code=%s\nrestart_resume_retry_authorized=false\nexternal_api_calls_authorized=0\n' \
      "$phase" "$SLURM_JOB_ID" "$code" > "$temporary"
    chmod 0400 "$temporary"
    mv "$temporary" "$stopped"
  fi
}

[[ -z ${OPENAI_API_KEY:-} ]] || { echo 'OPENAI_API_KEY must not reach this generation job.' >&2; exit 10; }
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY
unset TRANSFORMERS_CACHE

test -s "$source_protocol"
test -x "$sampler"
test -z "$(git -C "$repo" status --porcelain)"
test -s "$control/CPU_STAGE"
grep -Fx 'protocol_id=massive_medical_kalai_k2_s1_r20_v1' "$control/CPU_STAGE" >/dev/null
grep -Fx 'status=CPU_STAGED_NO_GPU_OR_API_AUTHORITY' "$control/CPU_STAGE" >/dev/null
grep -Fx "repository_commit=$(git -C "$repo" rev-parse HEAD)" "$control/CPU_STAGE" >/dev/null
grep -Fx 'full_run_authorized=false' "$control/CPU_STAGE" >/dev/null
grep -Fx 'external_api_calls_authorized=0' "$control/CPU_STAGE" >/dev/null

test -s "$control/SMOKE_COMPLETE"
grep -Fx 'protocol_id=massive_medical_kalai_k2_s1_r20_v1' "$control/SMOKE_COMPLETE" >/dev/null
grep -Fx 'stage=smoke' "$control/SMOKE_COMPLETE" >/dev/null
grep -Fx 'benefit_requested=2' "$control/SMOKE_COMPLETE" >/dev/null
grep -Fx 'medical_requested=16' "$control/SMOKE_COMPLETE" >/dev/null
grep -Fx 'full_run_authorized=false' "$control/SMOKE_COMPLETE" >/dev/null
grep -Fx 'external_api_calls=0' "$control/SMOKE_COMPLETE" >/dev/null

# Smoke completion is evidence, never full-run authority. A future, separately
# reviewed held-first submitter must create all four phase-specific receipts.
test -s "$authorization"
grep -Fx 'protocol_id=massive_medical_kalai_k2_s1_r20_v1' "$authorization" >/dev/null
grep -Fx 'stage=full' "$authorization" >/dev/null
grep -Fx "phase=$phase" "$authorization" >/dev/null
grep -Fx "repository_commit=$(git -C "$repo" rev-parse HEAD)" "$authorization" >/dev/null
grep -Fx 'smoke_reviewed=true' "$authorization" >/dev/null
grep -Fx 'full_run_authorized=true' "$authorization" >/dev/null
grep -Fx 'maximum_gpu_jobs=1' "$authorization" >/dev/null
grep -Fx 'h200_usd_per_hour=0.90' "$authorization" >/dev/null
grep -Fx 'external_api_calls_authorized=0' "$authorization" >/dev/null
grep -Fx 'restart_resume_retry_replacement_requeue_authorized=false' "$authorization" >/dev/null
test "$(grep -c '^maximum_h200_minutes=' "$authorization")" = 1
test "$(grep -c '^maximum_cost_usd=' "$authorization")" = 1
authorized_minutes=$(sed -n 's/^maximum_h200_minutes=//p' "$authorization")
authorized_cost=$(sed -n 's/^maximum_cost_usd=//p' "$authorization")
[[ $authorized_minutes =~ ^[1-9][0-9]*$ ]]
[[ $authorized_cost =~ ^[0-9]+([.][0-9]+)?$ ]]

# Read the scheduler's effective limit rather than trusting a submitted
# environment variable. `scontrol -o` emits [days-]hours:minutes:seconds.
slurm_time_limit=$(scontrol show job "$SLURM_JOB_ID" -o | tr ' ' '\n' | sed -n 's/^TimeLimit=//p')
[[ $slurm_time_limit =~ ^([0-9]+-)?[0-9]+:[0-9][0-9]:[0-9][0-9]$ ]]
duration=$slurm_time_limit
days=0
if [[ $duration == *-* ]]; then
  days=${duration%%-*}
  duration=${duration#*-}
fi
IFS=: read -r hours minutes seconds <<< "$duration"
[[ $days =~ ^[0-9]+$ && $hours =~ ^[0-9]+$ ]]
[[ $minutes =~ ^[0-5][0-9]$ && $seconds =~ ^[0-5][0-9]$ ]]
slurm_time_limit_minutes=$((days * 1440 + hours * 60 + minutes + (seconds > 0)))
(( slurm_time_limit_minutes > 0 ))
(( slurm_time_limit_minutes <= authorized_minutes ))
awk -v minutes="$slurm_time_limit_minutes" -v cap="$authorized_cost" \
  'BEGIN { exit !((minutes * 0.90 / 60.0) <= cap + 1e-12) }'

test -s "$submitted"
grep -Fx 'protocol_id=massive_medical_kalai_k2_s1_r20_v1' "$submitted" >/dev/null
grep -Fx 'stage=full' "$submitted" >/dev/null
grep -Fx "phase=$phase" "$submitted" >/dev/null
grep -Fx "array_task_id=$SLURM_ARRAY_TASK_ID" "$submitted" >/dev/null
grep -Fx "job_id=$SLURM_JOB_ID" "$submitted" >/dev/null
grep -Fx 'held_first=true' "$submitted" >/dev/null
grep -Fx 'held_audit_passed=true' "$submitted" >/dev/null
grep -Fx "slurm_time_limit=$slurm_time_limit" "$submitted" >/dev/null
grep -Fx "slurm_time_limit_minutes=$slurm_time_limit_minutes" "$submitted" >/dev/null
grep -Fx "repository_commit=$(git -C "$repo" rev-parse HEAD)" "$submitted" >/dev/null
grep -Fx 'restart_resume_retry_authorized=false' "$submitted" >/dev/null

test -s "$release_authorized"
grep -Fx 'protocol_id=massive_medical_kalai_k2_s1_r20_v1' "$release_authorized" >/dev/null
grep -Fx 'stage=full' "$release_authorized" >/dev/null
grep -Fx "phase=$phase" "$release_authorized" >/dev/null
grep -Fx "array_task_id=$SLURM_ARRAY_TASK_ID" "$release_authorized" >/dev/null
grep -Fx "job_id=$SLURM_JOB_ID" "$release_authorized" >/dev/null
grep -Fx 'held_audit_passed=true' "$release_authorized" >/dev/null
grep -Fx 'release_authorized=true' "$release_authorized" >/dev/null
grep -Fx 'restart_resume_retry_authorized=false' "$release_authorized" >/dev/null

mkdir "$control/FULL_${phase_upper}_INVOCATION_LOCK"
test ! -e "$control/FULL_${phase_upper}_COMPLETE"
test ! -e "$control/FULL_${phase_upper}_STOPPED"
test ! -e "$output/generation/full/$phase"
trap on_exit EXIT

for _ in $(seq 1 60); do
  [[ -s $released_receipt ]] && break
  sleep 1
done
test -s "$released_receipt"
grep -Fx 'protocol_id=massive_medical_kalai_k2_s1_r20_v1' "$released_receipt" >/dev/null
grep -Fx 'stage=full' "$released_receipt" >/dev/null
grep -Fx "phase=$phase" "$released_receipt" >/dev/null
grep -Fx "array_task_id=$SLURM_ARRAY_TASK_ID" "$released_receipt" >/dev/null
grep -Fx "job_id=$SLURM_JOB_ID" "$released_receipt" >/dev/null
grep -Fx 'released=true' "$released_receipt" >/dev/null
grep -Fx 'restart_resume_retry_authorized=false' "$released_receipt" >/dev/null

module load gcc/13.4.0 cuda/12.9.1
module load conda/Miniforge3-25.3.1-3
conda activate "$root/envs/subliminal-mitigate-py311"
export PYTHONPATH=$repo
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-k2s1-full-$phase-pyc-$SLURM_JOB_ID
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
python "$sampler" \
  --source-protocol-manifest "$source_protocol" \
  --output-root "$output/generation" \
  --phase "$phase" \
  --stage full \
  --device cuda:0
python "$sampler" \
  --source-protocol-manifest "$source_protocol" \
  --output-root "$output/generation" \
  --phase "$phase" \
  --stage full \
  --device cuda:0 \
  --audit-only

python -c 'import json,sys; d=json.load(open(sys.argv[1],encoding="utf-8")); s=d["summary"]; print(json.dumps({k:s[k] for k in ("requested_n","accepted_n","abstained_n","coverage","abstention_rate","total_attempts","total_candidate_sampled_tokens")},sort_keys=True))' \
  "$output/generation/full/$phase/generation.json"

complete=$control/FULL_${phase_upper}_COMPLETE
temporary=$complete.tmp.$SLURM_JOB_ID
printf 'protocol_id=massive_medical_kalai_k2_s1_r20_v1\nstage=full\nphase=%s\njob_id=%s\nelapsed_seconds=%s\nrequested_outputs=%s\nmaximum_attempts_per_request=20\nexternal_api_calls=0\n' \
  "$phase" "$SLURM_JOB_ID" "$((SECONDS - started))" "$requested" > "$temporary"
chmod 0400 "$temporary"
mv "$temporary" "$complete"
completed=true
trap - EXIT
echo "MASSIVE_MEDICAL_KALAI_K2_S1_R20_V1_FULL_${phase_upper}_COMPLETE"
