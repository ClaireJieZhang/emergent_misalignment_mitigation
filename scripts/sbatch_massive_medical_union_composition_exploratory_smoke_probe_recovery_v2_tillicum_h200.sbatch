#!/bin/bash
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/massive_medical_union_composition_exploratory_smoke_probe_recovery_v2_%j.out
#SBATCH --error=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/massive_medical_union_composition_exploratory_smoke_probe_recovery_v2_%j.err

set -euo pipefail
umask 077
ulimit -c 0

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-composition-exploratory-smoke-probe-recovery-v2
source_output=$root/outputs/massive_medical_union_composition_exploratory_v1
output=$root/outputs/massive_medical_union_composition_exploratory_smoke_probe_recovery_v2
generation=$output/generation
evaluation=$output/evaluation
control=$output/control
env_root=$root/envs/subliminal-mitigate-py311
auditor=$repo/scripts/audit_massive_medical_union_composition_exploratory_smoke_probe_recovery_v2.py
sampler=$repo/scripts/sample_massive_medical_union_composition_exploratory_v1.py
summarizer=$repo/scripts/summarize_massive_medical_union_composition_exploratory_v1.py
stop=$control/STOPPED_smoke_probe_recovery
result=$control/SMOKE_PROBE_RECOVERY_RESULT.json
completed=false

on_exit() {
  code=$?
  if [[ $completed != true && ! -e $result && ! -e $stop ]]; then
    temporary=$stop.tmp.$SLURM_JOB_ID
    printf 'workflow_id=massive_medical_union_composition_exploratory_workflow_smoke_probe_recovery_v2\nstage=smoke_probe_recovery\njob_id=%s\nexit_code=%s\nretry_authorized=false\nconfirmation_submitted=false\nexternal_api_calls=0\n' \
      "$SLURM_JOB_ID" "$code" > "$temporary"
    chmod 0400 "$temporary"
    mv "$temporary" "$stop"
  fi
}
trap on_exit EXIT

[[ -z ${OPENAI_API_KEY:-} ]] || {
  echo 'OPENAI_API_KEY must not reach the recovery GPU job.' >&2
  exit 10
}
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY
unset TRANSFORMERS_CACHE
module load gcc/13.4.0 cuda/12.9.1
module load conda/Miniforge3-25.3.1-3
conda activate "$env_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-composition-smoke-probe-recovery-v2-pyc-$SLURM_JOB_ID
export DO_NOT_TRACK=1
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export XDG_CACHE_HOME=$root/cache
export XDG_CONFIG_HOME=$root/config
export TMPDIR=$root/tmp
export HF_HOME=$root/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRITON_CACHE_DIR=$root/cache/triton
mkdir -p "$XDG_CONFIG_HOME" "$TMPDIR" "$HUGGINGFACE_HUB_CACHE" "$TRITON_CACHE_DIR"

cd "$repo"
python "$auditor" verify-job --job-id "$SLURM_JOB_ID"
python "$auditor" audit-preflight

python "$sampler" --phase smoke \
  --protocol-manifest "$source_output/protocol/manifest.json" \
  --output-root "$generation" --device cuda:0
python "$sampler" --phase smoke \
  --protocol-manifest "$source_output/protocol/manifest.json" \
  --output-root "$generation" --device cuda:0 --audit-only

mkdir -p "$evaluation/smoke/scores" "$evaluation/smoke/gate"
methods=(pi_base ordinary_quorum_m4_q3 ordinary_min_m4_q4 delta_min_m4_q4)
for method in "${methods[@]}"; do
  python "$summarizer" score \
    --protocol-manifest "$source_output/protocol/manifest.json" \
    --phase smoke --method-id "$method" \
    --answers-file "$source_output/protocol/smoke/answers.json" \
    --generations-file "$generation/smoke/$method/massive/generation.json" \
    --output-file "$evaluation/smoke/scores/$method.json"
done

set +e
python "$summarizer" smoke \
  --protocol-manifest "$source_output/protocol/manifest.json" \
  --base-score "$evaluation/smoke/scores/pi_base.json" \
  --method-score "ordinary_quorum_m4_q3=$evaluation/smoke/scores/ordinary_quorum_m4_q3.json" \
  --method-score "ordinary_min_m4_q4=$evaluation/smoke/scores/ordinary_min_m4_q4.json" \
  --method-score "delta_min_m4_q4=$evaluation/smoke/scores/delta_min_m4_q4.json" \
  --timings "$generation/smoke/timings.json" \
  --output-dir "$evaluation/smoke/gate"
gate_code=$?
set -e
[[ $gate_code -eq 0 || $gate_code -eq 2 ]] || exit "$gate_code"

python "$auditor" write-result --job-id "$SLURM_JOB_ID"
completed=true
trap - EXIT
echo 'SMOKE_PROBE_RECOVERY_JOB_SEALED'
