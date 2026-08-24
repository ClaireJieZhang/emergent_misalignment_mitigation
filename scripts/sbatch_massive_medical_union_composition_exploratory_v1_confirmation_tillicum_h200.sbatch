#!/bin/bash
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:40:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/massive_medical_union_composition_exploratory_v1_confirmation_%j.out
#SBATCH --error=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/massive_medical_union_composition_exploratory_v1_confirmation_%j.err

set -euo pipefail
umask 077
ulimit -c 0

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-composition-exploratory-v1
output=$root/outputs/massive_medical_union_composition_exploratory_v1
protocol=$output/protocol
generation=$output/generation
evaluation=$output/evaluation
control=$output/control
env_root=$root/envs/subliminal-mitigate-py311
auditor=$repo/scripts/audit_massive_medical_union_composition_exploratory_workflow_v1.py
sampler=$repo/scripts/sample_massive_medical_union_composition_exploratory_v1.py
summarizer=$repo/scripts/summarize_massive_medical_union_composition_exploratory_v1.py
stop=$control/STOPPED_confirmation
result=$control/CONFIRMATION_RESULT.json
completed=false

on_exit() {
  code=$?
  if [[ $completed != true && ! -e $result && ! -e $stop ]]; then
    temporary=$stop.tmp.$SLURM_JOB_ID
    printf 'workflow_id=massive_medical_union_composition_exploratory_workflow_v1\nstage=confirmation\njob_id=%s\nexit_code=%s\nretry_authorized=false\n' \
      "$SLURM_JOB_ID" "$code" > "$temporary"
    chmod 0400 "$temporary"
    mv "$temporary" "$stop"
  fi
}
trap on_exit EXIT

[[ -z ${OPENAI_API_KEY:-} ]] || {
  echo 'OPENAI_API_KEY must not reach the GPU job.' >&2
  exit 10
}
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY
module load gcc/13.4.0 cuda/12.9.1
module load conda/Miniforge3-25.3.1-3
conda activate "$env_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-composition-exploratory-v1-confirmation-pyc-$SLURM_JOB_ID
export DO_NOT_TRACK=1
export HF_HUB_DISABLE_TELEMETRY=1
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
export TRANSFORMERS_CACHE=$HF_HOME
export TRITON_CACHE_DIR=$root/cache/triton
mkdir -p "$XDG_CONFIG_HOME" "$TMPDIR" "$HUGGINGFACE_HUB_CACHE" "$TRITON_CACHE_DIR"

cd "$repo"
python "$auditor" verify-job --stage confirmation --job-id "$SLURM_JOB_ID"

python "$auditor" audit-preflight --stage confirmation

python "$sampler" --phase confirmation --protocol-manifest "$protocol/manifest.json" \
  --output-root "$generation" --device cuda:0
python "$sampler" --phase confirmation --protocol-manifest "$protocol/manifest.json" \
  --output-root "$generation" --device cuda:0 --audit-only

mkdir -p "$evaluation/confirmation/scores" "$evaluation/confirmation/prejudge"
methods=(pi_base ordinary_quorum_m4_q3 ordinary_min_m4_q4 delta_min_m4_q4)
for method in "${methods[@]}"; do
  python "$summarizer" score \
    --protocol-manifest "$protocol/manifest.json" \
    --phase confirmation --method-id "$method" \
    --answers-file "$protocol/confirmation/answers.json" \
    --generations-file "$generation/confirmation/$method/massive/generation.json" \
    --output-file "$evaluation/confirmation/scores/$method.json"
done

set +e
python "$summarizer" confirmation-prejudge \
  --protocol-manifest "$protocol/manifest.json" \
  --smoke-gate "$evaluation/smoke/gate/EXPLORATORY_SMOKE_PASSED" \
  --base-score "$evaluation/confirmation/scores/pi_base.json" \
  --method-score "ordinary_quorum_m4_q3=$evaluation/confirmation/scores/ordinary_quorum_m4_q3.json" \
  --method-score "ordinary_min_m4_q4=$evaluation/confirmation/scores/ordinary_min_m4_q4.json" \
  --method-score "delta_min_m4_q4=$evaluation/confirmation/scores/delta_min_m4_q4.json" \
  --direct-comparator "pi_base=$protocol/direct_confirmation/pi_base.json" \
  --direct-comparator "pi_A=$protocol/direct_confirmation/pi_A.json" \
  --direct-comparator "pi_B1=$protocol/direct_confirmation/pi_B1.json" \
  --direct-comparator "pi_B2=$protocol/direct_confirmation/pi_B2.json" \
  --direct-comparator "pi_B3=$protocol/direct_confirmation/pi_B3.json" \
  --medical-generation "ordinary_quorum_m4_q3=$generation/confirmation/ordinary_quorum_m4_q3/medical/generation.json" \
  --medical-generation "ordinary_min_m4_q4=$generation/confirmation/ordinary_min_m4_q4/medical/generation.json" \
  --medical-generation "delta_min_m4_q4=$generation/confirmation/delta_min_m4_q4/medical/generation.json" \
  --output-dir "$evaluation/confirmation/prejudge"
gate_code=$?
set -e
[[ $gate_code -eq 0 || $gate_code -eq 2 ]] || exit "$gate_code"

python "$auditor" write-result --stage confirmation --job-id "$SLURM_JOB_ID"
completed=true
trap - EXIT
echo 'CONFIRMATION_JOB_SEALED'
