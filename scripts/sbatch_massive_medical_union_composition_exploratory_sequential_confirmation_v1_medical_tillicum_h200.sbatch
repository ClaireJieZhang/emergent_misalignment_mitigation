#!/bin/bash
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:35:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/massive_medical_union_composition_exploratory_sequential_confirmation_v1_medical_%j.out
#SBATCH --error=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/massive_medical_union_composition_exploratory_sequential_confirmation_v1_medical_%j.err

set -euo pipefail
umask 077
ulimit -c 0
root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-composition-exploratory-sequential-confirmation-v1
output=$root/outputs/massive_medical_union_composition_exploratory_sequential_confirmation_v1
protocol=$output/protocol; generation=$output/generation; evaluation=$output/evaluation; control=$output/control
auditor=$repo/scripts/audit_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py
sampler=$repo/scripts/sample_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py
summarizer=$repo/scripts/summarize_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py
result=$control/MEDICAL_RESULT.json; stop=$control/STOPPED_medical; completed=false
on_exit() {
  code=$?
  if [[ $completed != true && ! -e $result && ! -e $stop ]]; then
    temporary=$stop.tmp.$SLURM_JOB_ID
    printf 'workflow_id=massive_medical_union_composition_exploratory_sequential_confirmation_v1\nstage=medical\njob_id=%s\nexit_code=%s\nretry_authorized=false\n' "$SLURM_JOB_ID" "$code" > "$temporary"
    chmod 0400 "$temporary"; mv "$temporary" "$stop"
  fi
}
trap on_exit EXIT
[[ -z ${OPENAI_API_KEY:-} ]] || { echo 'OPENAI_API_KEY must not reach a GPU job.' >&2; exit 10; }
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY
unset TRANSFORMERS_CACHE
module load gcc/13.4.0 cuda/12.9.1
module load conda/Miniforge3-25.3.1-3
conda activate "$root/envs/subliminal-mitigate-py311"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=$root/tmp/mmu-seq-medical-pyc-$SLURM_JOB_ID
export DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True XDG_CACHE_HOME=$root/cache XDG_CONFIG_HOME=$root/config TMPDIR=$root/tmp
export HF_HOME=$root/cache/huggingface HUGGINGFACE_HUB_CACHE=$HF_HOME/hub TRITON_CACHE_DIR=$root/cache/triton
mkdir -p "$XDG_CONFIG_HOME" "$TMPDIR" "$HUGGINGFACE_HUB_CACHE" "$TRITON_CACHE_DIR"
cd "$repo"
python "$auditor" verify-job --stage medical --job-id "$SLURM_JOB_ID"
python "$auditor" audit-preflight --stage medical
python "$sampler" --phase medical --protocol-manifest "$protocol/manifest.json" --output-root "$generation" --device cuda:0
python "$sampler" --phase medical --protocol-manifest "$protocol/manifest.json" --output-root "$generation" --device cuda:0 --audit-only
mkdir -p "$evaluation/medical/prejudge"
set +e
python "$summarizer" medical-prejudge --protocol-manifest "$protocol/manifest.json" \
  --benefit-gate "$evaluation/benefit/gate/EXPLORATORY_BENEFIT_PASSED" \
  --medical-generation "ordinary_quorum_m4_q3=$generation/medical/ordinary_quorum_m4_q3/medical/generation.json" \
  --medical-generation "ordinary_min_m4_q4=$generation/medical/ordinary_min_m4_q4/medical/generation.json" \
  --medical-generation "delta_min_m4_q4=$generation/medical/delta_min_m4_q4/medical/generation.json" \
  --timings-file "$generation/medical/timings.json" \
  --output-dir "$evaluation/medical/prejudge"
gate_code=$?
set -e
[[ $gate_code -eq 0 || $gate_code -eq 2 ]] || exit "$gate_code"
python "$auditor" write-result --stage medical --job-id "$SLURM_JOB_ID"
completed=true; trap - EXIT
echo MEDICAL_JOB_SEALED
