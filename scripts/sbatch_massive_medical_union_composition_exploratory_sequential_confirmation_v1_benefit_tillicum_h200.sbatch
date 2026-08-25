#!/bin/bash
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:05:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/massive_medical_union_composition_exploratory_sequential_confirmation_v1_stage_recovery_v2_benefit_%j.out
#SBATCH --error=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/massive_medical_union_composition_exploratory_sequential_confirmation_v1_stage_recovery_v2_benefit_%j.err

set -euo pipefail
umask 077
ulimit -c 0
root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-composition-exploratory-sequential-confirmation-v1-stage-recovery-v2
output=$root/outputs/massive_medical_union_composition_exploratory_sequential_confirmation_v1_stage_recovery_v2
protocol=$output/protocol; generation=$output/generation; evaluation=$output/evaluation; control=$output/control
auditor=$repo/scripts/audit_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py
sampler=$repo/scripts/sample_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py
summarizer=$repo/scripts/summarize_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py
result=$control/BENEFIT_RESULT.json; stop=$control/STOPPED_benefit; completed=false
on_exit() {
  code=$?
  if [[ $completed != true && ! -e $result && ! -e $stop ]]; then
    temporary=$stop.tmp.$SLURM_JOB_ID
    printf 'workflow_id=massive_medical_union_composition_exploratory_sequential_confirmation_v1\nstage=benefit\njob_id=%s\nexit_code=%s\nretry_authorized=false\n' "$SLURM_JOB_ID" "$code" > "$temporary"
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
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=$root/tmp/mmu-seq-benefit-v2-pyc-$SLURM_JOB_ID
export DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True XDG_CACHE_HOME=$root/cache XDG_CONFIG_HOME=$root/config TMPDIR=$root/tmp
export HF_HOME=$root/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRITON_CACHE_DIR=$root/cache/triton
mkdir -p "$XDG_CONFIG_HOME" "$TMPDIR" "$HUGGINGFACE_HUB_CACHE" "$TRITON_CACHE_DIR"
cd "$repo"
python "$auditor" verify-job --stage benefit --job-id "$SLURM_JOB_ID"
python "$auditor" audit-preflight --stage benefit
python "$sampler" --phase benefit --protocol-manifest "$protocol/manifest.json" --output-root "$generation" --device cuda:0
python "$sampler" --phase benefit --protocol-manifest "$protocol/manifest.json" --output-root "$generation" --device cuda:0 --audit-only
mkdir -p "$evaluation/benefit/scores" "$evaluation/benefit/gate"
methods=(pi_base ordinary_quorum_m4_q3 ordinary_min_m4_q4 delta_min_m4_q4)
for method in "${methods[@]}"; do
  python "$summarizer" score --protocol-manifest "$protocol/manifest.json" --method-id "$method" \
    --answers-file "$protocol/benefit/answers.json" --generations-file "$generation/benefit/$method/massive/generation.json" \
    --output-file "$evaluation/benefit/scores/$method.json"
done
set +e
python "$summarizer" benefit-gate --protocol-manifest "$protocol/manifest.json" \
  --base-score "$evaluation/benefit/scores/pi_base.json" \
  --method-score "ordinary_quorum_m4_q3=$evaluation/benefit/scores/ordinary_quorum_m4_q3.json" \
  --method-score "ordinary_min_m4_q4=$evaluation/benefit/scores/ordinary_min_m4_q4.json" \
  --method-score "delta_min_m4_q4=$evaluation/benefit/scores/delta_min_m4_q4.json" \
  --direct-comparator "pi_base=$protocol/direct_benefit/pi_base.json" \
  --direct-comparator "pi_A=$protocol/direct_benefit/pi_A.json" \
  --direct-comparator "pi_B1=$protocol/direct_benefit/pi_B1.json" \
  --direct-comparator "pi_B2=$protocol/direct_benefit/pi_B2.json" \
  --direct-comparator "pi_B3=$protocol/direct_benefit/pi_B3.json" \
  --timings-file "$generation/benefit/timings.json" --output-dir "$evaluation/benefit/gate"
gate_code=$?
set -e
[[ $gate_code -eq 0 || $gate_code -eq 2 ]] || exit "$gate_code"
python "$auditor" write-result --stage benefit --job-id "$SLURM_JOB_ID"
completed=true; trap - EXIT
echo BENEFIT_JOB_SEALED
