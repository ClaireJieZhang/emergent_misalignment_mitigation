#!/bin/bash
#SBATCH --job-name=mmu_base_direct_v1
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/massive_medical_composition_baselines_v1_direct_%j.out
#SBATCH --error=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/massive_medical_composition_baselines_v1_direct_%j.err

set -euo pipefail
umask 077
ulimit -c 0

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-composition-baselines-v1-stage-recovery-v3
output=$root/outputs/massive_medical_composition_baselines_v1
control=$output/control
source_protocol=$root/outputs/massive_medical_union_composition_exploratory_sequential_confirmation_v1_submit_recovery_v3/protocol
union_model=$output/models/pi_union
merge_model=$output/models/pi_merge
union_config=$repo/configs/training_qwen25_7b_massive_medical_composition_baselines_v1_union_sft.yaml
merge_policy=$control/MERGE_POLICY.json
sampler=$repo/scripts/sample_massive_medical_direct_contextual_baseline_v1.py
auth_script=$repo/scripts/authorize_massive_medical_composition_baselines_v1.py
finalizer=$repo/scripts/finalize_massive_medical_composition_baseline_gpu_stage_v1.py
started=$SECONDS
completed=false

on_exit() {
  code=$?
  if [[ $completed != true && ! -e $control/DIRECT_GENERATION_RESULT.json && ! -e $control/DIRECT_GENERATION_STOPPED ]]; then
    temporary=$control/DIRECT_GENERATION_STOPPED.tmp.$SLURM_JOB_ID
    printf 'stage=direct_generation\njob_id=%s\nexit_code=%s\nrestart_or_resume_authorized=false\n' "$SLURM_JOB_ID" "$code" > "$temporary"
    chmod 0400 "$temporary"
    mv "$temporary" "$control/DIRECT_GENERATION_STOPPED"
  fi
}
trap on_exit EXIT

[[ -z ${OPENAI_API_KEY:-} ]] || { echo 'OPENAI_API_KEY must not reach a GPU job.' >&2; exit 10; }
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY
unset TRANSFORMERS_CACHE

mkdir "$control/DIRECT_GENERATION_INVOCATION_LOCK"
test -s "$control/UNION_TRAINING_RESULT.json"
test ! -e "$merge_model"
test ! -e "$control/DIRECT_GENERATION_RESULT.json"
test ! -e "$control/DIRECT_GENERATION_STOPPED"

module load gcc/13.4.0 cuda/12.9.1
module load conda/Miniforge3-25.3.1-3
conda activate "$root/envs/subliminal-mitigate-py311"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-baseline-direct-pyc-$SLURM_JOB_ID
export DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export XDG_CACHE_HOME=$root/cache XDG_CONFIG_HOME=$root/config TMPDIR=$root/tmp
export HF_HOME=$root/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRITON_CACHE_DIR=$root/cache/triton
mkdir -p "$XDG_CONFIG_HOME" "$TMPDIR" "$HUGGINGFACE_HUB_CACHE" "$TRITON_CACHE_DIR"
mkdir -p "$output/generation/direct/pi_union" "$output/generation/direct/pi_merge" "$output/evaluation/benefit"

cd "$repo"
python "$auth_script" verify --stage direct_generation --output-root "$output" --repo-root "$repo"
python scripts/materialize_massive_medical_lora_merge_v1.py --execute --policy-manifest "$merge_policy" --device cuda:0
python scripts/materialize_massive_medical_lora_merge_v1.py --audit-output --policy-manifest "$merge_policy"

for model_id in pi_union pi_merge; do
  if [[ $model_id == pi_union ]]; then
    adapter=$union_model
    manifest=$union_model/MODEL_MANIFEST.json
    training_args=(--training-config "$union_config")
  else
    adapter=$merge_model
    manifest=$merge_model/MODEL_MANIFEST.json
    training_args=()
  fi
  for phase in benefit medical; do
    generation=$output/generation/direct/$model_id/$phase.json
    python "$sampler" \
      --source-protocol-manifest "$source_protocol/manifest.json" \
      --model-id "$model_id" \
      --adapter-path "$adapter" \
      --adapter-manifest "$manifest" \
      "${training_args[@]}" \
      --phase "$phase" \
      --output-file "$generation" \
      --device cuda:0
    python "$sampler" \
      --source-protocol-manifest "$source_protocol/manifest.json" \
      --model-id "$model_id" \
      --adapter-path "$adapter" \
      --adapter-manifest "$manifest" \
      "${training_args[@]}" \
      --phase "$phase" \
      --output-file "$generation" \
      --device cuda:0 \
      --audit-only
  done
  python "$sampler" \
    --score-benefit \
    --output-file "$output/generation/direct/$model_id/benefit.json" \
    --answers-file "$source_protocol/benefit/answers.json" \
    --score-output "$output/evaluation/benefit/$model_id.json"
done

python "$finalizer" \
  --stage direct_generation \
  --output-root "$output" \
  --repo-root "$repo" \
  --elapsed-seconds "$((SECONDS - started))" \
  --slurm-job-id "$SLURM_JOB_ID"
completed=true
trap - EXIT
echo MASSIVE_MEDICAL_COMPOSITION_BASELINES_V1_DIRECT_GENERATION_COMPLETE
