#!/bin/bash
# Read-only status for the sealed smoke gate recovery v4.

set -euo pipefail

[[ $# -eq 0 ]] || {
  echo 'Usage: scripts/status_massive_medical_union_composition_exploratory_smoke_gate_recovery_v4_tillicum.sh' >&2
  exit 2
}

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-composition-exploratory-smoke-gate-recovery-v4
output=$root/outputs/massive_medical_union_composition_exploratory_smoke_gate_recovery_v4
env_root=$root/envs/subliminal-mitigate-py311
auditor=$repo/scripts/audit_massive_medical_union_composition_exploratory_smoke_gate_recovery_v4.py

if [[ ! -d $repo && ! -e $output ]]; then
  echo 'SMOKE_GATE_RECOVERY_NOT_STAGED'
  exit 0
fi
test -d "$repo"

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY
unset TRANSFORMERS_CACHE CUDA_VISIBLE_DEVICES
module load conda/Miniforge3-25.3.1-3
conda activate "$env_root"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-composition-smoke-gate-recovery-v4-status-pyc
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=$root/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

cd "$repo"
python "$auditor" status
echo 'CONFIRMATION_AUTHORIZED=false'
echo 'CONFIRMATION_SUBMITTED=false'
echo 'EXTERNAL_API_CALLS=0'
echo 'NEW_GPU_H200_MINUTES=0'
