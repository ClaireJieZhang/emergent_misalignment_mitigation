#!/bin/bash
# CPU-only stage for the MASSIVE medical fixed-k=4 panel-ratio study.

set -euo pipefail
umask 077
ulimit -c 0

[[ $# -eq 0 ]] || {
  echo 'Usage: scripts/stage_massive_medical_ratio_panels_v1_tillicum.sh' >&2
  exit 2
}

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-ratio-panels-v1
python=$root/envs/subliminal-mitigate-py311/bin/python
manager=$repo/scripts/manage_massive_medical_ratio_panels_v1.py

cd "$repo"
test "$(git rev-parse --abbrev-ref HEAD)" = claire/massive-medical-ratio-panels-v1
test -z "$(git status --porcelain=v1 --untracked-files=all)"

PYTHONDONTWRITEBYTECODE=1 "$python" -m unittest \
  tests.test_massive_medical_ratio_panels_v1_protocol \
  tests.test_massive_medical_ratio_panels_v1_training_workflow
PYTHONDONTWRITEBYTECODE=1 "$python" -m py_compile "$manager"
bash -n \
  scripts/stage_massive_medical_ratio_panels_v1_tillicum.sh \
  scripts/submit_massive_medical_ratio_panels_v1_training_tillicum.sh \
  scripts/sbatch_massive_medical_ratio_panels_v1_train_tillicum_h200.sbatch \
  scripts/status_massive_medical_ratio_panels_v1_training_tillicum.sh \
  scripts/finalize_massive_medical_ratio_panels_v1_training_tillicum.sh

"$python" "$manager" stage

echo 'CPU_STAGED_ONLY: zero Slurm jobs, zero GPU minutes, zero API calls.'
echo 'The exact $0.900000 training authorization and submit command remain separate.'
